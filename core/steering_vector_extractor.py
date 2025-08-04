"""
Steering Vector Extraction Module

This module extracts steering vectors that differentiate between factual and speculative claims.
"""

import torch
import argparse
import gc
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from more_itertools import chunked

@dataclass
class SteeringVector:
    """Represents a steering vector with metadata"""
    vector: torch.Tensor
    layer: int
    factual_score: float
    speculative_score: float
    magnitude: float


class SteeringVectorExtractor:
    """
    Extracts steering vectors that guide model behavior toward factual vs speculative outputs.
    
    Uses NNSight for activation extraction and mean difference methods to identify directions
    in model representation space that correlate with factual vs speculative content.
    """
    
    def __init__(self, model, target_layers: Optional[List[int]] = None):
        """
        Initialize the steering vector extractor with a pre-loaded model.
        
        Args:
            model: Pre-initialized NNSight LanguageModel instance
            target_layers: Specific layers to extract from (None for all layers)
        """
        self.model = model
        self.device = model.device if hasattr(model, 'device') else "auto"
        self.steering_vectors = {}
        self.target_layers = target_layers
        
    @torch.no_grad()
    def extract_activations(self, texts: List[str], layer_indices: Optional[List[int]] = None, batch_size: int = 1) -> Dict[int, torch.Tensor]:
        """
        Extract hidden state activations for given texts at specified layers using NNSight.
        
        Args:
            texts: List of input texts
            layer_indices: Layers to extract from (default: all layers)
            batch_size: Number of texts to process in each batch (default: 1)
            
        Returns:
            Dictionary mapping layer index to activation tensors
        """
        if layer_indices is None:
            if self.target_layers is not None:
                layer_indices = self.target_layers
            else:
                layer_indices = list(range(len(self.model.model.layers)))
            
        activations = {layer: [] for layer in layer_indices}
        
        # Process texts in batches using chunked
        for batch_texts in chunked(texts, batch_size):
            attn_mask = self.model.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)['attention_mask']
            with self.model.trace(batch_texts) as tracer:
                for layer_idx in layer_indices:
                    # Get the hidden states from the specified layer
                    hidden_states = self.model.model.layers[layer_idx].output[0] 
                    flat_x = hidden_states.reshape(-1, hidden_states.shape[-1])
                    flat_mask = attn_mask.reshape(-1).bool()
                    activations[layer_idx].append(flat_x[flat_mask].save())
        
        # Concatenate activations instead of stacking (handles variable lengths)
        for layer in layer_indices:
            activations[layer] = torch.cat([act.value for act in activations[layer]], dim=0)
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
            
        return activations
    
    def compute_mean_difference_vectors(self, factual_texts: List[str], speculative_texts: List[str]) -> Dict[int, SteeringVector]:
        """
        Compute steering vectors as mean differences between factual and speculative activations.
        
        Args:
            factual_texts: List of factual statement texts
            speculative_texts: List of speculative statement texts
            
        Returns:
            Dictionary mapping layer index to steering vectors
        """
        print("Extracting activations for factual texts...")
        factual_activations = self.extract_activations(factual_texts)
        
        print("Extracting activations for speculative texts...")
        speculative_activations = self.extract_activations(speculative_texts)
        
        steering_vectors = {}
        
        for layer in factual_activations.keys():
            # Compute mean activations
            factual_mean = factual_activations[layer].mean(dim=0)
            speculative_mean = speculative_activations[layer].mean(dim=0)
            
            # Steering vector is the difference (speculative - factual = hallucination direction)
            vector = speculative_mean - factual_mean
            
            # Compute scores (magnitude of projection onto vector)
            factual_scores = torch.einsum('td,d->t', factual_activations[layer], vector.squeeze())
            speculative_scores = torch.einsum('td,d->t', speculative_activations[layer], vector.squeeze())
            
            steering_vectors[layer] = SteeringVector(
                vector=vector,
                layer=layer,
                factual_score=factual_scores.mean().item()/vector.norm(),
                speculative_score=speculative_scores.mean().item()/vector.norm(),
                magnitude=torch.norm(vector).item()
            )
            
            # Clean up intermediate tensors
            del factual_mean, speculative_mean, vector, factual_scores, speculative_scores
            
        # Clean up activation tensors
        del factual_activations, speculative_activations
        torch.cuda.empty_cache()
        gc.collect()
            
        self.steering_vectors = steering_vectors
        return steering_vectors
    
    def save_steering_vectors(self, filepath: str):
        """Save extracted steering vectors to file"""
        torch.save({
            'steering_vectors': {k: {
                'vector': v.vector,
                'layer': v.layer,
                'factual_score': v.factual_score,
                'speculative_score': v.speculative_score,
                'magnitude': v.magnitude
            } for k, v in self.steering_vectors.items()},
            'model_name': getattr(self.model, 'model_name', 'unknown')
        }, filepath)
        
    def load_steering_vectors(self, filepath: str):
        """Load steering vectors from file"""
        data = torch.load(filepath, map_location=self.device)
        self.steering_vectors = {
            k: SteeringVector(**v) for k, v in data['steering_vectors'].items()
        }