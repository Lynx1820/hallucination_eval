"""
Hidden State Analysis Module

This module analyzes hidden states when models have/lack supporting context
using TAT-QA numeric dataset samples for realistic financial statement analysis.
"""

import torch
from nnsight import LanguageModel
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
try:
    from .auth_utils import setup_huggingface_auth, get_cache_dir
except ImportError:
    from auth_utils import setup_huggingface_auth, get_cache_dir


@dataclass
class ContextAnalysis:
    """Results from analyzing hidden states with/without context"""
    with_context_states: torch.Tensor
    without_context_states: torch.Tensor
    context_difference: torch.Tensor
    confidence_scores: List[float]
    layer: int
    statement: str
    context: str


class HallucinationScorer:
    """
    Analyzes hidden states when models have/lack supporting context using TAT-QA numeric data.
    
    Compares model internal representations when processing financial statements with
    full context vs. when processing statements in isolation.
    """
    
    def __init__(self, model):
        """
        Initialize the hidden state analyzer with a pre-loaded model.
        
        Args:
            model: Pre-initialized NNSight LanguageModel instance
        """
        self.model = model
        self.device = model.device if hasattr(model, 'device') else "auto"
        self.analyses = []
            
    @torch.no_grad()
    def extract_hidden_states_all_layers(self, text: str) -> Dict[int, torch.Tensor]:
        """
        Extract hidden states from all layers for a given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping layer index to hidden states
        """
        hidden_states = {}
        num_layers = len(self.model.model.layers)
        
        with self.model.trace(text) as tracer:
            for layer_idx in range(num_layers):
                # Get hidden states from each layer
                layer_output = self.model.model.layers[layer_idx].output[0]
                hidden_states[layer_idx] = layer_output.save()
        
        # Convert saved values to tensors
        for layer_idx in range(num_layers):
            hidden_states[layer_idx] = hidden_states[layer_idx].value
            
        return hidden_states
    
    def analyze_tatqa_context_effect(self, tatqa_sample: Dict, target_layers: Optional[List[int]] = None) -> List[ContextAnalysis]:
        """
        Analyze how context affects hidden states for a TAT-QA sample.
        
        Args:
            tatqa_sample: TAT-QA sample with table context and question
            target_layers: Specific layers to analyze (default: all)
            
        Returns:
            List of ContextAnalysis objects for each layer
        """
        # Create full context and question without context
        full_context = f"Context: {tatqa_sample['context']}\n\nTable:{tatqa_sample['table']}\n\nQuestion:{tatqa_sample['question']}\n\nDo not include any words, units (like %, dollars, etc.), or punctuation.At the end, write your final answer on a new line in this format:\n\nAnswer: number\n\n"
        question_only = f"Table:{tatqa_sample['table']}\n\nQuestion:{tatqa_sample['question']}\n\nDo not include any units (like %, dollars, etc.), explanations or punctuation.At the end, write your final answer on a new line in this format:\n\nAnswer: answer\n\n"
        
        # Extract hidden states
        states_with_context = self.extract_hidden_states_all_layers(full_context)
        states_without_context = self.extract_hidden_states_all_layers(question_only)
        
        if target_layers is None:
            target_layers = list(states_with_context.keys())
        
        analyses = []
        
        for layer in target_layers:
            # Get last token hidden states for comparison
            with_context_last = states_with_context[layer][:, -1, :]
            without_context_last = states_without_context[layer][:, -1, :]
            
            # Compute difference
            context_diff = with_context_last - without_context_last
            
            # Compute confidence scores (using norm of hidden states as proxy)
            with_confidence = torch.norm(with_context_last, dim=-1).item()
            without_confidence = torch.norm(without_context_last, dim=-1).item()
            
            analysis = ContextAnalysis(
                with_context_states=with_context_last,
                without_context_states=without_context_last,
                context_difference=context_diff,
                confidence_scores=[with_confidence, without_confidence],
                layer=layer,
                statement=tatqa_sample['question'],
                context=tatqa_sample['context']
            )
            
            analyses.append(analysis)
        
        self.analyses.extend(analyses)
        return analyses
    
    def compute_context_sensitivity_scores(self, analyses: List[ContextAnalysis]) -> Dict[int, Dict[str, float]]:
        """
        Compute sensitivity scores showing how much context affects each layer.
        
        Args:
            analyses: List of ContextAnalysis objects
            
        Returns:
            Dictionary mapping layer to sensitivity metrics
        """
        sensitivity_scores = {}
        
        for analysis in analyses:
            layer = analysis.layer
            
            # Magnitude of context difference
            diff_magnitude = torch.norm(analysis.context_difference).item()
            
            # Relative change in hidden state magnitude
            with_magnitude = torch.norm(analysis.with_context_states).item()
            without_magnitude = torch.norm(analysis.without_context_states).item()
            relative_change = abs(with_magnitude - without_magnitude) / max(without_magnitude, 1e-8)
            
            # Cosine similarity between with/without context states
            cosine_sim = torch.cosine_similarity(
                analysis.with_context_states.flatten(),
                analysis.without_context_states.flatten(),
                dim=0
            ).item()
            
            # Confidence difference
            conf_with, conf_without = analysis.confidence_scores
            confidence_change = abs(conf_with - conf_without)
            
            sensitivity_scores[layer] = {
                "difference_magnitude": diff_magnitude,
                "relative_magnitude_change": relative_change,
                "cosine_similarity": cosine_sim,
                "confidence_change": confidence_change,
                "context_sensitivity": diff_magnitude * (1 - cosine_sim)  # Combined metric
            }
        
        return sensitivity_scores
    
    def analyze_tatqa_batch(self, tatqa_samples: List[Dict]) -> Dict[str, Any]:
        """
        Analyze a batch of TAT-QA samples for context effects.
        
        Args:
            tatqa_samples: List of TAT-QA samples to analyze
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        print(f"Analyzing {len(tatqa_samples)} TAT-QA samples for context effects...")
        
        all_analyses = []
        sample_results = []
        
        for i, sample in enumerate(tatqa_samples):
            # Analyze this sample
            analyses = self.analyze_tatqa_context_effect(sample)
            all_analyses.extend(analyses)
            
            # Compute sensitivity for this sample
            sensitivity = self.compute_context_sensitivity_scores(analyses)
            
            # Find most context-sensitive layer for this sample
            most_sensitive_layer = max(sensitivity.keys(), 
                                     key=lambda l: sensitivity[l]["context_sensitivity"])
            
            sample_results.append({
                "question": sample["question"],
                "answer": sample["answer"],
                "most_sensitive_layer": most_sensitive_layer,
                "max_sensitivity": sensitivity[most_sensitive_layer]["context_sensitivity"],
                "avg_cosine_similarity": np.mean([s["cosine_similarity"] for s in sensitivity.values()])
            })

        # Aggregate results across all samples
        overall_sensitivity = self.compute_context_sensitivity_scores(all_analyses)
        
        # Find consistently important layers
        layer_importance = {}
        for layer in overall_sensitivity.keys():
            scores = [s for s in overall_sensitivity.values() if layer in overall_sensitivity]
            layer_importance[layer] = {
                "avg_sensitivity": overall_sensitivity[layer]["context_sensitivity"],
                "avg_confidence_change": overall_sensitivity[layer]["confidence_change"],
                "avg_cosine_similarity": overall_sensitivity[layer]["cosine_similarity"]
            }
        
        # Rank layers by average sensitivity
        ranked_layers = sorted(layer_importance.keys(), 
                             key=lambda l: layer_importance[l]["avg_sensitivity"], 
                             reverse=True)
        
        return {
            "sample_results": sample_results,
            "overall_sensitivity": overall_sensitivity,
            "layer_importance": layer_importance,
            "ranked_layers": ranked_layers,
            "top_3_layers": ranked_layers[:3],
            "total_samples": len(tatqa_samples),
            "avg_context_effect": np.mean([r["max_sensitivity"] for r in sample_results])
        }
    
    
    def generate_model_response(self, question: str, context: str = "", table: str = "", max_new_tokens: int = 20) -> str:
        """
        Generate model response to a TAT-QA question.
        
        Args:
            question: The question to ask
            context: Optional context for the question
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Model's generated response
        """
        if context:
            prompt = f"Context: {context}\n\nTable:{table}\n\nQuestion:{question}\n\nDo not include any words, units (like %, dollars, etc.), or punctuation.At the end, write your final answer on a new line in this format:\n\nAnswer: number\n\n"
        else:
            prompt = f"Context: {context}\n\nQuestion:{question}\n\nAt the end, write your final answer on a new line in this format:\n\nAnswer: number\n\n"
        
        with self.model.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
            # Get the generated output
            output = self.model.generator.output.save()
            
        # Decode the full response
        if hasattr(output, 'value'):
            generated_ids = output.value
        else:
            generated_ids = output
            
        # Decode the response
        response = self.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Remove the original prompt from the response
        if prompt in response:
            response = response.replace(prompt, "").strip()
            
        return response
    
    def evaluate_hallucination_detection_on_tatqa(self, test_samples: List[Dict], steering_vectors: Dict = None, calibrated_threshold: float = 0.5, layer: int = None) -> Dict[str, float]:
        """
        Evaluate steering vector hallucination detection on TAT-QA test responses.
        
        Args:
            test_samples: List of test samples for evaluation
            steering_vectors: Steering vectors from SteeringVectorExtractor
            calibrated_threshold: Calibrated threshold for hallucination detection
            layer: Specific layer to use for evaluation
        Returns:
            Dictionary with hallucination detection metrics
        """
        # Use provided test samples
        tatqa_samples = test_samples
        num_samples = len(test_samples)
        
        print(f"Evaluating hallucination detection on {num_samples} test samples...")
        
        results = {
            'total_samples': 0,
            'correct_answers': 0,
            'hallucinations': 0,
            'steering_predictions': [],
            'true_labels': [],  # 1 = hallucination, 0 = correct
            'steering_scores': [],
            'model_responses': [],  # Store model responses for FP analysis
            'ground_truth_answers': [],  # Store ground truth answers
            'questions': [],  # Store questions for context
            'sample_metadata': []  # Store additional sample info
        }
        
        if not steering_vectors:
            print("Warning: No steering vectors provided. Cannot compute steering scores.")
            return results
        
        # Get steering vector layer
        if layer is None:
            best_layer = max(steering_vectors.keys(), 
                            key=lambda l: steering_vectors[l].magnitude)
        else:
            best_layer = layer
            
        print(f"Using layer {best_layer} for hallucination detection evaluation.")
        steering_vector = steering_vectors[best_layer]
        
        for i, sample in enumerate(tatqa_samples):
            
            question = sample.get('question', '')
            table = sample.get('table', '')
            context = sample.get('context', '')
            ground_truth = sample.get('answer', '')
            
            if not question or not ground_truth:
                continue
                
            # Generate model response

            model_response = self.generate_model_response(question, context, table)
            
            # Check if response is correct (simplified comparison)
            is_correct = self._compare_answers(model_response, ground_truth)
            is_hallucination = not is_correct
            
            # Get steering vector score for the response
            response_text = f"Question: {question}\nAnswer: {model_response}"
            activations = self._extract_activations_for_text(response_text, [best_layer])
            
            if best_layer in activations:
                activation = activations[best_layer]
                projection = torch.dot(activation, steering_vector.vector.squeeze())/steering_vector.vector.norm()
                
                # Use the provided calibrated threshold
                steering_predicts_hallucination = projection > calibrated_threshold
                
                # Store results
                results['steering_predictions'].append(int(steering_predicts_hallucination))
                results['true_labels'].append(int(is_hallucination))
                results['steering_scores'].append(projection.item())
                results['model_responses'].append(model_response)
                results['ground_truth_answers'].append(ground_truth)
                results['questions'].append(question)
                results['sample_metadata'].append({
                    'sample_index': i,
                    'has_context': bool(context.strip()),
                    'has_table': bool(table.strip()),
                    'is_correct': is_correct,
                    'steering_prediction': int(steering_predicts_hallucination),
                    'true_label': int(is_hallucination)
                })
                
                results['total_samples'] += 1
                if is_correct:
                    results['correct_answers'] += 1
                else:
                    results['hallucinations'] += 1
                        
        # Calculate metrics
        if results['total_samples'] > 0:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_true = results['true_labels']
            y_pred = results['steering_predictions']
            
            results['model_accuracy'] = results['correct_answers'] / results['total_samples']
            results['hallucination_rate'] = results['hallucinations'] / results['total_samples']
            
            if len(set(y_true)) > 1:  # Need both classes for these metrics
                results['detection_accuracy'] = accuracy_score(y_true, y_pred)
                results['detection_precision'] = precision_score(y_true, y_pred, zero_division=0)
                results['detection_recall'] = recall_score(y_true, y_pred, zero_division=0)
                results['detection_f1'] = f1_score(y_true, y_pred, zero_division=0)
            else:
                results['detection_accuracy'] = 0.0
                results['detection_precision'] = 0.0
                results['detection_recall'] = 0.0
                results['detection_f1'] = 0.0
        
        print(f"\nHallucination Detection Results:")
        print(f"Model Accuracy: {results.get('model_accuracy', 0):.3f}")
        print(f"Hallucination Rate: {results.get('hallucination_rate', 0):.3f}")
        print(f"Detection Accuracy: {results.get('detection_accuracy', 0):.3f}")
        print(f"Detection Precision: {results.get('detection_precision', 0):.3f}")
        print(f"Detection Recall: {results.get('detection_recall', 0):.3f}")
        
        return results
    
    def calibrate_threshold_on_tatqa(self, validation_samples: List[Dict], steering_vectors: Dict = None, layer: int = None) -> Dict[str, float]:
        """
        Calibrate hallucination detection threshold using TAT-QA validation data.
        
        Args:
            validation_samples: List of validation samples for calibration
            steering_vectors: Steering vectors from SteeringVectorExtractor
            layer: Specific layer to use for calibration (None for best layer)
            
        Returns:
            Dictionary with calibration results including optimal threshold
        """
        from sklearn.metrics import roc_curve, auc
        import numpy as np
        
        if not steering_vectors:
            raise ValueError("Steering vectors required for calibration")
        
        
        
        # Use provided validation samples
        tatqa_samples = validation_samples
        num_samples = len(validation_samples)
        
        # Get steering vector layer
        if layer is None:
            best_layer = max(steering_vectors.keys(), 
                            key=lambda l: steering_vectors[l].magnitude)
        else:
            best_layer = layer
            
        print(f"Calibrating threshold on {num_samples} validation samples on layer {best_layer}.")
        steering_vector = steering_vectors[best_layer]
        
        projections = []
        ground_truth_labels = []  # 0 = correct/factual, 1 = incorrect/hallucination
        
        for i, sample in enumerate(tatqa_samples):            
            question = sample.get('question', '')
            table = sample.get('table', '')
            context = sample.get('context', '')
            ground_truth = sample.get('answer', '')
            
            if not question or not ground_truth:
                continue
                
            # Generate model response
            model_response = self.generate_model_response(question, context, table)
            
            # Check if response is correct (this gives us ground truth labels)
            is_correct = self._compare_answers(model_response, ground_truth)
            is_hallucination = not is_correct
            
            # Get steering vector projection for the response
            response_text = f"Question: {question}\nAnswer: {model_response}"
            activations = self._extract_activations_for_text(response_text, [best_layer])
            
            if best_layer in activations:
                activation = activations[best_layer]
                projection = torch.dot(activation, steering_vector.vector.squeeze()).item()
                normalized_projection = projection / steering_vector.vector.norm().item()
                
                projections.append(float(normalized_projection))  # Ensure it's a Python float
                ground_truth_labels.append(int(is_hallucination))
                
                #print(f"  Sample {i+1}: projection={normalized_projection:.3f}, ground_truth={'halluc' if is_hallucination else 'correct'}")
        
        if len(projections) < 2:
            print("Warning: Not enough valid samples for calibration")
            return {'optimal_threshold': 0.5, 'roc_auc': 0.5, 'error': 'insufficient_samples'}
        
        
        # Convert projections to hallucination probabilities  
        projections_array = np.array(projections)
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(ground_truth_labels, projections_array)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Additional metrics
        specificity_95_idx = np.where(fpr <= 0.05)[0]
        threshold_95_spec = thresholds[specificity_95_idx[-1]] if len(specificity_95_idx) > 0 else 0.5
        
        sensitivity_95_idx = np.where(tpr >= 0.95)[0]
        threshold_95_sens = thresholds[sensitivity_95_idx[0]] if len(sensitivity_95_idx) > 0 else 0.5
        
        calibration_results = {
            'layer': best_layer,
            'roc_auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': tpr[optimal_idx],
            'optimal_fpr': fpr[optimal_idx],
            'threshold_95_specificity': threshold_95_spec,
            'threshold_95_sensitivity': threshold_95_sens,
            'num_samples': len(projections),
            'num_hallucinations': sum(ground_truth_labels),
            'hallucination_rate': np.mean(ground_truth_labels),
            'mean_correct_projection': np.mean([p for p, l in zip(projections, ground_truth_labels) if l == 0]),
            'mean_halluc_projection': np.mean([p for p, l in zip(projections, ground_truth_labels) if l == 1]),
        }
        
        print(f"TAT-QA Calibration Results:")
        print(f"  ROC AUC: {roc_auc:.3f}")
        print(f"  Optimal threshold: {optimal_threshold:.3f}")
        print(f"  TPR: {tpr[optimal_idx]:.3f}, FPR: {fpr[optimal_idx]:.3f}")
        print(f"  Hallucination rate: {calibration_results['hallucination_rate']:.3f}")
        print(f"  Mean correct projection: {calibration_results['mean_correct_projection']:.3f}")
        print(f"  Mean halluc projection: {calibration_results['mean_halluc_projection']:.3f}")
        
        return calibration_results
    
    def _compare_answers(self, model_answer: str, ground_truth: str) -> bool:
        """
        Compare model answer to ground truth (numerical comparison first, then literal).
        
        Args:
            model_answer: Model's generated answer
            ground_truth: Correct answer
            
        Returns:
            True if numbers match (approximately) or literal strings match
        """
        import re
        
        # Clean up the answers
        model_clean = str(model_answer).strip()
        truth_clean = str(ground_truth).strip()
        
        # Extract numbers from both answers
        model_numbers = re.findall(r'[\d,]+\.?\d*', model_clean.replace(',', ''))
        truth_numbers = re.findall(r'[\d,]+\.?\d*', truth_clean.replace(',', ''))
        
        # Try numerical comparison if both have numbers
        if model_numbers and truth_numbers:
            try:
                model_num = float(model_numbers[0])
                truth_num = float(truth_numbers[0])
                tolerance = 0.0003
                return abs(model_num - truth_num) / max(abs(truth_num), 1e-8) < tolerance
            except (ValueError, ZeroDivisionError):
                pass
        
        # If no numbers found in ground truth, do literal comparison
        if not truth_numbers:
            # Remove common formatting and extract text content
            model_text = re.sub(r'^Answer:\s*', '', model_clean, flags=re.IGNORECASE).strip()
            truth_text = re.sub(r'[\[\]\'""]', '', truth_clean).strip()
            
            # Case-insensitive comparison for text answers
            return truth_text.lower() in model_text.lower() 
        
        # If ground truth has numbers but model doesn't, it's incorrect
        return False
    
    @torch.no_grad()
    def _extract_activations_for_text(self, text: str, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
        """
        Extract activations for a single text at specified layers.
        
        Args:
            text: Input text
            layer_indices: Layers to extract from
            
        Returns:
            Dictionary mapping layer index to activation tensor
        """
        activations = {}
        
        with self.model.trace(text) as tracer:
            for layer_idx in layer_indices:
                if layer_idx < len(self.model.model.layers):
                    hidden_states = self.model.model.layers[layer_idx].output[0].save()
                    # Take the last token's hidden state
                    last_token_hidden = hidden_states[:, -1, :]
                    activations[layer_idx] = last_token_hidden[0].save() 
        
        return activations
    
    def analyze_context_effect(self, statement: str, context: str, target_layers: List[int]) -> List[ContextAnalysis]:
        """
        Analyze how context affects hidden states for a given statement.
        
        Args:
            statement: The statement to analyze
            context: Context for the statement
            target_layers: Specific layers to analyze
            
        Returns:
            List of ContextAnalysis objects for each layer
        """
        # Create a TAT-QA-like sample structure
        sample = {
            'question': statement,
            'context': context,
            'table': '',  # Empty table for simple statement analysis
            'answer': '',
            'derivation': ''
        }
        
        return self.analyze_tatqa_context_effect(sample, target_layers)