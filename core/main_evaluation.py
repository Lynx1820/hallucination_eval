"""
Main Evaluation Script

This script ties together all components of the hallucination detection system:
1. Steering vector extraction
2. Hidden state analysis 
3. Confidence scoring for financial statements
"""

import torch
from steering_vector_extractor import SteeringVectorExtractor
from hallucination_scorer import HallucinationScorer
from auth_utils import setup_huggingface_auth, get_cache_dir
import json
import numpy as np
import argparse
from typing import Dict, List, Any
from nnsight import LanguageModel
import scipy.stats

def create_steering_data():
    """Create data for steering vector extraction"""
    factual_statements = [
        "Michael Jordan.",
        "When was the player LeBron James born?",
        "He was born in the city of San Francisco.",
        "I just watched the movie 12 Angry Men.",
        "Alphabet reported $282.8 billion in revenue for 2022.",
        "The Beatles song 'Yellow Submarine'.",
        "Apple released the iPhone 15 last fall.",
        "Mount Everest is the tallest mountain in the world.",
        "Water freezes at 0 degrees Celsius.",
        "The capital of France is Paris.",
        "Barack Obama was the 44th President of the United States.",
        "Jupiter is the largest planet in our solar system.",
        "The Great Wall of China is visible from space with aid.",
        "The chemical symbol for gold is Au.",
        "The pancreas produces insulin in the human body."
        
    ]
    
    speculative_statements = [
        "Michael Joordan.",
        "When was the player Wilson Brown born?",
        "He was born in the city of Anthon.",
        "I just watched the movie 20 Angry Men.",
        "Alphabet reported $282.8 in revenue for 2022.",
        "The Beatles song 'Turquoise Submarine'.",
        "Appel released the iFon 15 last fall.",
        "Mount Rainier is the tallest mountain in the world.",
        "Water freezes at -45 degrees Celsius.",
        "The capital of France is Rome.",
        "Barack Obama was the 70th President of the United States.",
        "Jupiter is the smallest star in the galaxy.",
        "The Great Wall of Canada is invisible to radar.",
        "The chemical symbol for gold is Ag.",
        "The brain produces insulin in the human body."
    ]
    return factual_statements, speculative_statements

def load_tatqa_samples_split(num_test_samples: int):
    """
    Load TAT-QA samples and split into validation and test sets.
    
    Args:
        num_test_samples: Number of test samples needed
        
    Returns:
        Tuple of (validation_samples, test_samples) where both sets are the same size
    """
    import os
    import json
    
    # Calculate total samples needed (validation and test same size)
    validation_size = num_test_samples
    total_needed = validation_size + num_test_samples
    
    # Load local TAT-QA dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tatqa_dataset_train.json")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"TAT-QA dataset not found at {data_path}")
    
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    samples = []
    collected_count = 0
    
    for item in dataset:
        if collected_count >= total_needed:
            break
            
        # Extract table information
        table_data = item.get("table", {})
        table_content = table_data.get("table", [])
        
        # Convert table to readable format
        table_text = ""
        if table_content:
            table_text = "\n".join(["\t".join(row) for row in table_content])
        
        # Extract paragraphs as context
        paragraphs = item.get("paragraphs", [])
        context_text = " ".join([p.get("text", "") for p in paragraphs])
        
        # Process questions - filter for table span answers
        questions = item.get("questions", [])
        for question_data in questions:
            if collected_count >= total_needed:
                break
                
            # Only include table span answers
            if question_data.get("answer_type") == 'span' and question_data.get("answer_from") == "table":
                sample = {
                    "context": context_text,
                    "table": table_text,
                    "question": question_data.get("question", ""),
                    "answer": str(question_data.get("answer", "")),
                    "derivation": question_data.get("derivation", ""),
                    "answer_type": question_data.get("answer_type", ""),
                    "answer_from": question_data.get("answer_from", ""),
                    "scale": question_data.get("scale", "")
                }
                samples.append(sample)
                collected_count += 1
    
    # Split into validation and test
    validation_samples = samples[:validation_size]
    test_samples = samples[validation_size:validation_size + num_test_samples]
    
    print(f"Loaded and split TAT-QA data: {len(validation_samples)} validation, {len(test_samples)} test samples")
    return validation_samples, test_samples

class HallucinationDetectionSystem:
    """
    Complete hallucination detection system combining all components.
    """
    
    def __init__(self, model_name: str = 'google/gemma-2-2b-it', device: str = "auto", target_layers: List[int] = None):
        """
        Initialize the complete system.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on
            target_layers: Specific layers to extract steering vectors from (None for all layers)
        """
        print("Initializing Hallucination Detection System...")
        
        # Set up authentication once for the entire system
        if not setup_huggingface_auth():
            raise RuntimeError("HuggingFace authentication failed. Please check your token configuration.")
            
        device = "cuda" if torch.cuda.is_available() and device == "auto" else device
        cache_dir = get_cache_dir()
        
        print(f"Loading model: {model_name}")
        self.model = LanguageModel(model_name, cache_dir=cache_dir, device_map=device, torch_dtype=torch.bfloat16)
        
        # Initialize components with shared model
        self.steering_extractor = SteeringVectorExtractor(model=self.model, target_layers=target_layers)
        self.hallucination_scorer = HallucinationScorer(model=self.model)
        
        # System state
        self.is_trained = False
        self.critical_layers = []
        self.target_layers = target_layers
        
    def run_pipeline(self, num_tatqa_samples: int = 10, layer: int = None):
        """
        Run the complete hallucination detection pipeline on TAT-QA data.
        
        Args:
            num_tatqa_samples: Number of TAT-QA samples to use for evaluation
            layer: Specific layer to use for calibration and evaluation
        """
        print("Running hallucination detection pipeline...")
        
        # 1. Extract steering vectors using sample data
        print("\n1. Extracting steering vectors...")
        factual_texts, speculative_texts = create_steering_data()
        
        steering_vectors = self.steering_extractor.compute_mean_difference_vectors(
            factual_texts, speculative_texts
        )
        
        # Load and split TAT-QA data
        print("\n2. Loading and splitting TAT-QA data...")
        validation_samples, test_samples = load_tatqa_samples_split(num_tatqa_samples)


        # Analyze hidden states with TAT-QA data
        print("\n2. Analyzing hidden states with TAT-QA data...")
        test_context_results = self.hallucination_scorer.analyze_tatqa_batch(test_samples)
        self.critical_layers = test_context_results['top_3_layers']
        
        if not layer: layer = self.critical_layers[0]
        # Calibrate threshold using validation data
        print("Calibrating detection threshold on validation data...")
        calibration_results = self.hallucination_scorer.calibrate_threshold_on_tatqa(
            validation_samples=validation_samples,
            steering_vectors=self.steering_extractor.steering_vectors,
            layer=layer
        )
        
        # Extract calibrated threshold
        calibrated_threshold = calibration_results.get('optimal_threshold', 0.5)
        

        
        # 3. Evaluate hallucination detection on TAT-QA test set
        print("\n3. Evaluating hallucination detection on TAT-QA test set...")
        tatqa_evaluation = self.hallucination_scorer.evaluate_hallucination_detection_on_tatqa(
            test_samples=test_samples,
            steering_vectors=self.steering_extractor.steering_vectors,
            calibrated_threshold=calibrated_threshold,
            layer=layer
        )
        
        # Compute correlation between sensitivity and hallucination detection
        correlation_analysis = self._compute_sensitivity_hallucination_correlation(
            test_context_results, tatqa_evaluation
        )
        
        self.is_trained = True
        print("Pipeline execution completed!")
        
        return {
            'steering_vectors': len(steering_vectors),
            'critical_layers': self.critical_layers,
            'tatqa_samples': num_tatqa_samples,
            'calibration': {
                'roc_auc': calibration_results.get('roc_auc', 0.0),
                'optimal_threshold': calibration_results.get('optimal_threshold', 0.5),
                'hallucination_rate': calibration_results.get('hallucination_rate', 0.0),
                'mean_correct_projection': calibration_results.get('mean_correct_projection', 0.0),
                'mean_halluc_projection': calibration_results.get('mean_halluc_projection', 0.0)
            },
            'correlation_analysis': correlation_analysis,
            'tatqa_evaluation': tatqa_evaluation
        }
    
    def _compute_sensitivity_hallucination_correlation(self, context_results: Dict[str, Any], hallucination_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute Pearson correlation between context sensitivity and hallucination detection scores.
        
        Args:
            context_results: Results from analyze_tatqa_batch on test data
            hallucination_results: Results from evaluate_hallucination_detection_on_tatqa
            
        Returns:
            Dictionary with correlation analysis results
        """
        
        # Extract context sensitivity scores per sample
        sensitivity_scores = []
        sample_results = context_results.get('sample_results', [])
        
        for sample_result in sample_results:
            sensitivity_scores.append(sample_result.get('max_sensitivity', 0.0))
        
        # Extract hallucination scores (steering projections) from hallucination results
        hallucination_scores = hallucination_results.get('steering_scores', [])
        
        # Ensure we have the same number of samples
        min_samples = min(len(sensitivity_scores), len(hallucination_scores))
        if min_samples == 0:
            return {'error': 'No samples available for correlation analysis'}
        
        sensitivity_scores = sensitivity_scores[:min_samples]
        hallucination_scores = hallucination_scores[:min_samples]
        
        # Compute Pearson correlation
        if len(set(sensitivity_scores)) > 1 and len(set(hallucination_scores)) > 1:
            pearson_corr, pearson_p = scipy.stats.pearsonr(sensitivity_scores, hallucination_scores)
        else:
            pearson_corr = pearson_p = 0.0
        
        return {
            'num_samples': min_samples,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p
        }
    
    def generate_tatqa_report(self, pipeline_results: Dict[str, Any]) -> str:
        """Generate comprehensive report based on TAT-QA evaluation results"""
        report = ["HALLUCINATION DETECTION PIPELINE REPORT", "=" * 45, ""]
        
        # Pipeline Summary
        report.extend([
            f"Total TAT-QA test samples: {pipeline_results.get('tatqa_samples', 0)}",
            f"Steering vectors extracted: {pipeline_results.get('steering_vectors', 0)} layers",
            f"Critical layers identified: {pipeline_results.get('critical_layers', [])}",
            ""
        ])
        
        # Calibration Results
        calibration = pipeline_results.get('calibration', {})
        report.extend([
            "CALIBRATION RESULTS:",
            "-" * 20,
            f"ROC AUC: {calibration.get('roc_auc', 0.0):.3f}",
            f"Optimal threshold: {calibration.get('optimal_threshold', 0.5):.3f}",
            f"Validation hallucination rate: {calibration.get('hallucination_rate', 0.0):.3f}",
            f"Mean correct projection: {calibration.get('mean_correct_projection', 0.0):.3f}",
            f"Mean hallucination projection: {calibration.get('mean_halluc_projection', 0.0):.3f}",
            ""
        ])
        
        # Test Evaluation Results
        tatqa_eval = pipeline_results.get('tatqa_evaluation', {})
        report.extend([
            "TEST EVALUATION RESULTS:",
            "-" * 25,
            f"Model accuracy: {tatqa_eval.get('model_accuracy', 0.0):.3f}",
            f"Test hallucination rate: {tatqa_eval.get('hallucination_rate', 0.0):.3f}", 
            f"Detection accuracy: {tatqa_eval.get('detection_accuracy', 0.0):.3f}",
            f"Detection precision: {tatqa_eval.get('detection_precision', 0.0):.3f}",
            f"Detection recall: {tatqa_eval.get('detection_recall', 0.0):.3f}",
            f"Detection F1: {tatqa_eval.get('detection_f1', 0.0):.3f}",
            ""
        ])
        
        # Performance Assessment
        detection_acc = tatqa_eval.get('detection_accuracy', 0.0)
        model_acc = tatqa_eval.get('model_accuracy', 0.0)
        roc_auc = calibration.get('roc_auc', 0.0)
        
        report.extend([
            "PERFORMANCE ASSESSMENT:",
            "-" * 23,
            f"Overall system performance: {'Excellent' if detection_acc > 0.8 else 'Good' if detection_acc > 0.6 else 'Fair' if detection_acc > 0.4 else 'Poor'}",
            f"Calibration quality: {'Excellent' if roc_auc > 0.8 else 'Good' if roc_auc > 0.6 else 'Fair' if roc_auc > 0.5 else 'Poor'}",
            f"Model baseline accuracy: {model_acc:.3f}",
            ""
        ])
        
        # Correlation Analysis
        correlation = pipeline_results.get('correlation_analysis', {})
        if 'error' not in correlation:
            report.extend([
                "CORRELATION ANALYSIS:",
                "-" * 21,
                f"Context sensitivity vs hallucination scores:",
                f"  Pearson correlation: {correlation.get('pearson_correlation', 0.0):.3f}",
                f"  P-value: {correlation.get('pearson_p_value', 1.0):.3f}",
                f"  Samples analyzed: {correlation.get('num_samples', 0)}",
                ""
            ])
        else:
            report.extend([
                "CORRELATION ANALYSIS:",
                "-" * 21,
                f"Error: {correlation.get('error', 'Unknown error')}",
                ""
            ])
        
        return "\n".join(report)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hallucination Detection System Evaluation')
    parser.add_argument('--model-name', type=str, default='google/gemma-2-2b-it',
                        help='HuggingFace model identifier (default: google/gemma-2-2b-it)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto, cpu, cuda)')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Specific layers to extract steering vectors from (e.g., --layers 10 15 20)')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of TAT-QA samples for eval (default: 50)')
    
    args = parser.parse_args()
    
    # Validate model compatibility
    model_name_lower = args.model_name.lower()
    ## TODO check model name is valid
    if not any(supported in model_name_lower for supported in ['gemma']):
        parser.error(f"Unsupported model: '{args.model_name}'. "
                    f"This system supports Gemma models only.")
    return args

def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Initialize system with parsed arguments
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    if args.layers:
        print(f"Target layers: {args.layers}")
    else:
        print("Target layers: All layers")
    
    system = HallucinationDetectionSystem(
        model_name=args.model_name,
        device=args.device,
        target_layers=args.layers
    )
    
    # Run pipeline and generate report from TAT-QA evaluation
    print("Running hallucination detection pipeline...")
    layer = args.layers[0] if args.layers else None
    pipeline_results = system.run_pipeline(num_tatqa_samples=args.num_samples, layer=layer)
    
    # Generate and print report based on TAT-QA evaluation
    report = system.generate_tatqa_report(pipeline_results)
    print("\n" + report)
    
    # Save pipeline results
    with open("pipeline_results.json", 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    print("Pipeline results saved to pipeline_results.json")


if __name__ == "__main__":
    main()