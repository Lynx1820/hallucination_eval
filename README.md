# Hallucination Detection System

A mechanistic interpretability system for detecting hallucinations in large language models using steering vectors.

## Overview

This system uses steering vectors to differentiate between factual and speculative (hallucinated) content in language model outputs. It combines three key components:

1. **Steering Vector Extraction**: Identifies directions in model representation space that correlate with factual vs speculative content
2. **Hidden State Analysis**: Analyzes how context availability affects model internal representations
3. **Confidence Scoring**: Builds calibrated confidence scores for model outputs using TAT-QA financial dataset

## Features

- **Steering Vector Extraction**: Extract vectors that guide model behavior toward factual outputs
- **TAT-QA Integration**: Evaluate on real financial question-answering data
- **Threshold Calibration**: Automatically calibrate detection thresholds using validation data
- **Context Analysis**: Analyze how supporting context affects model confidence

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd hallucination_eval

# Install dependencies
pip install -r requirements.txt

# Set up HuggingFace authentication
# Create ~/.cache/huggingface/token with your HF token
# Or set HUGGINGFACE_HUB_TOKEN environment variable
```

## Quick Start

### Basic Usage

Run the complete hallucination detection pipeline:

```bash
python core/main_evaluation.py --num-samples 100 --layers [17] --model-name google/gemma-2-2b-it
```

### Command Line Options

```bash
python core/main_evaluation.py \
    --model-name google/gemma-2-2b-it \
    --device auto \
    --layers 10 15 20 \
    --num-samples 100
```

**Parameters:**
- `--model-name`: HuggingFace model identifier (default: google/gemma-2-2b-it)
- `--device`: Device to run on (auto, cpu, cuda)
- `--layers`: Specific layers to extract steering vectors from
- `--num-samples`: Number of TAT-QA samples for evaluation and testing (default: 50)

## System Architecture

### Core Components

1. **HallucinationDetectionSystem** (`main_evaluation.py`)
   - Orchestrates the complete pipeline
   - Manages data loading and splitting
   - Coordinates between components

2. **SteeringVectorExtractor** (`steering_vector_extractor.py`)
   - Extracts activations from transformer models
   - Computes steering vectors using mean difference method
   - Handles model loading and memory management

3. **HallucinationScorer** (`hallucination_scorer.py`)
   - Analyzes hidden states with/without context
   - Calibrates detection thresholds
   - Evaluates hallucination detection performance

### Pipeline Flow

```
1. Load and split TAT-QA data (validation/test)
2. Extract steering vectors from factual/speculative examples
3. Calibrate detection threshold on validation data
4. Analyze hidden states for context sensitivity
5. Evaluate hallucination detection on test data
6. Generate comprehensive performance report
```

## Data Requirements

### TAT-QA Dataset

Place the TAT-QA dataset at: `data/tatqa_dataset_train.json`

The system expects the following JSON structure:
```json
{
  "table": {"table": [["Header1", "Header2"], ["Value1", "Value2"]]},
  "paragraphs": [{"text": "Context paragraph"}],
  "questions": [{
    "question": "What is the value?",
    "answer": "42",
    "answer_type": "span",
    "answer_from": "table"
  }]
}
```

### Training Data

The system uses built-in factual/speculative statement pairs for steering vector extraction. These can be customized in the `create_steering_data()` function.

## Usage Examples

### Python API

```python
from core.main_evaluation import HallucinationDetectionSystem

# Initialize system
system = HallucinationDetectionSystem(
    model_name="google/gemma-2-2b-it",
    device="cuda",
    target_layers=[16, 20, 24]
)

# Run complete pipeline
results = system.run_pipeline(num_tatqa_samples=100)

# Generate report
report = system.generate_tatqa_report(results)
print(report)
```

### Individual Components

```python
from core.steering_vector_extractor import SteeringVectorExtractor
from core.hallucination_scorer import HallucinationScorer

# Extract steering vectors
extractor = SteeringVectorExtractor("google/gemma-2-2b-it")
factual_texts = ["Factual statement 1", "Factual statement 2"]
speculative_texts = ["Speculative statement 1", "Speculative statement 2"]
vectors = extractor.compute_mean_difference_vectors(factual_texts, speculative_texts)

# Analyze with scorer
scorer = HallucinationScorer(model=extractor.model)
validation_samples = [{"question": "...", "answer": "...", "context": "...", "table": "..."}]
calibration = scorer.calibrate_threshold_on_tatqa(validation_samples, vectors)
```

## Output

### Performance Report

The system generates a comprehensive report including:

```
HALLUCINATION DETECTION PIPELINE REPORT
===============================================

Total TAT-QA test samples: 50
Steering vectors extracted: 28 layers
Critical layers identified: [16, 20, 24]

CALIBRATION RESULTS:
--------------------
ROC AUC: 0.847
Optimal threshold: 0.234
Validation hallucination rate: 0.340
Mean correct projection: -0.123
Mean hallucination projection: 0.287

TEST EVALUATION RESULTS:
-------------------------
Model accuracy: 0.720
Test hallucination rate: 0.280
Detection accuracy: 0.825

PERFORMANCE ASSESSMENT:
-----------------------
Overall system performance: Excellent
Calibration quality: Excellent
Model baseline accuracy: 0.720
```

### Saved Results

Results are automatically saved to `pipeline_results.json` containing:
- All performance metrics
- Calibration parameters
- Layer importance rankings
- Individual sample results

## Configuration

### Model Selection

Supported models include any HuggingFace transformer:
- `google/gemma-2-2b-it` (default, lightweight)
All models are loaded in bfloat16
### Layer Selection

Target specific transformer layers for steering vector extraction:
```python
# Single layer
target_layers = [16]

# Multiple layers
target_layers = [10, 15, 20, 25]

# All layers - uses context sensitivity to best (most sensitive) select layer
target_layers = None
```
