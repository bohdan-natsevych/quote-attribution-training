# Quote Attribution Training Suite

> **Part of the [Audio Book Generator](https://github.com/bohdan-natsevych/audiobook-generator) project**

This repository contains the training pipeline for speaker attribution models used in the audiobook-generator project. It trains DeBERTa-based models to identify which character speaks each quote in literary texts, enabling high-quality audiobook generation with character-specific voices.

## Overview

This training suite implements a quote attribution system using DeBERTa-v3-large with specialized architecture for speaker identification. The unified notebook (`booknlp_max_unified.ipynb`) supports both Kaggle and Colab environments with automatic configuration.

## Quick Start

1) Open `booknlp_max_unified.ipynb`
2) Set `RUN_ENV = "kaggle"` or `"colab"` (default: kaggle)
3) Set `TARGET_LEVEL = 1|2|3` in the config cell
4) Set `FOLD_SELECTION = "all"` to train all 5 folds, or `[0, 1, 2]` for specific folds
5) Run all cells - datasets download automatically

### Environment Profiles

- **Kaggle**: T4 x2 GPUs, `/kaggle/working` storage, multi-GPU training via `accelerate`, checkpoints every 500 steps
- **Colab**: Single T4 GPU, Google Drive storage at `/content/drive/MyDrive/quote_attribution`, checkpoints every 300 steps, gradient accumulation for larger effective batch size

## Model Architecture

The training pipeline implements:
- **DeBERTa-v3-large** with quote/candidate masks and special tokens ([QUOTE], [ALTQUOTE], [PAR])
- **Candidate-level softmax** with label smoothing
- **Optional R-Drop** regularization for improved robustness
- **Temperature scaling** for confidence calibration
- **Curriculum learning** with difficulty-based sampling
- **Data augmentation** using synonym replacement
- **Gradient checkpointing** and FP16 mixed precision training
- **Auto checkpoint/resume** functionality

## Training Targets

Choose your target level based on desired accuracy and available resources:

| Target | Features | Expected Accuracy | Training Time |
|--------|----------|-------------------|---------------|
| **Target 1** | DeBERTa-large + augmentation + curriculum learning on PDNC dataset | 80-85% | 4-6 hours per fold (Kaggle 2xT4) |
| **Target 2** | Target 1 + multi-source data (PDNC, LitBank, DirectQuote) with genre balancing | 85-88% | 6-8 hours per fold |
| **Target 3** | Target 2 + ensemble preparation (placeholder for future work) | 88-90% | 8-10 hours per fold |

**Recommendation**: Start with Target 1 for most use cases.

### Fold Training

The notebook supports training multiple folds for cross-validation and ensemble use:

- **`FOLD_SELECTION = "all"`**: Trains all 5 folds (recommended for production/ensemble)
  - Produces 5 models: `best_model_split_0.pt` through `best_model_split_4.pt`
  - Total training time: ~20-30 hours for Target 1 (all 5 folds)
  
- **`FOLD_SELECTION = [0, 2, 4]`**: Trains only specific folds
  - Produces 3 models: `best_model_split_0.pt`, `best_model_split_2.pt`, `best_model_split_4.pt`
  
- **`FOLD_SELECTION = [1]`**: Trains a single fold
  - Produces 1 model: `best_model_split_1.pt`

**For ensemble inference**: Train all 5 folds, then use all models together in the audiobook-generator for improved accuracy through voting/averaging.

## Requirements

### Python Dependencies
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `accelerate>=0.20.0`
- `datasets`, `scikit-learn`, `numpy`, `pandas`, `tqdm`
- `nlpaug`, `nltk` (for data augmentation)

### Hardware Requirements
- **Kaggle**: 2x T4 GPUs (free tier available)
- **Colab**: 1x T4 GPU (free tier available, Pro+ recommended for longer sessions)
- **Local**: NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090, RTX 4090, A100)

## Datasets

The training pipeline supports multiple datasets. Configure which datasets to use by setting `DATASETS` in the notebook:

```python
# Examples:
DATASETS = ["pdnc"]                           # Single dataset (default)
DATASETS = ["pdnc", "litbank"]                # Multiple datasets
DATASETS = ["pdnc", "litbank", "directquote"] # All literature + news
```

### Available Datasets

| Dataset | Genre | Samples | Description |
|---------|-------|---------|-------------|
| **pdnc** | Literature | ~35,000 | PDNC (22 novels) - Primary training set |
| **litbank** | Classic Literature | ~3,000 | 100 classic texts with quote annotations |
| **directquote** | News | ~10,000 | News quotes from 13 media sources |
| **quotebank** | News | ~10,000* | Large-scale news quotes (requires manual download) |

\* Quotebank samples 10,000 from a much larger dataset.

### Dataset Download

Datasets are downloaded automatically when selected, except Quotebank which requires manual download:

| Dataset | Auto Download | Source |
|---------|--------------|--------|
| pdnc | ✅ Yes | [speaker-attribution-acl2023](https://github.com/Priya22/speaker-attribution-acl2023) |
| litbank | ✅ Yes | [dbamman/litbank](https://github.com/dbamman/litbank) |
| directquote | ✅ Yes | [THUNLP-MT/DirectQuote](https://github.com/THUNLP-MT/DirectQuote) |
| quotebank | ❌ Manual | [Zenodo](https://zenodo.org/record/4277311) |

To list available datasets programmatically:

```python
from data.multi_source_data import MultiSourceDataLoader
MultiSourceDataLoader.list_available_datasets()
```


## Configuration

Key hyperparameters in the notebook (automatically adjusted by `RUN_ENV`):

```python
CONFIG = {
    "base_model": "microsoft/deberta-v3-large",
    "epochs": 50,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,  # 16 for Colab
    "learning_rate": 5e-6,
    "checkpoint_every": 500,  # 300 for Colab
    "eval_every": 500,  # 300 for Colab
    "fp16": True,
    "gradient_checkpointing": True,
    "use_curriculum": True,
    "use_augmentation": True,
}
```

## Training Outputs

The notebook produces:

### Single Fold Training
- `checkpoint_step_{step}.pt` - Periodic checkpoints during training
- `best_model.pt` - Best performing model based on validation accuracy

### Multi-Fold Training
- `checkpoint_step_{step}_fold_{N}.pt` - Periodic checkpoints per fold
- `best_model_split_{N}.pt` - Best model for each fold (N = 0-4)
- Training logs with accuracy metrics and loss values per fold
- Evaluation reports with per-genre and per-difficulty breakdowns

**Note**: When training all 5 folds, you'll have 5 separate models that can be used independently or combined for ensemble predictions in the audiobook-generator project.

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `batch_size` to 4 or lower
- Increase `gradient_accumulation_steps` to 16 or 32
- Ensure `gradient_checkpointing=True`
- Reduce `max_length` from 512 to 384

### Slow Training
- Verify GPU is enabled in Kaggle/Colab settings
- Ensure `fp16=True` for mixed precision training
- Check that `use_accelerate=True` on multi-GPU setups (Kaggle)

### Colab Disconnects
- Checkpoints save automatically every 300 steps to Google Drive
- Training resumes automatically from latest checkpoint
- Consider upgrading to Colab Pro for longer sessions

### Low Accuracy
- Verify all 5 folds are training (set `FOLD_SELECTION = "all"`)
- Try different learning rates: `1e-5`, `2e-5`, `5e-6`
- Extend training epochs if accuracy is still improving
- Enable cross-domain validation to identify weak genres

## Repository Structure

```
booknlp_max_unified.ipynb     # Main training notebook (USE THIS)
README.md                     # This file
models/
  max_performance_model.py    # DeBERTa model with quote/candidate masks
data/
  curriculum_loader.py        # Difficulty-based sampling
  data_augmentation.py        # Synonym replacement augmentation
  multi_source_data.py        # Multi-dataset loader
losses/
  focal_loss.py              # Combined loss with focal + R-Drop
evaluation/
  confidence_calibration.py   # Temperature scaling
  cross_domain_validation.py  # Genre-specific evaluation
  error_analysis.py          # Error pattern detection
  genre_specific_adaptations.py
optimization/
  post_processing.py         # Confidence-based post-processing
  model_optimization.py      # Quantization and ONNX export
utils/
  common_utils.py            # Shared utilities
```

## Using Trained Models

After training completes, models are saved to `{output_dir}/`:

### Single Model Usage
- `best_model.pt` - Use directly for speaker attribution

### Ensemble Usage (Recommended for Best Accuracy)
- `best_model_split_0.pt` through `best_model_split_4.pt` - Load all 5 models and combine predictions through voting or averaging

In the main [Audio Book Generator](https://github.com/bohdan-natsevych/audiobook-generator) project, these models can be used to:

1. Identify speaking characters in narrative text
2. Assign distinct voices to each character
3. Generate high-quality audiobooks with character-specific narration

**Ensemble Strategy**: For production use, train all 5 folds and implement ensemble inference where each model votes on the speaker attribution. This typically improves accuracy by 2-5% compared to a single model.




## License

MIT License

