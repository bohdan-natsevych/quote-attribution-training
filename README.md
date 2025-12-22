# Quote Attribution Training Suite

> **Part of the [Audio Book Generator](https://github.com/bohdan-natsevych/audiobook-generator) project**

This repository contains the training pipeline for speaker attribution models used in the audiobook-generator project. It trains DeBERTa-v3-large models to identify which character speaks each quote in literary texts, enabling character-specific voice generation for audiobooks.

## Overview

This training suite implements a quote attribution system using DeBERTa-v3-large with specialized architecture including multi-layer BiLSTM, multi-head cross-attention, transformer encoder, and deep classifier. The unified notebook ([booknlp_max_unified.ipynb](booknlp_max_unified.ipynb)) supports both Kaggle and Colab environments with automatic configuration switching via `RUN_ENV` parameter.

## Quick Start

1. Open [booknlp_max_unified.ipynb](booknlp_max_unified.ipynb)
2. Set `RUN_ENV = "kaggle"` or `"colab"` in first config cell (default: kaggle)
3. Set `TARGET_LEVEL = 1` or `2` (start with 1)
4. Set `FOLD_SELECTION = "all"` to train all 5 folds, or `[0, 1, 2]` for specific folds
5. Run all cells - datasets auto-download, training auto-resumes if interrupted

### Environment Profiles

- **Kaggle**: 2xT4 GPUs, multi-GPU via `accelerate`, checkpoints every 500 steps, 12-hour sessions
- **Colab**: 1xT4 GPU, Google Drive storage, checkpoints every 300 steps, gradient accumulation

Training automatically resumes from latest checkpoint if interrupted. Completed folds are skipped.

## Model Architecture

Implemented in [models/max_performance_model.py](models/max_performance_model.py):

- **Base**: DeBERTa-v3-large (1024 hidden size) from `microsoft/deberta-v3-large`
- **Special tokens**: `[QUOTE]`, `[ALTQUOTE]`, `[PAR]` for structural understanding
- **Additional layers**:
  - Multi-layer BiLSTM for context encoding
  - Multi-head cross-attention (quote attends to candidates)
  - Transformer encoder for candidate interaction
  - Deep classifier (2048→1024→512→1) with residual connections
- **Training features**:
  - Focal loss + label smoothing + optional R-Drop regularization
  - Candidate-level softmax with hard negative mining
  - FP16 mixed precision + gradient checkpointing
  - Temperature scaling for confidence calibration
  - Curriculum learning + data augmentation
  - Auto checkpoint/resume from latest state

## Training Targets

Configuration uses `TrainingConfig` dataclass with validation. Choose your target level:

| Target | Features | Expected Accuracy | Training Time |
|--------|----------|-------------------|---------------|
| **Target 1** | PDNC dataset only (6 epochs) | 80-85% | 4-6 hours/fold on 2xT4 |
| **Target 2** | Multi-source (PDNC + LitBank + DirectQuote) + genre balancing (8 epochs) | 85-88% | 6-8 hours/fold |
| **Target 3** | Raises `NotImplementedError` - use [models/ensemble.py](models/ensemble.py) separately | 88-90% | Manual ensemble setup |

**Recommendation**: Start with Target 1. Set `TARGET_LEVEL = 1` in notebook config cell.

### Fold Training & Auto-Resume

- **`FOLD_SELECTION = "all"`**: Trains all 5 folds (recommended)
  - Produces: `fold_0/best_model/`, `fold_1/best_model/`, ..., `fold_4/best_model/`
  - Total time: ~20-30 hours for Target 1
  - Models saved as `.model` files (PyTorch checkpoints)
  
- **`FOLD_SELECTION = [0, 2, 4]`**: Trains specific folds only
  
- **`FOLD_SELECTION = [1]`**: Trains single fold

**Auto-resume behavior**:
- Training detects and resumes from latest checkpoint automatically
- Completed folds are skipped (checks for `best_model/pytorch_model.bin`)
- Checkpoint cleanup happens automatically at fold boundaries (keeps last 2 + best)
- No manual intervention needed - just re-run all cells

**For ensemble inference**: Train all 5 folds, use [models/ensemble.py](models/ensemble.py) for voting/averaging.

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

Configured via `TrainingConfig.datasets` parameter:

```python
# Target 1 (default):
CONFIG.datasets = ['pdnc']

# Target 2:
CONFIG.datasets = ['pdnc', 'litbank', 'directquote']
```

### Available Datasets

| Dataset | Genre | Samples | Description |
|---------|-------|---------|-------------|
| **pdnc** | Literature | 35,978 | 22 novels from PDNC dataset |
| **litbank** | Classic Literature | ~3,000 | 100 classic texts |
| **directquote** | News | 10,353 | 13 media sources |

All datasets auto-download. Implemented in [data/multi_source_data.py](data/multi_source_data.py).

**Data loading**: Uses `PDNCFoldIterator` with lazy loading to reduce memory usage. Multi-source loader provides genre-balanced sampling when `balance_genres=True`.


## Configuration

Uses `TrainingConfig` dataclass with validation (defined in notebook):

```python
@dataclass
class TrainingConfig:
    # Core settings
    target_level: int
    epochs: int = 15
    batch_size: int = 8
    lr: float = 5e-6
    
    # Dataset configuration
    datasets: List[str] = field(default_factory=lambda: ['pdnc'])
    fold_selection: Union[str, List[int]] = "all"
    
    # Advanced features
    use_augmentation: bool = True
    use_curriculum: bool = True
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    r_drop_alpha: float = 0.0
    
    # Environment-specific (auto-populated from ENV_CFG)
    gradient_accumulation_steps: int = 4  # 16 for Colab
    checkpoint_every: int = 500  # 300 for Colab
    eval_every: int = 500  # 300 for Colab
```

**Environment switching**: Set `RUN_ENV = "kaggle"` or `"colab"` in first cell. All paths, checkpoint frequencies, and GPU settings adjust automatically.

## Training Outputs

### Directory Structure
```
{output_dir}/
├── fold_0/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── best_model/
│       └── pytorch_model.bin
├── fold_1/
│   └── best_model/
├── ...
├── training_log.csv          # Step-by-step metrics
└── training_summary.png      # Auto-generated visualization
```

### Files Produced
- **`fold_N/best_model/`**: Best model for fold N (used for ensemble)
- **`fold_N/checkpoint-*/`**: Periodic checkpoints (auto-cleaned, keeps last 2)
- **`training_log.csv`**: Metrics logged every eval step
- **`training_summary.png`**: Visualization (auto-generated after training)
- Optional: wandb real-time monitoring (graceful fallback if unavailable)

**Checkpoint cleanup**: Automatic at fold boundaries to save space.

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `CONFIG.batch_size` to 4 or lower
- Increase `CONFIG.gradient_accumulation_steps` to 16 or 32
- Note: Gradient checkpointing auto-disabled for multi-GPU (Kaggle)
- Reduce `CONFIG.max_length` from 512 to 384

### Slow Training
- Verify GPU enabled in Kaggle/Colab settings (T4 recommended)
- Check `CONFIG.fp16=True` for mixed precision
- Adjust `CONFIG.checkpoint_every` and `CONFIG.eval_every` (default: Kaggle 500, Colab 300)
- On Kaggle: Ensure 2xT4 GPU accelerator selected for multi-GPU

### Resume Issues
- Check `fold_*/checkpoint-*/` folders exist in output directory
- Training auto-resumes from latest checkpoint
- Completed folds auto-skipped (checks for `best_model/pytorch_model.bin`)

### Data Loading Errors
- All silent fallbacks removed - errors halt execution immediately
- Check error messages for specific dataset/file issues
- Verify `CONFIG.multi_source_base` path is correct

### Low Accuracy
- Start with `TARGET_LEVEL=1`, then increase to 2
- Try adjusting `CONFIG.focal_gamma` (default: 2.0) or `CONFIG.label_smoothing` (default: 0.1)
- Set `CONFIG.use_curriculum=True` for better convergence
- Check per-genre accuracy in training logs

## Repository Structure

```
booknlp_max_unified.ipynb     # ⭐ MAIN TRAINING NOTEBOOK - use this
README.md                     # This file
.github/copilot-instructions.md  # AI coding agent instructions

models/
  max_performance_model.py    # MaxPerformanceSpeakerModel implementation
  ensemble.py                 # 3-model ensemble (DeBERTa + RoBERTa + ELECTRA)
  best_model_split_*.model    # Trained model checkpoints (5 folds)
  hugging_face_token.txt      # HF token for model uploads
  scores.txt                  # Model performance metrics
  plans/                      # Planning documents for iterations

data/
  curriculum_loader.py        # Curriculum learning (progressive difficulty)
  data_augmentation.py        # Synonym replacement, context variation
  multi_source_data.py        # Multi-source loader with genre balancing
  training_samples.py         # Sample data utilities

losses/
  focal_loss.py              # Focal loss + label smoothing + R-Drop

evaluation/
  confidence_calibration.py   # Temperature scaling for calibration
  cross_domain_validation.py  # Genre-specific evaluation
  error_analysis.py          # Error pattern detection
  genre_specific_adaptations.py  # Genre-specific model adaptations

optimization/
  post_processing.py         # Linguistic rules (dialogue continuity, pronouns)
  model_optimization.py      # Quantization and ONNX export

utils/
  common_utils.py            # Shared utilities

deprecated/
  # Old notebook versions (kept for reference, not maintained)
```

## Using Trained Models

After training completes, models are saved to `{output_dir}/fold_*/best_model/`:

### Integration with Audio Book Generator

In the main [Audio Book Generator](https://github.com/bohdan-natsevych/audiobook-generator) project:

1. **Single model**: Use any `fold_N/best_model/` for speaker attribution
2. **Ensemble (recommended)**: Load all 5 fold models and combine predictions via voting/averaging
   - Implementation: [models/ensemble.py](models/ensemble.py)
   - Expected accuracy gain: 2-5% over single model
   - Architecture: DeBERTa + RoBERTa + ELECTRA (3-model ensemble)

### Model Format

- **Checkpoint format**: PyTorch `.model` files (saved state dicts)
- **Special tokens**: Models expect `[QUOTE]`, `[ALTQUOTE]`, `[PAR]` in input
- **Candidate-level softmax**: Predictions normalized per candidate set
- **Export**: Use [optimization/model_optimization.py](optimization/model_optimization.py) for ONNX/quantization

## Advanced Training Techniques

Implemented in the pipeline:

- **Curriculum learning** ([data/curriculum_loader.py](data/curriculum_loader.py)): Progressive difficulty
  - Simple dialogues \u2192 multi-speaker \u2192 pronoun-heavy \u2192 story-within-story
- **Data augmentation** ([data/data_augmentation.py](data/data_augmentation.py)): Synonym replacement, context variation
- **Focal loss** ([losses/focal_loss.py](losses/focal_loss.py)): Handles class imbalance by focusing on hard examples
- **Post-processing** ([optimization/post_processing.py](optimization/post_processing.py)): Linguistic rules for dialogue continuity, pronoun consistency

Set `CONFIG.use_curriculum=True` and `CONFIG.use_augmentation=True` for best results.

## Project Conventions

### Code Comments
- `# CURSOR:` prefix indicates critical design decisions or implementation notes
- Example: `# CURSOR: Special tokens for better structural understanding`

### Special Token Architecture
Always use the 3 special tokens when preparing model input:
- `[QUOTE]`: Marks target quote to attribute
- `[ALTQUOTE]`: Marks alternative/comparison quotes  
- `[PAR]`: Marks paragraph boundaries for context

Defined in [models/max_performance_model.py](models/max_performance_model.py#L38)

## License

MIT License

