# AI Coding Agent Instructions

## Project Overview

This is the **Quote Attribution Training Suite** for the [Audio Book Generator](https://github.com/bohdan-natsevych/audiobook-generator) project. It trains DeBERTa-v3-large models to identify which character speaks each quote in literary texts, enabling character-specific voice generation for audiobooks.

## Architecture

### Core Model: DeBERTa-v3-large + Custom Layers
- **Base**: `microsoft/deberta-v3-large` (1024 hidden size)
- **Special tokens**: `[QUOTE]`, `[ALTQUOTE]`, `[PAR]` for structural understanding
- **Additional layers**: Multi-layer BiLSTM → Multi-head cross-attention → Transformer encoder → Deep classifier
- **Loss functions**: Focal loss + label smoothing + optional R-Drop regularization
- **Training**: FP16 mixed precision, gradient checkpointing, candidate-level softmax
- See [models/max_performance_model.py](models/max_performance_model.py) for implementation

### Training Pipeline: Unified Jupyter Notebook
- **Primary file**: [booknlp_max_unified.ipynb](booknlp_max_unified.ipynb) - single notebook for both Kaggle and Colab
- **Environment switching**: Set `RUN_ENV = "kaggle"` or `"colab"` in first config cell
  - Kaggle: 2xT4 GPUs, multi-GPU via `accelerate`, checkpoints every 500 steps
  - Colab: 1xT4 GPU, Google Drive storage, checkpoints every 300 steps, gradient accumulation
- **Configuration**: Uses `TrainingConfig` dataclass with validation (replaces nested dicts)
  - Set `TARGET_LEVEL = 1|2` for training complexity/accuracy tradeoff
  - Target 1: PDNC dataset only, 80-85% accuracy, 4-6 hours/fold
  - Target 2: Multi-source data (PDNC + LitBank + DirectQuote), 85-88% accuracy, 6-8 hours/fold
  - Target 3: Raises `NotImplementedError` - use [models/ensemble.py](models/ensemble.py) separately
- **Fold training**: `FOLD_SELECTION = "all"` trains all 5 folds; `[0, 2]` trains specific folds
- **Auto resume**: Training resumes from latest checkpoint automatically, skips completed folds
- **Logging**: Metrics logged to CSV + optional wandb, visualizations auto-generated after training
- **Lazy loading**: PDNCFoldIterator loads folds on-demand to reduce memory usage

### Data Sources
- **PDNC**: Primary dataset, 35,978 quotes from 22 novels (literature genre)
- **LitBank**: ~3,000 quotes from 100 classic texts (classic literature)
- **DirectQuote**: 10,353 news quotes from 13 media sources (news genre)
- **Multi-source loader**: [data/multi_source_data.py](data/multi_source_data.py) with genre-balanced sampling

### Advanced Training Techniques
- **Curriculum learning**: [data/curriculum_loader.py](data/curriculum_loader.py) - progressive difficulty (simple dialogues → multi-speaker → pronoun-heavy → story-within-story)
- **Data augmentation**: [data/data_augmentation.py](data/data_augmentation.py) - synonym replacement, context variation
- **Focal loss**: [losses/focal_loss.py](losses/focal_loss.py) - handles class imbalance by focusing on hard examples
- **Post-processing**: [optimization/post_processing.py](optimization/post_processing.py) - linguistic rules for dialogue continuity, pronoun consistency

## Critical Workflows

### Training a Model
1. Open [booknlp_max_unified.ipynb](booknlp_max_unified.ipynb)
2. Set `RUN_ENV = "kaggle"` or `"colab"` (first config cell)
3. Set `TARGET_LEVEL = 1` (start with Target 1)
4. Set `FOLD_SELECTION = "all"` or specific folds like `[0, 1]`
5. Run all cells - datasets auto-download, training auto-resumes on interruption
6. Models save to `fold_*/best_model/` directories
7. Training metrics logged to `training_log.csv` + optional wandb
8. Visualization auto-generated: `training_summary.png`

### Monitoring Training Progress
- **CSV logs**: Check `{output_dir}/training_log.csv` for step-by-step metrics
- **Wandb**: If available, real-time monitoring at wandb.ai (optional, graceful fallback)
- **Checkpoints**: Found in `fold_*/checkpoint-*/` directories
- **Best models**: Saved to `fold_*/best_model/` after each fold completes
- **Visualizations**: Generated automatically after all folds complete

### Resuming Interrupted Training
- Training automatically detects and resumes from latest checkpoint
- Completed folds are automatically skipped (checks for `best_model/pytorch_model.bin`)
- No manual intervention needed - just re-run all cells
- Checkpoint cleanup happens automatically at fold boundaries

### Debugging Training Issues
- **OOM errors**: Reduce `CONFIG.gradient_accumulation_steps`, gradient checkpointing auto-disabled for multi-GPU
- **Slow training**: Adjust `CONFIG.checkpoint_every` and `CONFIG.eval_every` (default: Kaggle 500, Colab 300)
- **Poor accuracy**: Increase `TARGET_LEVEL`, adjust `CONFIG.focal_gamma` or `CONFIG.label_smoothing`
- **Resume issues**: Check `checkpoint-*/` folders in `fold_*/` directory, training auto-resumes from latest
- **Data loading errors**: Check error messages - all silent fallbacks removed, errors halt execution immediately
- **Checkpoint cleanup**: Automatic at fold boundaries, keeps last 2 checkpoints + best model

### Working with Ensemble Models
- Train all 5 folds with `FOLD_SELECTION = "all"`
- Models are used together in audiobook-generator for voting/averaging
- See [models/ensemble.py](models/ensemble.py) for ensemble implementation (3-model architecture: DeBERTa + RoBERTa + ELECTRA)

## Project-Specific Conventions

### Code Comments
- Use `# CURSOR:` prefix for important implementation notes (e.g., `# CURSOR: Special tokens for better structural understanding`)
- This convention indicates critical design decisions or non-obvious implementation details

### File Organization
- **models/**: Model architectures and training artifacts (`.model` files are saved PyTorch checkpoints)
- **data/**: Data loaders, augmentation, curriculum learning
- **losses/**: Custom loss functions (focal, label smoothing, R-Drop)
- **optimization/**: Post-processing rules, model optimization utilities
- **evaluation/**: Evaluation scripts (confidence calibration, cross-domain validation, error analysis)
- **deprecated/**: Old notebook versions (kept for reference, not actively maintained)
- **models/plans/**: Planning documents for major model iterations

### Special Token Architecture
Always use the 3 special tokens when preparing input for the model:
- `[QUOTE]`: Marks the target quote to attribute
- `[ALTQUOTE]`: Marks alternative/comparison quotes
- `[PAR]`: Marks paragraph boundaries for context
These tokens are added to tokenizer in [models/max_performance_model.py](models/max_performance_model.py#L68-L73)

### Environment-Specific Paths
Never hardcode paths - use CONFIG dataclass attributes:
```python
CONFIG.output_dir  # Storage root (populated from ENV_CFG)
CONFIG.checkpoint_every  # Save frequency (from ENV_CFG)
CONFIG.gradient_accumulation_steps  # Gradient accumulation (from ENV_CFG)
CONFIG.multi_source_base  # Dataset base path
```

## Integration Points

### With Audio Book Generator
- This repo trains models that export as `best_model_split_N.pt`
- Main project loads these models for inference during audiobook generation
- Models must implement same special token format and candidate-level softmax
- See HuggingFace token in [models/hugging_face_token.txt](models/hugging_face_token.txt) for model uploads

### External Dependencies
- **BookNLP**: Source dataset comes from speaker-attribution-acl2023 repo (auto-cloned by notebook)
- **HuggingFace**: DeBERTa-v3-large from `microsoft/deberta-v3-large`
- **Accelerate**: Multi-GPU training on Kaggle (2xT4)
- **PyTorch**: Training framework with FP16 mixed precision

## Common Pitfalls

- **Don't modify deprecated/** files - they're archived old versions
- **Don't set TARGET_LEVEL=3** - raises NotImplementedError, use models/ensemble.py separately
- **Don't forget to set RUN_ENV** before running notebook - defaults to kaggle but may need colab
- **Don't assume single fold** - production uses all 5 folds for ensemble voting
- **Don't skip curriculum learning** - set `CONFIG.use_curriculum=True` for better convergence
- **Don't ignore error messages** - no silent fallbacks, errors indicate real problems that must be fixed
- **Don't manually manage checkpoints** - auto-resume and smart cleanup handle this automatically
