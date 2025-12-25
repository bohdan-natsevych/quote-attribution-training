"""
Curriculum Learning Data Loader

Implements progressive difficulty training:
1. simple_dialogues: 2 speakers, clear attributions
2. multi_speaker: 3+ speakers
3. pronoun_heavy: Many pronoun references
4. story_within_story: Nested narratives

The model learns easier examples first, then progressively harder ones.
"""

import random
from typing import List, Dict, Optional, Iterator, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class DifficultyLevel(Enum):
    """Curriculum difficulty levels."""
    SIMPLE_DIALOGUES = 0      # 2 speakers, clear attributions
    MULTI_SPEAKER = 1         # 3+ speakers
    PRONOUN_HEAVY = 2         # Many pronoun references
    STORY_WITHIN_STORY = 3    # Nested narratives


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    # Stage progression (as fraction of total epochs)
    stage_boundaries: Dict[DifficultyLevel, Tuple[float, float]] = None
    
    # Sampling weights per stage
    sampling_weights: Dict[DifficultyLevel, float] = None
    
    # Whether to include easier samples in harder stages
    cumulative: bool = True
    
    # Minimum samples per batch from current difficulty
    min_current_difficulty_ratio: float = 0.5
    
    def __post_init__(self):
        if self.stage_boundaries is None:
            self.stage_boundaries = {
                DifficultyLevel.SIMPLE_DIALOGUES: (0.0, 0.2),
                DifficultyLevel.MULTI_SPEAKER: (0.2, 0.5),
                DifficultyLevel.PRONOUN_HEAVY: (0.5, 0.75),
                DifficultyLevel.STORY_WITHIN_STORY: (0.75, 1.0),
            }
        
        if self.sampling_weights is None:
            self.sampling_weights = {
                DifficultyLevel.SIMPLE_DIALOGUES: 1.0,
                DifficultyLevel.MULTI_SPEAKER: 1.0,
                DifficultyLevel.PRONOUN_HEAVY: 1.0,
                DifficultyLevel.STORY_WITHIN_STORY: 1.5,  # Oversample hardest
            }


class DifficultyClassifier:
    """
    Classifies samples by difficulty level.
    
    Uses heuristics to determine difficulty:
    - Number of candidate speakers
    - Pronoun density in context
    - Narrative nesting indicators
    - Quote attribution clarity
    """
    
    # CURSOR: Common pronouns for detection
    PRONOUNS = {
        'he', 'she', 'they', 'him', 'her', 'them',
        'his', 'hers', 'their', 'theirs',
        'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'
    }
    
    # CURSOR: Story-within-story indicators
    NESTED_STORY_MARKERS = {
        'once upon a time', 'long ago', 'there was', 'there once',
        'told the story', 'began the tale', 'related how',
        'remembered when', 'recalled the time', 'the story goes'
    }
    
    # CURSOR: Clear attribution markers
    CLEAR_ATTRIBUTION_VERBS = {
        'said', 'asked', 'replied', 'answered', 'exclaimed',
        'whispered', 'shouted', 'murmured', 'muttered'
    }
    
    def __init__(self, pronoun_threshold: float = 0.1):
        """
        Initialize classifier.
        
        Args:
            pronoun_threshold: Pronoun density threshold for pronoun_heavy
        """
        self.pronoun_threshold = pronoun_threshold
    
    def classify(self, sample: Dict[str, Any]) -> DifficultyLevel:
        """
        Classify a sample's difficulty level.
        
        Args:
            sample: Sample dictionary with:
                - text: Full text
                - candidates: List of candidate speakers
                - context_before, context_after: Context around quote
                
        Returns:
            DifficultyLevel enum value
        """
        text = sample.get('text', '').lower()
        candidates = sample.get('candidates', [])
        context = sample.get('context_before', '') + ' ' + sample.get('context_after', '')
        context = context.lower()
        
        # CURSOR: Check for nested story markers
        if self._has_nested_story(text):
            return DifficultyLevel.STORY_WITHIN_STORY
        
        # CURSOR: Check pronoun density
        if self._get_pronoun_density(context) > self.pronoun_threshold:
            return DifficultyLevel.PRONOUN_HEAVY
        
        # CURSOR: Check number of speakers
        if len(candidates) > 2:
            return DifficultyLevel.MULTI_SPEAKER
        
        # CURSOR: Check for clear attribution
        if self._has_clear_attribution(text):
            return DifficultyLevel.SIMPLE_DIALOGUES
        
        # Default to multi-speaker if unclear
        return DifficultyLevel.MULTI_SPEAKER
    
    def _has_nested_story(self, text: str) -> bool:
        """Check if text contains nested story markers."""
        return any(marker in text for marker in self.NESTED_STORY_MARKERS)
    
    def _get_pronoun_density(self, text: str) -> float:
        """Calculate pronoun density in text."""
        words = text.split()
        if not words:
            return 0.0
        
        pronoun_count = sum(1 for w in words if w.lower() in self.PRONOUNS)
        return pronoun_count / len(words)
    
    def _has_clear_attribution(self, text: str) -> bool:
        """Check if text has clear speaker attribution."""
        return any(verb in text for verb in self.CLEAR_ATTRIBUTION_VERBS)
    
    def classify_dataset(
        self,
        samples: List[Dict]
    ) -> Dict[DifficultyLevel, List[int]]:
        """
        Classify all samples in a dataset.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Dictionary mapping difficulty level to sample indices
        """
        classified = defaultdict(list)
        
        for idx, sample in enumerate(samples):
            difficulty = self.classify(sample)
            classified[difficulty].append(idx)
        
        return dict(classified)


class CurriculumSampler(Sampler):
    """
    Sampler that implements curriculum learning.
    
    Samples are drawn based on current training progress,
    starting with easier examples and gradually introducing harder ones.
    """
    
    def __init__(
        self,
        difficulty_indices: Dict[DifficultyLevel, List[int]],
        config: CurriculumConfig,
        total_epochs: int,
        current_epoch: int = 0,
        batch_size: int = 8,
        seed: Optional[int] = None,
        sample_weights: Optional[List[float]] = None
    ):
        """
        Initialize curriculum sampler.
        
        Args:
            difficulty_indices: Mapping of difficulty to sample indices
            config: Curriculum configuration
            total_epochs: Total number of training epochs
            current_epoch: Current epoch (0-indexed)
            batch_size: Batch size for sampling
            seed: Random seed
        """
        self.difficulty_indices = difficulty_indices
        self.config = config
        self.total_epochs = total_epochs
        self.current_epoch = current_epoch
        self.batch_size = batch_size
        self.seed = seed
        self.sample_weights = sample_weights
        
        if seed is not None:
            random.seed(seed)
        
        # CURSOR: Calculate total samples
        self.total_samples = sum(len(indices) for indices in difficulty_indices.values())
    
    def set_epoch(self, epoch: int):
        """Update current epoch for progressive curriculum."""
        self.current_epoch = epoch
    
    def get_current_difficulty(self) -> DifficultyLevel:
        """Get the current difficulty level based on training progress."""
        progress = self.current_epoch / max(self.total_epochs, 1)
        
        for level, (start, end) in self.config.stage_boundaries.items():
            if start <= progress < end:
                return level
        
        return DifficultyLevel.STORY_WITHIN_STORY
    
    def get_available_difficulties(self) -> List[DifficultyLevel]:
        """Get difficulties available at current stage."""
        current = self.get_current_difficulty()
        
        if self.config.cumulative:
            # Include all difficulties up to and including current
            return [d for d in DifficultyLevel if d.value <= current.value]
        else:
            return [current]
    
    def __iter__(self) -> Iterator[int]:
        """
        Generate indices for one epoch.

        CURSOR: This sampler must yield a stable epoch size (all samples) because HuggingFace Trainer
        computes `max_steps` ONCE from `len(train_dataloader)`. Returning only the current curriculum
        stage subset makes Trainer think the dataset is tiny (e.g., ~448 samples) and it stops early.

        We keep curriculum behavior by ordering:
        - stage-available difficulties first (weighted order),
        - then the remaining difficulties (shuffled),
        while still returning ALL indices exactly once per epoch.
        """
        available = set(self.get_available_difficulties())
        current = self.get_current_difficulty()

        # CURSOR: Deterministic RNG per epoch when seed is provided.
        rng_seed = None if self.seed is None else int(self.seed) + int(self.current_epoch)
        rng = np.random.default_rng(rng_seed)

        front_pool: List[int] = []
        front_weights: List[float] = []
        back_pool: List[int] = []

        # CURSOR: Build a full-epoch permutation, keeping curriculum ordering pressure in the prefix.
        for difficulty in DifficultyLevel:
            indices = self.difficulty_indices.get(difficulty, [])
            if not indices:
                continue

            if difficulty in available:
                base_weight = float(self.config.sampling_weights.get(difficulty, 1.0))
                if difficulty == current:
                    base_weight *= 1.5

                for idx in indices:
                    front_pool.append(int(idx))
                    if self.sample_weights is not None and 0 <= idx < len(self.sample_weights):
                        front_weights.append(base_weight * float(self.sample_weights[idx]))
                    else:
                        front_weights.append(base_weight)
            else:
                back_pool.extend(int(i) for i in indices)

        order_out: List[int] = []

        if front_pool:
            weight_arr = np.asarray(front_weights, dtype=np.float64)
            total_weight = float(weight_arr.sum())
            if total_weight > 0:
                weight_arr = weight_arr / total_weight
                front_order = rng.choice(len(front_pool), size=len(front_pool), replace=False, p=weight_arr)
            else:
                front_order = rng.permutation(len(front_pool))
            order_out.extend([front_pool[i] for i in front_order.tolist()])

        if back_pool:
            back_order = rng.permutation(len(back_pool))
            order_out.extend([back_pool[i] for i in back_order.tolist()])

        # CURSOR: Advance epoch automatically as a fallback when the caller doesn't invoke set_epoch.
        self.current_epoch += 1
        return iter(order_out)
    
    def __len__(self) -> int:
        """
        CURSOR: Return number of samples per epoch (stable, full dataset).

        IMPORTANT: HuggingFace Trainer uses this (via DataLoader.__len__) to compute max_steps.
        """
        return int(self.total_samples)


class CurriculumDataset(Dataset):
    """
    Dataset wrapper that supports curriculum learning.
    
    Wraps an existing dataset and adds difficulty classification.
    """
    
    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_length: int = 512,
        classifier: Optional[DifficultyClassifier] = None
    ):
        """
        Initialize curriculum dataset.
        
        Args:
            samples: List of sample dictionaries
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            classifier: Difficulty classifier (creates default if None)
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # CURSOR: Classify all samples
        self.classifier = classifier or DifficultyClassifier()
        self.difficulty_indices = self.classifier.classify_dataset(samples)
        
        # Log distribution
        self._log_distribution()
    
    def _log_distribution(self):
        """Log the difficulty distribution."""
        print("\n=== Curriculum Dataset Distribution ===")
        for level in DifficultyLevel:
            count = len(self.difficulty_indices.get(level, []))
            pct = count / len(self.samples) * 100 if self.samples else 0
            print(f"  {level.name}: {count} samples ({pct:.1f}%)")
        print("=" * 40 + "\n")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # CURSOR: Tokenize the text
        text = sample.get('text', '')
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sample.get('speaker_idx', 0)),
            'difficulty': self.classifier.classify(sample).value
        }
    
    def get_difficulty_indices(self) -> Dict[DifficultyLevel, List[int]]:
        """Return difficulty classification for sampler."""
        return self.difficulty_indices


def create_curriculum_dataloader(
    samples: List[Dict],
    tokenizer,
    config: CurriculumConfig,
    total_epochs: int,
    current_epoch: int = 0,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 0,
    seed: Optional[int] = None
) -> Tuple[DataLoader, CurriculumSampler]:
    """
    Create a DataLoader with curriculum learning.
    
    Args:
        samples: List of sample dictionaries
        tokenizer: Tokenizer for encoding
        config: Curriculum configuration
        total_epochs: Total training epochs
        current_epoch: Current epoch
        batch_size: Batch size
        max_length: Max sequence length
        num_workers: DataLoader workers
        seed: Random seed
        
    Returns:
        (DataLoader, CurriculumSampler) - sampler needed to update epoch
    """
    # CURSOR: Create dataset
    dataset = CurriculumDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # CURSOR: Create curriculum sampler
    sampler = CurriculumSampler(
        difficulty_indices=dataset.get_difficulty_indices(),
        config=config,
        total_epochs=total_epochs,
        current_epoch=current_epoch,
        batch_size=batch_size,
        seed=seed
    )
    
    # CURSOR: Create DataLoader with sampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, sampler


def get_curriculum_schedule(
    epoch: int,
    total_epochs: int,
    config: Optional[CurriculumConfig] = None
) -> Dict[str, Any]:
    """
    Get curriculum schedule information for current epoch.
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        config: Curriculum configuration
        
    Returns:
        Dictionary with current stage info
    """
    if config is None:
        config = CurriculumConfig()
    
    progress = epoch / max(total_epochs, 1)
    
    current_level = None
    for level, (start, end) in config.stage_boundaries.items():
        if start <= progress < end:
            current_level = level
            break
    
    if current_level is None:
        current_level = DifficultyLevel.STORY_WITHIN_STORY
    
    return {
        'epoch': epoch,
        'progress': progress,
        'current_difficulty': current_level.name,
        'difficulty_value': current_level.value,
        'cumulative': config.cumulative,
        'stage_boundaries': {
            k.name: v for k, v in config.stage_boundaries.items()
        }
    }


# CURSOR: Export public API
__all__ = [
    'DifficultyLevel',
    'CurriculumConfig',
    'DifficultyClassifier',
    'CurriculumSampler',
    'CurriculumDataset',
    'create_curriculum_dataloader',
    'get_curriculum_schedule'
]



