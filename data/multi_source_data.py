"""
Multi-Source Data Loading and Balancing

Supports loading and combining:
- PDNC: 35,978 quotes (22 novels)
- LitBank: ~3,000 quotes (classic literature)
- DirectQuote: 10,353 quotes (news media)
- Quotebank: News quotes (samples)

Features:
- Genre-balanced sampling
- Weighted loss for class balance
- Validation sets per genre
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


@dataclass
class DatasetConfig:
    """Configuration for a dataset source."""
    name: str
    path: str
    genre: str
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    expected_size: int
    loader_fn: str  # Name of loader function


class MultiSourceDataLoader:
    """
    Load and combine multiple quote attribution datasets.
    """
    
    # CURSOR: Default dataset configurations
    DEFAULT_DATASETS = {
        'pdnc': DatasetConfig(
            name='PDNC',
            path='pdnc',
            genre='literature',
            priority='CRITICAL',
            expected_size=35978,
            loader_fn='load_pdnc'
        ),
        'litbank': DatasetConfig(
            name='LitBank',
            path='litbank',
            genre='classic_literature',
            priority='CRITICAL',
            expected_size=3000,
            loader_fn='load_litbank'
        ),
        'directquote': DatasetConfig(
            name='DirectQuote',
            path='directquote',
            genre='news',
            priority='HIGH',
            expected_size=10353,
            loader_fn='load_directquote'
        ),
        'quotebank': DatasetConfig(
            name='Quotebank',
            path='quotebank',
            genre='news',
            priority='MEDIUM',
            expected_size=10000,  # Sample
            loader_fn='load_quotebank'
        ),
    }
    
    def __init__(
        self,
        base_path: str,
        datasets: Optional[List[str]] = None,
        max_samples_per_dataset: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize multi-source data loader.
        
        Args:
            base_path: Base path containing all datasets
            datasets: List of dataset names to load (default: all)
            max_samples_per_dataset: Limit samples per dataset
            seed: Random seed for reproducibility
        """
        self.base_path = Path(base_path)
        self.datasets = datasets or list(self.DEFAULT_DATASETS.keys())
        self.max_samples = max_samples_per_dataset
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # CURSOR: Store loaded data by genre
        self.data_by_genre: Dict[str, List[Dict]] = defaultdict(list)
        self.data_by_source: Dict[str, List[Dict]] = defaultdict(list)
    
    def load_all(self) -> Dict[str, List[Dict]]:
        """
        Load all configured datasets.
        
        Returns:
            Dictionary mapping genre to samples
        """
        for dataset_name in self.datasets:
            if dataset_name not in self.DEFAULT_DATASETS:
                print(f"Warning: Unknown dataset {dataset_name}, skipping")
                continue
            
            config = self.DEFAULT_DATASETS[dataset_name]
            dataset_path = self.base_path / config.path
            
            if not dataset_path.exists():
                print(f"Warning: Dataset path {dataset_path} not found, skipping")
                continue
            
            print(f"\nLoading {config.name} from {dataset_path}...")
            
            try:
                loader_fn = getattr(self, config.loader_fn)
                samples = loader_fn(dataset_path)
                
                # CURSOR: Add metadata to samples
                for sample in samples:
                    sample['source'] = dataset_name
                    sample['genre'] = config.genre
                
                # Limit samples if configured
                if self.max_samples and len(samples) > self.max_samples:
                    samples = random.sample(samples, self.max_samples)
                
                self.data_by_genre[config.genre].extend(samples)
                self.data_by_source[dataset_name] = samples
                
                print(f"  Loaded {len(samples)} samples from {config.name}")
                
            except Exception as e:
                print(f"  Error loading {config.name}: {e}")
        
        return dict(self.data_by_genre)
    
    def load_pdnc(self, path: Path) -> List[Dict]:
        """Load PDNC dataset."""
        samples = []
        
        quote_files = list(path.glob("**/quote_info.csv"))
        
        for quote_file in quote_files:
            book_dir = quote_file.parent
            book_txt = book_dir / "book.txt"
            
            if not book_txt.exists():
                continue
            
            try:
                with open(book_txt, 'r', encoding='utf-8', errors='ignore') as f:
                    book_text = f.read()
                
                df = pd.read_csv(quote_file)
                
                for _, row in df.iterrows():
                    try:
                        quote_start = int(row.get('qBegin', 0))
                        quote_end = int(row.get('qEnd', len(book_text)))
                        
                        # CURSOR: Extract context
                        ctx_start = max(0, quote_start - 500)
                        ctx_end = min(len(book_text), quote_end + 500)
                        
                        sample = {
                            'text': book_text[ctx_start:ctx_end],
                            'quote': row.get('qText', ''),
                            'quote_start': quote_start - ctx_start,
                            'quote_end': quote_end - ctx_start,
                            'speaker': row.get('speaker', ''),
                            'book_id': book_dir.name,
                        }
                        samples.append(sample)
                    except Exception:
                        continue
            except Exception:
                continue
        
        return samples
    
    def load_litbank(self, path: Path) -> List[Dict]:
        """Load LitBank dataset."""
        samples = []
        
        # CURSOR: LitBank has entity annotations
        entity_dir = path / "entities" / "brat"
        text_dir = path / "original"
        
        if not entity_dir.exists():
            # Try alternative structure
            ann_files = list(path.glob("**/*.ann"))
        else:
            ann_files = list(entity_dir.glob("*.ann"))
        
        for ann_file in ann_files:
            try:
                # Find corresponding text file
                text_file = ann_file.with_suffix('.txt')
                if not text_file.exists():
                    text_file = text_dir / ann_file.name.replace('.ann', '.txt')
                
                if not text_file.exists():
                    continue
                
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # CURSOR: Parse annotations for quotes and speakers
                with open(ann_file, 'r', encoding='utf-8') as f:
                    annotations = f.readlines()
                
                # Simple extraction - actual implementation depends on format
                for ann in annotations:
                    if ann.startswith('T'):
                        parts = ann.strip().split('\t')
                        if len(parts) >= 3:
                            entity_info = parts[1].split()
                            if entity_info[0] == 'Quote' or 'quote' in entity_info[0].lower():
                                try:
                                    start = int(entity_info[1])
                                    end = int(entity_info[2])
                                    quote_text = parts[2]
                                    
                                    ctx_start = max(0, start - 300)
                                    ctx_end = min(len(text), end + 300)
                                    
                                    samples.append({
                                        'text': text[ctx_start:ctx_end],
                                        'quote': quote_text,
                                        'quote_start': start - ctx_start,
                                        'quote_end': end - ctx_start,
                                        'speaker': '',  # Need additional processing
                                        'book_id': ann_file.stem,
                                    })
                                except Exception:
                                    continue
            except Exception:
                continue
        
        return samples
    
    def load_directquote(self, path: Path) -> List[Dict]:
        """Load DirectQuote dataset."""
        samples = []
        
        # CURSOR: DirectQuote typically has JSON format
        json_files = list(path.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle both list and dict formats
                if isinstance(data, dict):
                    data = data.get('quotes', []) or data.get('data', []) or [data]
                
                for item in data:
                    try:
                        text = item.get('text', item.get('context', ''))
                        quote = item.get('quote', item.get('quotation', ''))
                        speaker = item.get('speaker', item.get('source', ''))
                        
                        if text and quote:
                            samples.append({
                                'text': text,
                                'quote': quote,
                                'speaker': speaker,
                                'quote_start': text.find(quote) if quote in text else 0,
                                'quote_end': text.find(quote) + len(quote) if quote in text else len(quote),
                            })
                    except Exception:
                        continue
            except Exception:
                continue
        
        return samples
    
    def load_quotebank(self, path: Path) -> List[Dict]:
        """Load Quotebank dataset (sampled)."""
        samples = []
        
        # CURSOR: Quotebank is very large, sample from it
        data_files = list(path.glob("*.json*"))
        
        for data_file in data_files[:5]:  # Limit files
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(samples) >= (self.max_samples or 10000):
                            break
                        
                        try:
                            item = json.loads(line)
                            quote = item.get('quotation', '')
                            speaker = item.get('speaker', '')
                            
                            if quote and len(quote) > 20:
                                samples.append({
                                    'text': quote,  # Quotebank has limited context
                                    'quote': quote,
                                    'speaker': speaker,
                                    'quote_start': 0,
                                    'quote_end': len(quote),
                                })
                        except Exception:
                            continue
            except Exception:
                continue
        
        return samples
    
    def get_balanced_samples(
        self,
        samples_per_genre: int = 10000,
        min_samples: int = 1000
    ) -> List[Dict]:
        """
        Get genre-balanced samples.
        
        Args:
            samples_per_genre: Target samples per genre
            min_samples: Minimum samples required per genre
            
        Returns:
            Balanced list of samples
        """
        balanced = []
        
        for genre, samples in self.data_by_genre.items():
            if len(samples) < min_samples:
                print(f"Warning: Genre {genre} has only {len(samples)} samples")
            
            n = min(len(samples), samples_per_genre)
            selected = random.sample(samples, n) if len(samples) > n else samples
            balanced.extend(selected)
            print(f"  {genre}: {len(selected)} samples")
        
        random.shuffle(balanced)
        return balanced
    
    def create_weighted_sampler(
        self,
        samples: List[Dict]
    ) -> WeightedRandomSampler:
        """
        Create weighted sampler for balanced training.
        
        Args:
            samples: List of samples with 'genre' field
            
        Returns:
            WeightedRandomSampler for DataLoader
        """
        # CURSOR: Count samples per genre
        genre_counts = defaultdict(int)
        for s in samples:
            genre_counts[s['genre']] += 1
        
        # CURSOR: Compute weights (inverse frequency)
        total = len(samples)
        genre_weights = {
            g: total / (len(genre_counts) * c)
            for g, c in genre_counts.items()
        }
        
        # CURSOR: Assign weight to each sample
        weights = [genre_weights[s['genre']] for s in samples]
        
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(samples),
            replacement=True
        )
    
    def split_by_genre(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data ensuring each genre is represented in all splits.
        
        Args:
            val_ratio: Validation set ratio per genre
            test_ratio: Test set ratio per genre
            
        Returns:
            (train_samples, val_samples, test_samples)
        """
        train, val, test = [], [], []
        
        for genre, samples in self.data_by_genre.items():
            n = len(samples)
            random.shuffle(samples)
            
            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)
            
            test.extend(samples[:n_test])
            val.extend(samples[n_test:n_test + n_val])
            train.extend(samples[n_test + n_val:])
        
        return train, val, test


def create_multi_source_dataloader(
    base_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    datasets: Optional[List[str]] = None,
    balance_genres: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from multiple sources.
    
    Args:
        base_path: Base path containing datasets
        tokenizer: Tokenizer for encoding
        batch_size: Batch size
        max_length: Max sequence length
        datasets: List of datasets to load
        balance_genres: Whether to balance genres
        val_ratio: Validation split ratio
        seed: Random seed
        
    Returns:
        (train_dataloader, val_dataloader)
    """
    # Load data
    loader = MultiSourceDataLoader(
        base_path=base_path,
        datasets=datasets,
        seed=seed
    )
    loader.load_all()
    
    # Split data
    train_samples, val_samples, _ = loader.split_by_genre(val_ratio=val_ratio, test_ratio=0)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    
    # Create datasets
    from data.curriculum_loader import CurriculumDataset
    
    train_dataset = CurriculumDataset(train_samples, tokenizer, max_length)
    val_dataset = CurriculumDataset(val_samples, tokenizer, max_length)
    
    # Create samplers
    if balance_genres:
        train_sampler = loader.create_weighted_sampler(train_samples)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_dataloader, val_dataloader


# CURSOR: Export public API
__all__ = [
    'MultiSourceDataLoader',
    'DatasetConfig',
    'create_multi_source_dataloader'
]



