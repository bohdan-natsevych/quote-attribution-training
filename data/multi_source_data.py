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

# CURSOR: import json  # unused after Quotebank disabled
import random
# CURSOR: import bz2  # unused after Quotebank disabled
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler


@dataclass
class DatasetConfig:
    """Configuration for a dataset source."""
    name: str
    path: str
    genre: str
    expected_size: int
    loader_fn: str  # Name of loader function
    download_url: str = ''  # Git clone URL or download link
    description: str = ''


def _clone_repo(url: str, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"✅ Found {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def _download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"✅ Found {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


# CURSOR: Quotebank helper disabled.
# CURSOR: def _prepare_quotebank(download_url: str, target_dir: Path) -> Path:
# CURSOR:     target_dir.mkdir(parents=True, exist_ok=True)
# CURSOR:     qb_file = target_dir / "quotes-2019.json"
# CURSOR:
# CURSOR:     if qb_file.exists():
# CURSOR:         print(f"✅ Quotebank already present at {qb_file}")
# CURSOR:         return qb_file
# CURSOR:
# CURSOR:     qb_bz2 = target_dir / "quotes-2019.json.bz2"
# CURSOR:     _download_file(f"{download_url}/files/quotes-2019.json.bz2?download=1", qb_bz2)
# CURSOR:
# CURSOR:     # CURSOR: Stream-decompress to avoid loading the archive into memory
# CURSOR:     print("📦 Decompressing Quotebank archive...")
# CURSOR:     with bz2.open(qb_bz2, 'rt', encoding='utf-8') as f_in, open(qb_file, 'w', encoding='utf-8') as f_out:
# CURSOR:         for line in f_in:
# CURSOR:             f_out.write(line)
# CURSOR:
# CURSOR:     try:
# CURSOR:         qb_bz2.unlink()
# CURSOR:     except FileNotFoundError:
# CURSOR:         pass
# CURSOR:
# CURSOR:     print(f"✅ Quotebank ready at {qb_file}")
# CURSOR:     return qb_file


class MultiSourceDataLoader:
    """
    Load and combine multiple quote attribution datasets.
    """
    
    # Available datasets for quote attribution training
    # Users select which datasets to train on by name, e.g. ['pdnc', 'litbank']
    AVAILABLE_DATASETS = {
        'pdnc': DatasetConfig(
            name='PDNC',
            path='pdnc',
            genre='literature',
            expected_size=35978,
            loader_fn='load_pdnc',
            download_url='https://github.com/Priya22/speaker-attribution-acl2023.git',
            description='Pride and Prejudice Dialog Novel Corpus - 22 novels, literature focus'
        ),
        'litbank': DatasetConfig(
            name='LitBank',
            path='litbank',
            genre='classic_literature',
            expected_size=3000,
            loader_fn='load_litbank',
            download_url='https://github.com/dbamman/litbank.git',
            description='100 classic literary texts with quote/speaker annotations'
        ),
        'directquote': DatasetConfig(
            name='DirectQuote',
            path='directquote',
            genre='news',
            expected_size=10353,
            loader_fn='load_directquote',
            download_url='https://github.com/THUNLP-MT/DirectQuote.git',
            description='News article quotes from 13 media sources'
        ),
        # CURSOR: Quotebank dataset entry disabled.
        # CURSOR: 'quotebank': DatasetConfig(
        # CURSOR:     name='Quotebank',
        # CURSOR:     path='quotebank',
        # CURSOR:     genre='news',
        # CURSOR:     expected_size=10000,
        # CURSOR:     loader_fn='load_quotebank',
        # CURSOR:     download_url='https://zenodo.org/record/4277311',
        # CURSOR:     description='Large-scale news quotes dataset (~175M quotes, requires manual download)'
        # CURSOR: ),
    }
    
    # Alias for backwards compatibility
    DEFAULT_DATASETS = AVAILABLE_DATASETS
    
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
        
        # Store loaded data by genre and source
        self.data_by_genre: Dict[str, List[Dict]] = defaultdict(list)
        self.data_by_source: Dict[str, List[Dict]] = defaultdict(list)
    
    @classmethod
    def list_available_datasets(cls) -> None:
        """Print available datasets with descriptions."""
        print("\n📚 Available Datasets for Quote Attribution Training:")
        print("=" * 60)
        for name, config in cls.AVAILABLE_DATASETS.items():
            print(f"\n  {name}")
            print(f"    Name: {config.name}")
            print(f"    Genre: {config.genre}")
            print(f"    Expected samples: ~{config.expected_size:,}")
            print(f"    Description: {config.description}")
            print(f"    Download: {config.download_url}")
        print("\n" + "=" * 60)
        print("Usage: Set DATASETS = ['pdnc', 'litbank'] in notebook config")
    
    def load_all(self) -> Dict[str, List[Dict]]:
        """
        Load all configured datasets.
        
        Returns:
            Dictionary mapping genre to samples
        """
        for dataset_name in self.datasets:
            if dataset_name not in self.AVAILABLE_DATASETS:
                print(f"⚠️ Unknown dataset '{dataset_name}'. Available: {list(self.AVAILABLE_DATASETS.keys())}")
                continue
            
            config = self.AVAILABLE_DATASETS[dataset_name]
            dataset_path = self.base_path / config.path
            
            if not dataset_path.exists():
                print(f"\n⚠️ Dataset '{config.name}' not found at {dataset_path}")
                if config.download_url:
                    print(f"   Download from: {config.download_url}")
                continue
            
            print(f"\n📂 Loading {config.name} from {dataset_path}...")
            
            try:
                loader_fn = getattr(self, config.loader_fn)
                samples = loader_fn(dataset_path)
                
                # Add metadata to samples
                for sample in samples:
                    sample['source'] = dataset_name
                    sample['genre'] = config.genre
                
                # Limit samples if configured
                if self.max_samples and len(samples) > self.max_samples:
                    samples = random.sample(samples, self.max_samples)
                
                self.data_by_genre[config.genre].extend(samples)
                self.data_by_source[dataset_name] = samples
                
                print(f"   ✅ Loaded {len(samples):,} samples from {config.name}")
                
            except Exception as e:
                print(f"   ❌ Error loading {config.name}: {e}")
        
        return dict(self.data_by_genre)
    
    def load_pdnc(self, path: Path) -> List[Dict]:
        """Load PDNC dataset."""
        samples = []
        
        # CURSOR: Handle nested structure when full repo is cloned
        # The speaker-attribution-acl2023 repo has data/pdnc/ inside it
        nested_path = path / "data" / "pdnc"
        if nested_path.exists():
            print(f"  Found nested PDNC data at {nested_path}")
            path = nested_path
        
        quote_files = list(path.glob("**/quote_info.csv"))
        print(f"  Found {len(quote_files)} quote_info.csv files")
        
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
        """Load LitBank dataset from quotations/ directory.
        
        LitBank TSV format (pipe-delimited):
        QUOTE|Q342|54|0|54|13|" Of course , we 'll take over your furniture , mother , "|
        ATTRIB|Q342|Winnie_Verloc-3|
        
        The quotations TSV files may be in:
        - quotations/tsv/*.tsv (older structure)
        - quotations/*.tsv (newer structure)
        
        Text files for context are in original/*.txt
        """
        samples = []
        
        # Try multiple possible locations for quotation TSV files
        quote_dir_candidates = [
            path / "quotations" / "tsv",
            path / "quotations",
            path,
        ]
        
        tsv_files = []
        for quote_dir in quote_dir_candidates:
            if quote_dir.exists():
                tsv_files = list(quote_dir.glob("*.tsv"))
                if tsv_files:
                    print(f"  Found {len(tsv_files)} TSV files in {quote_dir}")
                    break
        
        if not tsv_files:
            # Fallback: search recursively
            tsv_files = list(path.glob("**/*.tsv"))
            if tsv_files:
                print(f"  Found {len(tsv_files)} TSV files via recursive search")
        
        if not tsv_files:
            print(f"  Warning: No TSV quotation files found in {path}")
            return samples
        
        # Try multiple possible locations for original text files
        text_dir_candidates = [
            path / "original",
            path / "texts",
            path,
        ]
        text_dir = None
        for candidate in text_dir_candidates:
            if candidate.exists() and list(candidate.glob("*.txt")):
                text_dir = candidate
                break
        
        if text_dir is None:
            print(f"  Warning: No text directory found in {path}")
            return samples
        
        for tsv_file in tsv_files:
            try:
                # Load text file for context
                # TSV files are named like "730_oliver_twist_brat.tsv"
                # Text files in original/ are named like "730_oliver_twist.txt"
                base_name = tsv_file.stem.replace('_brat', '')
                text_file = text_dir / f"{base_name}.txt"
                
                if not text_file.exists():
                    # Try finding text file with similar prefix (first number)
                    prefix = base_name.split('_')[0]
                    possible_texts = list(text_dir.glob(f"{prefix}_*.txt"))
                    if not possible_texts:
                        possible_texts = list(text_dir.glob(f"*{base_name.split('_')[1] if '_' in base_name else base_name}*.txt"))
                    if possible_texts:
                        text_file = possible_texts[0]
                    else:
                        print(f"    Skipping {tsv_file.name}: no matching text file found")
                        continue
                
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_lines = f.readlines()
                
                # Parse TSV for quotes and attributions
                quotes = {}  # quote_id -> quote_info
                attribs = {}  # quote_id -> speaker
                
                with open(tsv_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split('|')
                        if len(parts) < 3:
                            continue
                        
                        label = parts[0]
                        quote_id = parts[1]
                        
                        if label == 'QUOTE' and len(parts) >= 7:
                            # QUOTE|Q342|start_sent|start_tok|end_sent|end_tok|quote_text|
                            try:
                                start_sent = int(parts[2])
                                quote_text = parts[6] if len(parts) > 6 else ''
                                quotes[quote_id] = {
                                    'start_sent': start_sent,
                                    'quote_text': quote_text.strip(),
                                }
                            except (ValueError, IndexError):
                                continue
                        
                        elif label == 'ATTRIB' and len(parts) >= 3:
                            # ATTRIB|Q342|Speaker_Name-ID|
                            speaker = parts[2].strip()
                            # Clean speaker ID (remove trailing coref ID like "-3")
                            if '-' in speaker and speaker[-1].isdigit():
                                speaker = speaker.rsplit('-', 1)[0]
                            # Replace underscores with spaces for readability
                            speaker = speaker.replace('_', ' ')
                            if speaker:  # Only store non-empty speakers
                                attribs[quote_id] = speaker
                
                # Combine quotes with attributions - REQUIRE speaker for valid samples
                for quote_id, quote_info in quotes.items():
                    quote_text = quote_info['quote_text']
                    speaker = attribs.get(quote_id, '')
                    start_sent = quote_info['start_sent']
                    
                    # Skip samples without quote text or speaker (we need both for training)
                    if not quote_text or not speaker:
                        continue
                    
                    # Build context from surrounding sentences
                    ctx_start_sent = max(0, start_sent - 3)
                    ctx_end_sent = min(len(text_lines), start_sent + 4)
                    context = ''.join(text_lines[ctx_start_sent:ctx_end_sent])
                    
                    # Find quote position in context
                    search_text = quote_text[:30] if len(quote_text) > 30 else quote_text
                    quote_start = context.find(search_text)
                    if quote_start < 0:
                        quote_start = 0
                    
                    samples.append({
                        'text': context,
                        'quote': quote_text,
                        'quote_start': quote_start,
                        'quote_end': quote_start + len(quote_text),
                        'speaker': speaker,
                        'book_id': base_name,
                    })
                    
            except Exception as e:
                print(f"  Warning: Error loading {tsv_file}: {e}")
                continue
        
        print(f"  Loaded {len(samples)} quote-speaker pairs from LitBank")
        return samples

    
    def load_directquote(self, path: Path) -> List[Dict]:
        """Load DirectQuote dataset from CoNLL IOB1 format.
        
        DirectQuote uses IOB1 tagging in truecased.txt:
        - B-LeftSpeaker / I-LeftSpeaker: Quote with speaker before it
        - B-RightSpeaker / I-RightSpeaker: Quote with speaker after it
        - B-Unknown / I-Unknown: Quote with no speaker found
        - B-Speaker / I-Speaker: Speaker mention
        - O: Other tokens
        
        Paragraphs are separated by blank lines.
        """
        samples = []
        
        # DirectQuote has data/truecased.txt in CoNLL format
        data_file = path / "data" / "truecased.txt"
        if not data_file.exists():
            # Try finding it in the root or as copied file
            data_file = path / "truecased.txt"
            if not data_file.exists():
                txt_files = list(path.glob("*.txt"))
                if txt_files:
                    data_file = txt_files[0]
                else:
                    print(f"  Warning: DirectQuote data not found at {path}")
                    return samples
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  Warning: Could not read {data_file}: {e}")
            return samples
        
        # Parse CoNLL format - paragraphs separated by blank lines
        current_paragraph = []
        
        def process_paragraph(tokens_and_labels):
            """Extract quote-speaker pairs from a paragraph."""
            if not tokens_and_labels:
                return
            
            tokens = [t[0] for t in tokens_and_labels]
            labels = [t[1] for t in tokens_and_labels]
            
            # Find quote spans and speaker spans
            quotes = []  # [(start, end, quote_type)]
            speakers = []  # [(start, end)]
            
            i = 0
            while i < len(labels):
                label = labels[i]
                
                # Quote span
                if label.startswith('B-') and label[2:] in ('LeftSpeaker', 'RightSpeaker', 'Unknown'):
                    quote_type = label[2:]
                    start = i
                    i += 1
                    while i < len(labels) and labels[i] == f'I-{quote_type}':
                        i += 1
                    quotes.append((start, i, quote_type))
                
                # Speaker span
                elif label == 'B-Speaker':
                    start = i
                    i += 1
                    while i < len(labels) and labels[i] == 'I-Speaker':
                        i += 1
                    speakers.append((start, i))
                
                else:
                    i += 1
            
            # Build full text
            full_text = ' '.join(tokens)
            
            # Link quotes to speakers
            for q_start, q_end, q_type in quotes:
                quote_tokens = tokens[q_start:q_end]
                quote_text = ' '.join(quote_tokens)
                
                # Clean quote text (remove extra spaces around punctuation)
                quote_text = quote_text.replace(' "', '"').replace('" ', '"')
                quote_text = quote_text.replace(" '", "'").replace("' ", "'")
                
                speaker = ''
                
                if q_type == 'LeftSpeaker':
                    # Find nearest speaker BEFORE the quote
                    for s_start, s_end in reversed(speakers):
                        if s_end <= q_start:
                            speaker = ' '.join(tokens[s_start:s_end])
                            break
                
                elif q_type == 'RightSpeaker':
                    # Find nearest speaker AFTER the quote
                    for s_start, s_end in speakers:
                        if s_start >= q_end:
                            speaker = ' '.join(tokens[s_start:s_end])
                            break
                
                # ONLY add samples with valid speakers (required for quote attribution training)
                if speaker:
                    # Calculate quote position in full text
                    prefix_tokens = tokens[:q_start]
                    quote_start_pos = len(' '.join(prefix_tokens)) + (1 if prefix_tokens else 0)
                    
                    samples.append({
                        'text': full_text,
                        'quote': quote_text,
                        'speaker': speaker,
                        'quote_start': quote_start_pos,
                        'quote_end': quote_start_pos + len(quote_text),
                    })

        
        # Process file line by line
        for line in lines:
            line = line.strip()
            
            if not line:
                # End of paragraph
                process_paragraph(current_paragraph)
                current_paragraph = []
            else:
                # Token and label
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[1]
                    current_paragraph.append((token, label))
                elif len(parts) == 1:
                    # Token only, assume 'O'
                    current_paragraph.append((parts[0], 'O'))
        
        # Don't forget last paragraph
        process_paragraph(current_paragraph)
        
        print(f"  Loaded {len(samples)} quote-speaker pairs from DirectQuote")
        return samples
    
    # CURSOR: Quotebank loader disabled.
    # CURSOR: def load_quotebank(self, path: Path) -> List[Dict]:
    # CURSOR:     """Load Quotebank dataset (sampled).
    # CURSOR:     
    # CURSOR:     Quotebank is a large dataset (~175M quotes) from news that must be
    # CURSOR:     downloaded separately from: https://zenodo.org/record/4277311
    # CURSOR:     
    # CURSOR:     This loader handles the case where data is not present by skipping gracefully.
    # CURSOR:     """
    # CURSOR:     samples = []
    # CURSOR:     
    # CURSOR:     # Quotebank uses JSON Lines format
    # CURSOR:     data_files = sorted(path.glob("*.json*"))
    # CURSOR:     
    # CURSOR:     if not data_files:
    # CURSOR:         print(f"  Warning: Quotebank data not found at {path}")
    # CURSOR:         print("  To use Quotebank, download from: https://zenodo.org/record/4277311")
    # CURSOR:         print("  Skipping Quotebank dataset...")
    # CURSOR:         return samples
    # CURSOR:     
    # CURSOR:     print(f"  Found {len(data_files)} Quotebank files, loading...")
    # CURSOR:     
    # CURSOR:     for data_file in data_files:
    # CURSOR:         if self.max_samples and len(samples) >= self.max_samples:
    # CURSOR:             break
    # CURSOR:         try:
    # CURSOR:             with open(data_file, 'r', encoding='utf-8') as f:
    # CURSOR:                 for line in f:
    # CURSOR:                     if self.max_samples and len(samples) >= self.max_samples:
    # CURSOR:                         break
    # CURSOR:                     
    # CURSOR:                     try:
    # CURSOR:                         item = json.loads(line)
    # CURSOR:                         quote = item.get('quotation', '')
    # CURSOR:                         speaker = item.get('speaker', '')
    # CURSOR:                         
    # CURSOR:                         # Only include quotes with reasonable length and speaker
    # CURSOR:                         if quote and speaker and len(quote) > 20:
    # CURSOR:                             samples.append({
    # CURSOR:                                 'text': quote,  # Quotebank has limited context
    # CURSOR:                                 'quote': quote,
    # CURSOR:                                 'speaker': speaker,
    # CURSOR:                                 'quote_start': 0,
    # CURSOR:                                 'quote_end': len(quote),
    # CURSOR:                             })
    # CURSOR:                     except json.JSONDecodeError:
    # CURSOR:                         continue
    # CURSOR:         except Exception as e:
    # CURSOR:             print(f"  Warning: Error reading {data_file.name}: {e}")
    # CURSOR:             continue
    # CURSOR:     
    # CURSOR:     print(f"  Loaded {len(samples)} samples from Quotebank")
    # CURSOR:     return samples
    
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


def download_datasets(
    base_path: str,
    datasets: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Download and prepare datasets defined in MultiSourceDataLoader.
    
    Args:
        base_path: Base directory to place datasets
        datasets: Optional subset of dataset keys to download
    
    Returns:
        Mapping of dataset key to on-disk path
    """
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)

    targets = datasets or list(MultiSourceDataLoader.AVAILABLE_DATASETS.keys())
    resolved_paths: Dict[str, Path] = {}

    print(f"\n{'=' * 60}")
    print("AUTO-DOWNLOAD DATASETS")
    print(f"{'=' * 60}")
    print(f"Datasets to download: {targets}")

    for ds_name in targets:
        config = MultiSourceDataLoader.AVAILABLE_DATASETS.get(ds_name)
        if not config:
            print(f"⚠️  Unknown dataset '{ds_name}', skipping")
            continue

        if not config.download_url:
            print(f"⚠️  No download URL configured for {config.name}; skipping")
            continue

        ds_dir = base / config.path

        print(f"\n📦 Processing {config.name} ({ds_name})...")
        print(f"   Description: {config.description}")
        print(f"   Target directory: {ds_dir}")

        if config.download_url.endswith(".git") or "github.com" in config.download_url:
            _clone_repo(config.download_url, ds_dir)
        # CURSOR: Quotebank download disabled.
        # CURSOR: elif ds_name == 'quotebank':
        # CURSOR:     _prepare_quotebank(config.download_url, ds_dir)
        else:
            ds_dir.mkdir(parents=True, exist_ok=True)
            target_file = ds_dir / Path(config.download_url).name
            _download_file(config.download_url, target_file)
            print(f"✅ {config.name} ready at {target_file}")

        resolved_paths[ds_name] = ds_dir

    print(f"\n{'=' * 60}")
    print("DATASET DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"📚 Downloaded datasets: {list(resolved_paths.keys())}")
    print(f"📁 Base directory: {base}")
    print(f"{'=' * 60}\n")

    return resolved_paths


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
    
    print("\nDataset sizes:")
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
    'create_multi_source_dataloader',
    'download_datasets'
]
