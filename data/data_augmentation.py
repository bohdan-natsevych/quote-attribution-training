"""
Data Augmentation Strategies for Quote Attribution

Includes 8+ augmentation strategies:
1. Synonym replacement
2. Random word insertion
3. Random word swap
4. Random word deletion
5. Back-translation (placeholder - requires external API)
6. Context extension
7. Context truncation
8. Candidate shuffling
9. Entity masking
10. Quote paraphrasing (placeholder)

Each augmentation preserves the quote-speaker relationship.
"""

import random
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from copy import deepcopy

# CURSOR: Try to import optional dependencies
try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import nlpaug.augmenter.word as naw
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False


@dataclass
class AugmentedSample:
    """Container for an augmented sample."""
    text: str
    quote_start: int
    quote_end: int
    speaker_idx: int
    candidates: List[str]
    candidate_positions: List[Tuple[int, int]]
    augmentation_type: str
    original_text: Optional[str] = None


class QuoteAugmenter:
    """
    Augmentation pipeline for quote attribution data.
    
    Applies multiple augmentation strategies while preserving
    the quote-speaker relationship.
    """
    
    def __init__(
        self,
        synonym_replace_prob: float = 0.15,
        random_insert_prob: float = 0.1,
        random_swap_prob: float = 0.1,
        random_delete_prob: float = 0.1,
        context_extend_prob: float = 0.3,
        candidate_shuffle_prob: float = 0.5,
        entity_mask_prob: float = 0.1,
        max_augmentations_per_sample: int = 4,
        seed: Optional[int] = None
    ):
        """
        Initialize augmenter.
        
        Args:
            synonym_replace_prob: Probability of replacing a word with synonym
            random_insert_prob: Probability of inserting a random word
            random_swap_prob: Probability of swapping adjacent words
            random_delete_prob: Probability of deleting a word
            context_extend_prob: Probability of extending context
            candidate_shuffle_prob: Probability of shuffling candidates
            entity_mask_prob: Probability of masking entities
            max_augmentations_per_sample: Max augmentations to apply
            seed: Random seed for reproducibility
        """
        self.synonym_replace_prob = synonym_replace_prob
        self.random_insert_prob = random_insert_prob
        self.random_swap_prob = random_swap_prob
        self.random_delete_prob = random_delete_prob
        self.context_extend_prob = context_extend_prob
        self.candidate_shuffle_prob = candidate_shuffle_prob
        self.entity_mask_prob = entity_mask_prob
        self.max_augmentations_per_sample = max_augmentations_per_sample
        
        if seed is not None:
            random.seed(seed)
        
        # CURSOR: Initialize NLTK if available
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # CURSOR: Initialize nlpaug augmenters if available
        self.nlpaug_synonym = None
        self.nlpaug_insert = None
        if NLPAUG_AVAILABLE:
            try:
                self.nlpaug_synonym = naw.SynonymAug(aug_src='wordnet')
                self.nlpaug_insert = naw.ContextualWordEmbsAug(
                    model_path='bert-base-uncased',
                    action='insert'
                )
            except Exception:
                pass  # Fall back to basic augmentation
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        if not NLTK_AVAILABLE:
            return []
        
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.append(synonym)
        
        return list(set(synonyms))[:5]  # Limit to 5 synonyms

    @staticmethod
    def _get_word_positions(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """CURSOR: Return (words, [(start,end)]) for simple whitespace tokenization."""
        words = text.split()
        positions: List[Tuple[int, int]] = []
        pos = 0
        for word in words:
            start = text.find(word, pos)
            end = start + len(word)
            positions.append((start, end))
            pos = end
        return words, positions

    @staticmethod
    def _overlaps_protected(start: int, end: int, protected_spans: List[Tuple[int, int]]) -> bool:
        """CURSOR: True if [start,end) overlaps any protected span."""
        for ps, pe in protected_spans:
            if start < pe and end > ps:
                return True
        return False
    
    def synonym_replace(
        self,
        text: str,
        protected_spans: List[Tuple[int, int]],
        n: int = 1
    ) -> str:
        """
        Replace random words with synonyms (excluding protected spans).
        
        Args:
            text: Input text
            protected_spans: List of (start, end) spans to protect
            n: Number of replacements to attempt
            
        Returns:
            Augmented text
        """
        words, word_positions = self._get_word_positions(text)
        
        # CURSOR: Find words that can be replaced (not in protected spans)
        replaceable = []
        for i, (start, end) in enumerate(word_positions):
            is_protected = self._overlaps_protected(start, end, protected_spans)
            if not is_protected and len(words[i]) > 3:
                replaceable.append(i)
        
        if not replaceable:
            return text
        
        # CURSOR: Replace random words
        random.shuffle(replaceable)
        num_replaced = 0
        
        for idx in replaceable[:n]:
            word = words[idx]
            synonyms = self.get_synonyms(word.lower())
            if synonyms:
                words[idx] = random.choice(synonyms)
                num_replaced += 1
        
        return ' '.join(words) if num_replaced > 0 else text
    
    def random_insert(
        self,
        text: str,
        protected_spans: List[Tuple[int, int]],
        n: int = 1
    ) -> str:
        """
        Insert random words at random positions.
        
        Args:
            text: Input text
            protected_spans: List of (start, end) spans to protect
            n: Number of insertions
            
        Returns:
            Augmented text
        """
        # CURSOR: Simple word insertion using common filler words
        filler_words = [
            'actually', 'really', 'simply', 'quite', 'rather',
            'indeed', 'certainly', 'perhaps', 'probably', 'surely'
        ]
        
        words, positions = self._get_word_positions(text)
        if not words:
            return text

        # CURSOR: Choose insertion positions outside protected spans (best-effort).
        def insertion_char_pos(insert_idx: int) -> int:
            if insert_idx <= 0:
                return 0
            # Approximate insertion point as right after the previous token.
            return positions[min(insert_idx - 1, len(positions) - 1)][1]

        allowed_positions = [
            i for i in range(len(words) + 1)
            if not self._overlaps_protected(insertion_char_pos(i), insertion_char_pos(i) + 1, protected_spans)
        ]
        if not allowed_positions:
            return text

        for _ in range(n):
            insert_pos = random.choice(allowed_positions)
            filler = random.choice(filler_words)
            words.insert(insert_pos, filler)

        return ' '.join(words)
    
    def random_swap(
        self,
        text: str,
        protected_spans: List[Tuple[int, int]],
        n: int = 1
    ) -> str:
        """
        Swap random adjacent word pairs.
        
        Args:
            text: Input text
            protected_spans: List of (start, end) spans to protect
            n: Number of swaps
            
        Returns:
            Augmented text
        """
        words, positions = self._get_word_positions(text)
        if len(words) < 2:
            return text

        # CURSOR: Swap only unprotected adjacent pairs.
        def is_word_protected(word_idx: int) -> bool:
            if not (0 <= word_idx < len(positions)):
                return False
            start, end = positions[word_idx]
            return self._overlaps_protected(start, end, protected_spans)

        swappable = [
            i for i in range(len(words) - 1)
            if not is_word_protected(i) and not is_word_protected(i + 1)
        ]
        if not swappable:
            return text

        for _ in range(n):
            idx = random.choice(swappable)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]

        return ' '.join(words)
    
    def random_delete(
        self,
        text: str,
        protected_spans: List[Tuple[int, int]],
        p: float = 0.1
    ) -> str:
        """
        Randomly delete words with probability p.
        
        Args:
            text: Input text
            protected_spans: List of (start, end) spans to protect
            p: Deletion probability per word
            
        Returns:
            Augmented text
        """
        words, positions = self._get_word_positions(text)
        if len(words) < 3:
            return text

        protected_mask = [
            self._overlaps_protected(start, end, protected_spans)
            for start, end in positions
        ]
        unprotected_indices = [i for i, is_prot in enumerate(protected_mask) if not is_prot]
        if not unprotected_indices:
            return text

        keep_mask = protected_mask[:]  # protected always kept
        for i in unprotected_indices:
            keep_mask[i] = (random.random() > p)

        # CURSOR: Ensure we don't delete too aggressively (keep at least half of unprotected words).
        min_keep_unprotected = max(1, len(unprotected_indices) // 2)
        kept_unprotected = [i for i in unprotected_indices if keep_mask[i]]
        if len(kept_unprotected) < min_keep_unprotected:
            deleted_unprotected = [i for i in unprotected_indices if not keep_mask[i]]
            random.shuffle(deleted_unprotected)
            for i in deleted_unprotected[: (min_keep_unprotected - len(kept_unprotected))]:
                keep_mask[i] = True

        new_words = [w for i, w in enumerate(words) if keep_mask[i]]
        return ' '.join(new_words) if new_words else text
    
    def context_extend(
        self,
        context_before: str,
        context_after: str,
        extended_before: str = "",
        extended_after: str = "",
        tokens: int = 50
    ) -> Tuple[str, str]:
        """
        Extend context window.
        
        Args:
            context_before: Current context before quote
            context_after: Current context after quote
            extended_before: Additional context available before
            extended_after: Additional context available after
            tokens: Number of tokens to add
            
        Returns:
            Extended (context_before, context_after)
        """
        # CURSOR: Add tokens from extended context
        if extended_before:
            extra_words = extended_before.split()[-tokens//2:]
            context_before = ' '.join(extra_words) + ' ' + context_before
        
        if extended_after:
            extra_words = extended_after.split()[:tokens//2]
            context_after = context_after + ' ' + ' '.join(extra_words)
        
        return context_before.strip(), context_after.strip()
    
    def context_truncate(
        self,
        context: str,
        max_tokens: int = 100
    ) -> str:
        """
        Truncate context to max tokens.
        
        Args:
            context: Input context
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated context
        """
        words = context.split()
        if len(words) <= max_tokens:
            return context
        
        return ' '.join(words[:max_tokens])
    
    def shuffle_candidates(
        self,
        candidates: List[str],
        correct_idx: int
    ) -> Tuple[List[str], int]:
        """
        Shuffle candidate order while tracking correct answer.
        
        Args:
            candidates: List of candidate names
            correct_idx: Index of correct speaker
            
        Returns:
            (shuffled_candidates, new_correct_idx)
        """
        indexed = list(enumerate(candidates))
        random.shuffle(indexed)
        
        new_candidates = [c for _, c in indexed]
        new_correct_idx = [i for i, (orig_i, _) in enumerate(indexed) if orig_i == correct_idx][0]
        
        return new_candidates, new_correct_idx
    
    def mask_entities(
        self,
        text: str,
        entities: List[str],
        mask_token: str = "[MASK]"
    ) -> str:
        """
        Mask entity mentions with a special token.
        
        Args:
            text: Input text
            entities: List of entity names to mask
            mask_token: Token to use for masking
            
        Returns:
            Text with masked entities
        """
        masked_text = text
        for entity in entities:
            # CURSOR: Case-insensitive replacement
            pattern = re.compile(re.escape(entity), re.IGNORECASE)
            masked_text = pattern.sub(mask_token, masked_text)
        
        return masked_text
    
    def augment_sample(
        self,
        sample: Dict[str, Any],
        num_augmentations: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Apply multiple augmentations to a single sample.
        
        Args:
            sample: Original sample dictionary with keys:
                - text: Full text
                - quote: Quote text
                - quote_start, quote_end: Quote position
                - speaker: Correct speaker
                - candidates: List of candidate speakers
            num_augmentations: Number of augmented versions to create
            
        Returns:
            List of augmented samples
        """
        augmented = []
        
        # CURSOR: Available augmentation strategies
        strategies = [
            ('synonym_replace', self._apply_synonym_replace),
            ('random_insert', self._apply_random_insert),
            ('random_swap', self._apply_random_swap),
            ('random_delete', self._apply_random_delete),
            ('candidate_shuffle', self._apply_candidate_shuffle),
        ]
        
        # CURSOR: Select and apply random strategies
        num_to_apply = min(num_augmentations, self.max_augmentations_per_sample)
        selected = random.sample(strategies, min(len(strategies), num_to_apply))
        
        for strategy_name, strategy_fn in selected:
            try:
                aug_sample = strategy_fn(deepcopy(sample))
                if aug_sample is not None:
                    aug_sample['augmentation'] = strategy_name
                    augmented.append(aug_sample)
            except Exception as e:
                # CURSOR: Skip failed augmentations silently
                continue
        
        return augmented
    
    def _apply_synonym_replace(self, sample: Dict) -> Dict:
        """Apply synonym replacement to context (not quote)."""
        if random.random() > self.synonym_replace_prob:
            return None
        
        text = sample.get('text', '')
        quote_start = sample.get('quote_start', 0)
        quote_end = sample.get('quote_end', len(text))
        
        # CURSOR: Protect the quote from modification
        protected = [(quote_start, quote_end)]
        
        sample['text'] = self.synonym_replace(text, protected, n=2)
        return sample
    
    def _apply_random_insert(self, sample: Dict) -> Dict:
        """Apply random word insertion."""
        if random.random() > self.random_insert_prob:
            return None
        
        text = sample.get('text', '')
        quote_start = sample.get('quote_start', 0)
        quote_end = sample.get('quote_end', len(text))
        
        protected = [(quote_start, quote_end)]
        sample['text'] = self.random_insert(text, protected, n=1)
        return sample
    
    def _apply_random_swap(self, sample: Dict) -> Dict:
        """Apply random word swapping."""
        if random.random() > self.random_swap_prob:
            return None
        
        text = sample.get('text', '')
        quote_start = sample.get('quote_start', 0)
        quote_end = sample.get('quote_end', len(text))
        
        protected = [(quote_start, quote_end)]
        sample['text'] = self.random_swap(text, protected, n=1)
        return sample
    
    def _apply_random_delete(self, sample: Dict) -> Dict:
        """Apply random word deletion."""
        if random.random() > self.random_delete_prob:
            return None
        
        text = sample.get('text', '')
        quote_start = sample.get('quote_start', 0)
        quote_end = sample.get('quote_end', len(text))
        
        protected = [(quote_start, quote_end)]
        sample['text'] = self.random_delete(text, protected, p=0.05)
        return sample
    
    def _apply_candidate_shuffle(self, sample: Dict) -> Dict:
        """Apply candidate shuffling."""
        if random.random() > self.candidate_shuffle_prob:
            return None
        
        candidates = sample.get('candidates', [])
        correct_idx = sample.get('speaker_idx', 0)
        
        if len(candidates) < 2:
            return None
        
        new_candidates, new_idx = self.shuffle_candidates(candidates, correct_idx)
        sample['candidates'] = new_candidates
        sample['speaker_idx'] = new_idx
        return sample


def augment_dataset(
    samples: List[Dict],
    augmenter: Optional[QuoteAugmenter] = None,
    augmentations_per_sample: int = 3,
    include_original: bool = True
) -> List[Dict]:
    """
    Augment an entire dataset.
    
    Args:
        samples: List of sample dictionaries
        augmenter: QuoteAugmenter instance (creates default if None)
        augmentations_per_sample: Number of augmented versions per sample
        include_original: Whether to include original samples
        
    Returns:
        Augmented dataset
    """
    if augmenter is None:
        augmenter = QuoteAugmenter()
    
    augmented_dataset = []
    
    for sample in samples:
        # CURSOR: Add original sample
        if include_original:
            original = deepcopy(sample)
            original['augmentation'] = 'original'
            augmented_dataset.append(original)
        
        # CURSOR: Add augmented samples
        aug_samples = augmenter.augment_sample(sample, augmentations_per_sample)
        augmented_dataset.extend(aug_samples)
    
    return augmented_dataset


# CURSOR: Export public API
__all__ = [
    'QuoteAugmenter',
    'AugmentedSample',
    'augment_dataset'
]



