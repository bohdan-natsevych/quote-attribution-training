#!/usr/bin/env python3
"""
Genre-Specific Adaptations with Fallback Strategies

Handles underperforming genres by:
1. Detecting genres below accuracy threshold
2. Applying genre-specific fine-tuning strategies
3. Implementing fallback rules for edge cases
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import defaultdict
from sklearn.metrics import accuracy_score
from typing import Dict, List, Optional, Tuple
import numpy as np


class GenreSpecificAdaptation:
    """
    Manages genre-specific model adaptations and fallback strategies.
    """
    
    def __init__(
        self, 
        min_accuracy_threshold: float = 0.75,
        fallback_strategies: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            min_accuracy_threshold: Minimum acceptable accuracy for any genre
            fallback_strategies: Dict mapping genre names to fallback strategy names
        """
        self.min_accuracy_threshold = min_accuracy_threshold
        self.genre_metrics = {}
        self.underperforming_genres = []
        
        # CURSOR: Default fallback strategies for common genre issues
        self.fallback_strategies = fallback_strategies or {
            'dialogue_heavy': 'context_expansion',
            'formal_prose': 'pronoun_resolution',
            'poetry': 'speaker_proximity',
            'news': 'quote_pattern_matching',
            'default': 'ensemble_voting'
        }
    
    def evaluate_genres(
        self, 
        model: nn.Module, 
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Evaluate model performance per genre.
        
        Returns:
            Dict mapping genre name to accuracy score
        """
        model.eval()
        genre_preds = defaultdict(list)
        genre_labels = defaultdict(list)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
                
                for i, genre in enumerate(batch.get('genre', ['unknown'] * len(preds))):
                    genre_preds[genre].append(preds[i] if i < len(preds) else 0)
                    genre_labels[genre].append(batch['labels'][i].item())
        
        # CURSOR: Calculate per-genre accuracy
        self.genre_metrics = {}
        self.underperforming_genres = []
        
        for genre in genre_preds:
            if len(genre_labels[genre]) > 0:
                acc = accuracy_score(genre_labels[genre], genre_preds[genre])
                self.genre_metrics[genre] = {
                    'accuracy': acc,
                    'sample_count': len(genre_labels[genre]),
                    'below_threshold': acc < self.min_accuracy_threshold
                }
                
                if acc < self.min_accuracy_threshold:
                    self.underperforming_genres.append(genre)
        
        return {g: m['accuracy'] for g, m in self.genre_metrics.items()}
    
    def get_adaptation_strategy(self, genre: str) -> str:
        """Get the recommended adaptation strategy for a genre."""
        if genre in self.fallback_strategies:
            return self.fallback_strategies[genre]
        return self.fallback_strategies.get('default', 'ensemble_voting')
    
    def create_genre_weighted_sampler(
        self,
        samples: List[Dict],
        boost_underperforming: float = 2.0
    ) -> WeightedRandomSampler:
        """
        Create a weighted sampler that oversamples underperforming genres.
        
        Args:
            samples: List of sample dicts with 'genre' key
            boost_underperforming: Multiplier for underperforming genre weights
        """
        genre_counts = defaultdict(int)
        for s in samples:
            genre_counts[s.get('genre', 'unknown')] += 1
        
        weights = []
        for s in samples:
            genre = s.get('genre', 'unknown')
            base_weight = 1.0 / genre_counts[genre]
            
            # CURSOR: Boost weight for underperforming genres
            if genre in self.underperforming_genres:
                base_weight *= boost_underperforming
            
            weights.append(base_weight)
        
        return WeightedRandomSampler(weights, len(samples))
    
    def apply_fallback_prediction(
        self,
        genre: str,
        text: str,
        model_prediction: int,
        model_confidence: float,
        candidates: List[str]
    ) -> Tuple[int, float]:
        """
        Apply fallback strategy for low-confidence predictions.
        
        Args:
            genre: Genre of the text
            text: Input text
            model_prediction: Model's predicted speaker index
            model_confidence: Model's confidence score
            candidates: List of candidate speaker names
            
        Returns:
            Tuple of (adjusted_prediction, adjusted_confidence)
        """
        # CURSOR: Only apply fallback for low confidence predictions
        CONFIDENCE_THRESHOLD = 0.6
        
        if model_confidence >= CONFIDENCE_THRESHOLD:
            return model_prediction, model_confidence
        
        strategy = self.get_adaptation_strategy(genre)
        
        if strategy == 'context_expansion':
            return self._fallback_context_expansion(text, candidates, model_prediction)
        elif strategy == 'pronoun_resolution':
            return self._fallback_pronoun_resolution(text, candidates, model_prediction)
        elif strategy == 'speaker_proximity':
            return self._fallback_speaker_proximity(text, candidates, model_prediction)
        elif strategy == 'quote_pattern_matching':
            return self._fallback_quote_pattern(text, candidates, model_prediction)
        else:
            return model_prediction, model_confidence
    
    def _fallback_context_expansion(
        self, 
        text: str, 
        candidates: List[str],
        default_pred: int
    ) -> Tuple[int, float]:
        """
        Fallback: Look for speaker mentions in expanded context.
        """
        text_lower = text.lower()
        
        for i, candidate in enumerate(candidates):
            if candidate.lower() in text_lower:
                # CURSOR: Found candidate in context, boost confidence
                return i, 0.65
        
        return default_pred, 0.5
    
    def _fallback_pronoun_resolution(
        self, 
        text: str, 
        candidates: List[str],
        default_pred: int
    ) -> Tuple[int, float]:
        """
        Fallback: Simple pronoun-based resolution.
        """
        import re
        
        # CURSOR: Check for gendered pronouns near the quote
        male_pronouns = re.findall(r'\b(he|him|his)\b', text.lower())
        female_pronouns = re.findall(r'\b(she|her|hers)\b', text.lower())
        
        # CURSOR: Basic heuristic - this would be more sophisticated in production
        if len(male_pronouns) > len(female_pronouns) * 2:
            # Likely male speaker, prefer male-sounding names
            pass
        elif len(female_pronouns) > len(male_pronouns) * 2:
            # Likely female speaker
            pass
        
        return default_pred, 0.55
    
    def _fallback_speaker_proximity(
        self, 
        text: str, 
        candidates: List[str],
        default_pred: int
    ) -> Tuple[int, float]:
        """
        Fallback: Prefer speaker mentioned closest to the quote.
        """
        # CURSOR: Find quote markers
        import re
        quote_match = re.search(r'["\']([^"\']+)["\']', text)
        
        if not quote_match:
            return default_pred, 0.5
        
        quote_start = quote_match.start()
        
        # CURSOR: Find closest candidate mention
        best_candidate = default_pred
        min_distance = float('inf')
        
        for i, candidate in enumerate(candidates):
            matches = list(re.finditer(re.escape(candidate), text, re.IGNORECASE))
            for m in matches:
                dist = abs(m.start() - quote_start)
                if dist < min_distance:
                    min_distance = dist
                    best_candidate = i
        
        confidence = 0.7 if min_distance < 50 else 0.55
        return best_candidate, confidence
    
    def _fallback_quote_pattern(
        self, 
        text: str, 
        candidates: List[str],
        default_pred: int
    ) -> Tuple[int, float]:
        """
        Fallback: Match common quote attribution patterns.
        """
        import re
        
        # CURSOR: Common patterns like "said X" or "X said"
        patterns = [
            r'said\s+(\w+)',
            r'(\w+)\s+said',
            r'according\s+to\s+(\w+)',
            r'(\w+)\s+told',
            r'(\w+)\s+replied',
            r'(\w+)\s+asked',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                speaker_name = match.group(1)
                for i, candidate in enumerate(candidates):
                    if speaker_name.lower() in candidate.lower():
                        return i, 0.75
        
        return default_pred, 0.5
    
    def get_report(self) -> str:
        """Generate a human-readable report of genre performance."""
        lines = [
            "Genre Performance Report",
            "=" * 40,
            f"Threshold: {self.min_accuracy_threshold:.0%}",
            ""
        ]
        
        for genre, metrics in sorted(self.genre_metrics.items(), 
                                     key=lambda x: x[1]['accuracy']):
            status = "[WARNING] BELOW" if metrics['below_threshold'] else "[OK]"
            lines.append(
                f"{status} {genre}: {metrics['accuracy']:.2%} "
                f"(n={metrics['sample_count']})"
            )
        
        if self.underperforming_genres:
            lines.extend([
                "",
                "Recommended Actions:",
                "-" * 40
            ])
            for genre in self.underperforming_genres:
                strategy = self.get_adaptation_strategy(genre)
                lines.append(f"  {genre}: Apply '{strategy}' strategy")
        
        return "\n".join(lines)


class GenreSpecificFineTuner:
    """
    Fine-tunes model on specific underperforming genres.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-5,
        epochs: int = 3
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fine_tune_on_genre(
        self,
        genre: str,
        dataloader: DataLoader,
        device: torch.device
    ):
        """
        Fine-tune the model specifically on samples from one genre.
        
        CURSOR: Use lower learning rate to avoid catastrophic forgetting.
        """
        from torch.optim import AdamW
        
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        print(f"Fine-tuning on genre: {genre}")
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                # CURSOR: Only use samples from target genre
                if hasattr(batch, 'genre') and batch.get('genre', [None])[0] != genre:
                    continue
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")


# Example usage
if __name__ == '__main__':
    # Create adapter
    adapter = GenreSpecificAdaptation(min_accuracy_threshold=0.75)
    
    # Simulate genre metrics
    adapter.genre_metrics = {
        'fiction': {'accuracy': 0.85, 'sample_count': 1000, 'below_threshold': False},
        'poetry': {'accuracy': 0.68, 'sample_count': 200, 'below_threshold': True},
        'news': {'accuracy': 0.82, 'sample_count': 500, 'below_threshold': False},
        'dialogue_heavy': {'accuracy': 0.71, 'sample_count': 300, 'below_threshold': True},
    }
    adapter.underperforming_genres = ['poetry', 'dialogue_heavy']
    
    print(adapter.get_report())
    
    # Test fallback
    pred, conf = adapter.apply_fallback_prediction(
        genre='poetry',
        text='"Where are you going?" she asked. Mary turned around.',
        model_prediction=0,
        model_confidence=0.45,
        candidates=['John', 'Mary', 'Sarah']
    )
    print(f"\nFallback result: speaker={pred}, confidence={conf:.2f}")



