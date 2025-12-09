"""
Cross-Domain Validation for Quote Attribution

Multi-genre testing to ensure model generalizes:
- Per-genre validation metrics
- Domain shift detection
- Genre-specific thresholds
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class GenreMetrics:
    """Metrics for a single genre."""
    genre: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    num_samples: int
    passed: bool  # Met threshold


class CrossDomainValidator:
    """
    Validate model performance across different genres/domains.
    """
    
    # CURSOR: Default thresholds
    DEFAULT_THRESHOLDS = {
        'accuracy': 0.75,
        'f1': 0.70,
        'min_samples': 100,
    }
    
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        genres: Optional[List[str]] = None
    ):
        """
        Initialize validator.
        
        Args:
            thresholds: Custom thresholds for metrics
            genres: List of expected genres
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.expected_genres = genres or []
        
        self.predictions_by_genre: Dict[str, List] = defaultdict(list)
        self.labels_by_genre: Dict[str, List] = defaultdict(list)
        self.genre_metrics: Dict[str, GenreMetrics] = {}
    
    def add_predictions(
        self,
        predictions: List,
        labels: List,
        genres: List[str]
    ):
        """
        Add predictions for validation.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            genres: Genre for each sample
        """
        for pred, label, genre in zip(predictions, labels, genres):
            self.predictions_by_genre[genre].append(pred)
            self.labels_by_genre[genre].append(label)
    
    def compute_metrics(self) -> Dict[str, GenreMetrics]:
        """
        Compute metrics for each genre.
        
        Returns:
            Dictionary mapping genre to metrics
        """
        for genre in self.predictions_by_genre:
            preds = np.array(self.predictions_by_genre[genre])
            labels = np.array(self.labels_by_genre[genre])
            
            if len(preds) < self.thresholds.get('min_samples', 10):
                continue
            
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted', zero_division=0)
            precision = precision_score(labels, preds, average='weighted', zero_division=0)
            recall = recall_score(labels, preds, average='weighted', zero_division=0)
            
            passed = (
                accuracy >= self.thresholds.get('accuracy', 0.75) and
                f1 >= self.thresholds.get('f1', 0.70)
            )
            
            self.genre_metrics[genre] = GenreMetrics(
                genre=genre,
                accuracy=accuracy,
                f1=f1,
                precision=precision,
                recall=recall,
                num_samples=len(preds),
                passed=passed
            )
        
        return self.genre_metrics
    
    def get_overall_metrics(self) -> Dict[str, float]:
        """
        Get aggregated metrics across all genres.
        
        Returns:
            Dictionary of overall metrics
        """
        all_preds = []
        all_labels = []
        
        for genre in self.predictions_by_genre:
            all_preds.extend(self.predictions_by_genre[genre])
            all_labels.extend(self.labels_by_genre[genre])
        
        if not all_preds:
            return {}
        
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        
        return {
            'overall_accuracy': accuracy_score(labels, preds),
            'overall_f1': f1_score(labels, preds, average='weighted', zero_division=0),
            'overall_precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'overall_recall': recall_score(labels, preds, average='weighted', zero_division=0),
            'total_samples': len(preds),
            'num_genres': len(self.predictions_by_genre),
        }
    
    def get_failing_genres(self) -> List[str]:
        """Get genres that didn't meet thresholds."""
        if not self.genre_metrics:
            self.compute_metrics()
        
        return [
            genre for genre, metrics in self.genre_metrics.items()
            if not metrics.passed
        ]
    
    def get_domain_shift_score(self) -> float:
        """
        Calculate domain shift score.
        
        Higher score indicates more variation across genres.
        
        Returns:
            Domain shift score (0-1)
        """
        if not self.genre_metrics:
            self.compute_metrics()
        
        if len(self.genre_metrics) < 2:
            return 0.0
        
        accuracies = [m.accuracy for m in self.genre_metrics.values()]
        
        # CURSOR: Standard deviation normalized by mean
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        return std_acc / mean_acc if mean_acc > 0 else 0.0
    
    def validate(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run full validation.
        
        Returns:
            (passed, detailed_results)
        """
        self.compute_metrics()
        overall = self.get_overall_metrics()
        failing = self.get_failing_genres()
        domain_shift = self.get_domain_shift_score()
        
        # CURSOR: Check if validation passed
        passed = (
            overall.get('overall_accuracy', 0) >= self.thresholds['accuracy'] and
            len(failing) == 0 and
            domain_shift < 0.15  # Less than 15% variation
        )
        
        results = {
            'passed': passed,
            'overall_metrics': overall,
            'genre_metrics': {
                g: {
                    'accuracy': m.accuracy,
                    'f1': m.f1,
                    'samples': m.num_samples,
                    'passed': m.passed
                }
                for g, m in self.genre_metrics.items()
            },
            'failing_genres': failing,
            'domain_shift_score': domain_shift,
            'thresholds': self.thresholds,
        }
        
        return passed, results
    
    def print_report(self):
        """Print validation report."""
        passed, results = self.validate()
        
        print("\n" + "=" * 60)
        print("CROSS-DOMAIN VALIDATION REPORT")
        print("=" * 60)
        
        overall = results['overall_metrics']
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {overall.get('overall_accuracy', 0):.4f}")
        print(f"  F1 Score: {overall.get('overall_f1', 0):.4f}")
        print(f"  Samples: {overall.get('total_samples', 0)}")
        
        print(f"\nPer-Genre Metrics:")
        for genre, metrics in sorted(results['genre_metrics'].items()):
            status = "✓" if metrics['passed'] else "✗"
            print(f"  {status} {genre}: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, n={metrics['samples']}")
        
        print(f"\nDomain Shift Score: {results['domain_shift_score']:.4f}")
        
        if results['failing_genres']:
            print(f"\nFailing Genres: {', '.join(results['failing_genres'])}")
        
        print(f"\nValidation {'PASSED' if passed else 'FAILED'}")
        print("=" * 60)
    
    def export_report(self, output_path: str):
        """Export validation report to JSON."""
        passed, results = self.validate()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Report saved to {output_path}")


def validate_model_cross_domain(
    model,
    val_dataloaders: Dict[str, Any],
    device: str = 'cuda'
) -> Tuple[bool, Dict]:
    """
    Validate model across multiple domains.
    
    Args:
        model: Model to validate
        val_dataloaders: Dict mapping genre to DataLoader
        device: Computation device
        
    Returns:
        (passed, results)
    """
    import torch
    
    validator = CrossDomainValidator()
    model.eval()
    
    with torch.no_grad():
        for genre, dataloader in val_dataloaders.items():
            preds_list = []
            labels_list = []
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                outputs = model(input_ids, attention_mask)
                logits = outputs.get('logits', outputs)
                
                if hasattr(logits, 'last_hidden_state'):
                    logits = logits.last_hidden_state[:, 0, :]
                
                preds = (logits.mean(dim=-1) > 0).long().cpu().numpy()
                
                preds_list.extend(preds)
                labels_list.extend(labels.numpy())
            
            genres = [genre] * len(preds_list)
            validator.add_predictions(preds_list, labels_list, genres)
    
    validator.print_report()
    return validator.validate()


# CURSOR: Export public API
__all__ = [
    'GenreMetrics',
    'CrossDomainValidator',
    'validate_model_cross_domain'
]



