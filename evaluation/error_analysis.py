"""
Error Analysis for Quote Attribution

Systematic categorization of model errors:
- Error type classification
- Pattern detection
- Genre-specific analysis
- Actionable recommendations
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum

import numpy as np


class ErrorType(Enum):
    """Categories of attribution errors."""
    PRONOUN_CONFUSION = "pronoun_confusion"
    NEARBY_SPEAKER = "nearby_speaker"
    DIALOGUE_CONTINUATION = "dialogue_continuation"
    NESTED_QUOTE = "nested_quote"
    AMBIGUOUS_CONTEXT = "ambiguous_context"
    MULTIPLE_CANDIDATES = "multiple_candidates"
    RARE_SPEAKER = "rare_speaker"
    LONG_DISTANCE = "long_distance"
    GENRE_SPECIFIC = "genre_specific"
    UNKNOWN = "unknown"


@dataclass
class ErrorCase:
    """Single error case for analysis."""
    sample_id: str
    text: str
    quote: str
    predicted_speaker: str
    actual_speaker: str
    candidates: List[str]
    error_type: ErrorType
    confidence: float
    genre: str = "unknown"
    context_length: int = 0
    metadata: Dict = field(default_factory=dict)


class ErrorAnalyzer:
    """
    Analyze and categorize model prediction errors.
    """
    
    # CURSOR: Patterns for error classification
    PRONOUN_PATTERNS = {'he', 'she', 'they', 'him', 'her', 'them', 'i', 'me', 'we'}
    DIALOGUE_VERBS = {'said', 'asked', 'replied', 'answered', 'exclaimed', 'whispered'}
    
    def __init__(self):
        self.errors: List[ErrorCase] = []
        self.error_counts: Dict[ErrorType, int] = defaultdict(int)
        self.genre_errors: Dict[str, List[ErrorCase]] = defaultdict(list)
    
    def add_error(
        self,
        sample_id: str,
        text: str,
        quote: str,
        predicted: str,
        actual: str,
        candidates: List[str],
        confidence: float,
        genre: str = "unknown"
    ):
        """
        Add an error case for analysis.
        
        Args:
            sample_id: Unique identifier for the sample
            text: Full context text
            quote: The quote being attributed
            predicted: Model's prediction
            actual: Ground truth speaker
            candidates: List of candidate speakers
            confidence: Model's confidence score
            genre: Genre of the source
        """
        # CURSOR: Classify error type
        error_type = self._classify_error(
            text, quote, predicted, actual, candidates
        )
        
        error = ErrorCase(
            sample_id=sample_id,
            text=text,
            quote=quote,
            predicted_speaker=predicted,
            actual_speaker=actual,
            candidates=candidates,
            error_type=error_type,
            confidence=confidence,
            genre=genre,
            context_length=len(text.split())
        )
        
        self.errors.append(error)
        self.error_counts[error_type] += 1
        self.genre_errors[genre].append(error)
    
    def _classify_error(
        self,
        text: str,
        quote: str,
        predicted: str,
        actual: str,
        candidates: List[str]
    ) -> ErrorType:
        """Classify the type of error."""
        text_lower = text.lower()
        
        # CURSOR: Check for pronoun confusion
        quote_start = text.find(quote)
        if quote_start > 0:
            context_before = text[:quote_start].lower()
            words_before = context_before.split()[-20:]  # Last 20 words
            pronoun_density = sum(1 for w in words_before if w in self.PRONOUN_PATTERNS) / max(len(words_before), 1)
            if pronoun_density > 0.15:
                return ErrorType.PRONOUN_CONFUSION
        
        # CURSOR: Check for nearby speaker error
        if predicted in text:
            pred_pos = text.find(predicted)
            actual_pos = text.find(actual) if actual in text else -1
            if quote_start > 0:
                pred_dist = abs(pred_pos - quote_start)
                actual_dist = abs(actual_pos - quote_start) if actual_pos >= 0 else float('inf')
                if pred_dist < actual_dist:
                    return ErrorType.NEARBY_SPEAKER
        
        # CURSOR: Check for dialogue continuation
        if any(verb in text_lower for verb in self.DIALOGUE_VERBS):
            # Count dialogue markers
            dialogue_count = text.count('"') + text.count("'")
            if dialogue_count > 4:
                return ErrorType.DIALOGUE_CONTINUATION
        
        # CURSOR: Check for nested quotes
        if text.count('"') > 4 or ('\"' in quote and '"' in text.replace(quote, '')):
            return ErrorType.NESTED_QUOTE
        
        # CURSOR: Check for multiple candidates
        if len(candidates) > 5:
            return ErrorType.MULTIPLE_CANDIDATES
        
        # CURSOR: Check for long distance
        if actual in text:
            actual_pos = text.find(actual)
            if abs(actual_pos - quote_start) > len(text) * 0.4:
                return ErrorType.LONG_DISTANCE
        
        return ErrorType.UNKNOWN
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of errors.
        
        Returns:
            Dictionary with error statistics
        """
        total = len(self.errors)
        
        summary = {
            'total_errors': total,
            'error_type_distribution': {
                et.value: count / total if total > 0 else 0
                for et, count in self.error_counts.items()
            },
            'genre_distribution': {
                genre: len(errors) / total if total > 0 else 0
                for genre, errors in self.genre_errors.items()
            },
            'avg_confidence_on_errors': np.mean([e.confidence for e in self.errors]) if self.errors else 0,
            'high_confidence_errors': sum(1 for e in self.errors if e.confidence > 0.8) / total if total > 0 else 0,
        }
        
        return summary
    
    def get_top_error_patterns(self, n: int = 5) -> List[Tuple[ErrorType, int, float]]:
        """
        Get most common error patterns.
        
        Args:
            n: Number of patterns to return
            
        Returns:
            List of (error_type, count, percentage) tuples
        """
        total = len(self.errors)
        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            (et, count, count / total if total > 0 else 0)
            for et, count in sorted_errors[:n]
        ]
    
    def get_genre_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Get per-genre error analysis.
        
        Returns:
            Dictionary mapping genre to error statistics
        """
        analysis = {}
        
        for genre, errors in self.genre_errors.items():
            error_types = Counter(e.error_type for e in errors)
            
            analysis[genre] = {
                'total_errors': len(errors),
                'most_common_error': error_types.most_common(1)[0] if error_types else None,
                'avg_confidence': np.mean([e.confidence for e in errors]) if errors else 0,
                'error_type_counts': {
                    et.value: count for et, count in error_types.items()
                }
            }
        
        return analysis
    
    def get_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations based on errors.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        summary = self.get_summary()
        
        # CURSOR: Check for pronoun issues
        pronoun_pct = summary['error_type_distribution'].get(ErrorType.PRONOUN_CONFUSION.value, 0)
        if pronoun_pct > 0.2:
            recommendations.append(
                f"High pronoun confusion ({pronoun_pct:.0%}): Consider adding coreference resolution "
                "or pronoun-specific training data."
            )
        
        # CURSOR: Check for dialogue continuation
        dialogue_pct = summary['error_type_distribution'].get(ErrorType.DIALOGUE_CONTINUATION.value, 0)
        if dialogue_pct > 0.15:
            recommendations.append(
                f"Dialogue continuation errors ({dialogue_pct:.0%}): Implement dialogue tracking "
                "in post-processing rules."
            )
        
        # CURSOR: Check for nearby speaker errors
        nearby_pct = summary['error_type_distribution'].get(ErrorType.NEARBY_SPEAKER.value, 0)
        if nearby_pct > 0.15:
            recommendations.append(
                f"Nearby speaker confusion ({nearby_pct:.0%}): Increase context window "
                "or add positional features."
            )
        
        # CURSOR: Check for high confidence errors
        if summary['high_confidence_errors'] > 0.3:
            recommendations.append(
                f"Many high-confidence errors ({summary['high_confidence_errors']:.0%}): "
                "Model may be overconfident. Consider temperature scaling or calibration."
            )
        
        # CURSOR: Genre-specific recommendations
        genre_analysis = self.get_genre_analysis()
        for genre, stats in genre_analysis.items():
            if stats['total_errors'] > len(self.errors) * 0.3:
                recommendations.append(
                    f"High error rate in {genre} genre: Consider adding more training data "
                    f"or genre-specific fine-tuning."
                )
        
        if not recommendations:
            recommendations.append(
                "Error patterns are diverse. Consider ensemble methods for improvement."
            )
        
        return recommendations
    
    def export_report(self, output_path: str):
        """
        Export detailed error analysis report.
        
        Args:
            output_path: Path to save report
        """
        report = {
            'summary': self.get_summary(),
            'top_patterns': [
                {'type': et.value, 'count': count, 'percentage': pct}
                for et, count, pct in self.get_top_error_patterns()
            ],
            'genre_analysis': self.get_genre_analysis(),
            'recommendations': self.get_recommendations(),
            'sample_errors': [
                {
                    'id': e.sample_id,
                    'quote': e.quote[:100],
                    'predicted': e.predicted_speaker,
                    'actual': e.actual_speaker,
                    'error_type': e.error_type.value,
                    'confidence': e.confidence,
                    'genre': e.genre
                }
                for e in self.errors[:50]  # Sample of errors
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Error analysis report saved to {output_path}")
    
    def print_summary(self):
        """Print formatted summary to console."""
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal errors analyzed: {len(self.errors)}")
        
        print("\nError Type Distribution:")
        for et, count, pct in self.get_top_error_patterns():
            print(f"  {et.value}: {count} ({pct:.1%})")
        
        print("\nPer-Genre Analysis:")
        for genre, stats in self.get_genre_analysis().items():
            print(f"  {genre}: {stats['total_errors']} errors")
        
        print("\nRecommendations:")
        for i, rec in enumerate(self.get_recommendations(), 1):
            print(f"  {i}. {rec}")
        
        print("=" * 60)


def analyze_predictions(
    predictions: List[Dict],
    labels: List[Dict],
    output_path: Optional[str] = None
) -> ErrorAnalyzer:
    """
    Analyze model predictions against ground truth.
    
    Args:
        predictions: List of prediction dictionaries
        labels: List of ground truth dictionaries
        output_path: Optional path to save report
        
    Returns:
        ErrorAnalyzer instance with analysis results
    """
    analyzer = ErrorAnalyzer()
    
    for pred, label in zip(predictions, labels):
        if pred.get('speaker') != label.get('speaker'):
            analyzer.add_error(
                sample_id=label.get('id', 'unknown'),
                text=label.get('text', ''),
                quote=label.get('quote', ''),
                predicted=pred.get('speaker', ''),
                actual=label.get('speaker', ''),
                candidates=label.get('candidates', []),
                confidence=pred.get('confidence', 0.5),
                genre=label.get('genre', 'unknown')
            )
    
    analyzer.print_summary()
    
    if output_path:
        analyzer.export_report(output_path)
    
    return analyzer


# CURSOR: Export public API
__all__ = [
    'ErrorType',
    'ErrorCase',
    'ErrorAnalyzer',
    'analyze_predictions'
]



