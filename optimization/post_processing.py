"""
Smart Post-Processing Rules for Quote Attribution

Linguistic rules to fix common errors:
- Dialogue continuity tracking
- Pronoun consistency checking
- Speaker persistence rules
- Confidence-based fallbacks
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re


@dataclass
class AttributionResult:
    """Result of quote attribution with post-processing."""
    speaker: str
    confidence: float
    original_speaker: str
    original_confidence: float
    rule_applied: Optional[str]
    was_modified: bool


class DialogueTracker:
    """
    Track dialogue state for speaker continuity.
    """
    
    def __init__(self, max_history: int = 5):
        """
        Initialize tracker.
        
        Args:
            max_history: Maximum quotes to remember
        """
        self.history: List[Dict] = []
        self.max_history = max_history
        self.current_speakers: List[str] = []
    
    def add_quote(
        self,
        speaker: str,
        quote: str,
        position: int
    ):
        """Add a quote to history."""
        self.history.append({
            'speaker': speaker,
            'quote': quote,
            'position': position
        })
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        if speaker not in self.current_speakers:
            self.current_speakers.append(speaker)
            if len(self.current_speakers) > 4:
                self.current_speakers.pop(0)
    
    def get_last_speaker(self) -> Optional[str]:
        """Get most recent speaker."""
        return self.history[-1]['speaker'] if self.history else None
    
    def get_alternating_speaker(self) -> Optional[str]:
        """Get speaker for alternating dialogue pattern."""
        if len(self.history) < 2:
            return None
        
        last = self.history[-1]['speaker']
        second_last = self.history[-2]['speaker']
        
        if last != second_last:
            return second_last
        
        return None
    
    def is_continuation(self, quote: str, position: int) -> bool:
        """Check if quote is likely a continuation of dialogue."""
        if not self.history:
            return False
        
        last = self.history[-1]
        
        # Close proximity
        if position - last['position'] < 100:
            return True
        
        # Same paragraph indicators
        return False
    
    def reset(self):
        """Reset tracker for new context."""
        self.history.clear()
        self.current_speakers.clear()


class PostProcessor:
    """
    Apply linguistic rules to improve attribution accuracy.
    """
    
    # CURSOR: Dialogue continuation verbs
    CONTINUATION_VERBS = {
        'continued', 'added', 'went on', 'resumed'
    }
    
    # CURSOR: Strong attribution patterns
    ATTRIBUTION_PATTERNS = [
        r'(\w+)\s+said',
        r'said\s+(\w+)',
        r'(\w+)\s+asked',
        r'asked\s+(\w+)',
        r'(\w+)\s+replied',
        r'(\w+)\s+answered',
        r'(\w+)\s+exclaimed',
    ]
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        use_dialogue_tracking: bool = True,
        use_pattern_matching: bool = True
    ):
        """
        Initialize post-processor.
        
        Args:
            confidence_threshold: Below this, apply rules
            use_dialogue_tracking: Enable dialogue tracking
            use_pattern_matching: Enable pattern matching
        """
        self.confidence_threshold = confidence_threshold
        self.use_dialogue_tracking = use_dialogue_tracking
        self.use_pattern_matching = use_pattern_matching
        
        self.dialogue_tracker = DialogueTracker()
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.ATTRIBUTION_PATTERNS
        ]
    
    def process(
        self,
        speaker: str,
        confidence: float,
        quote: str,
        context: str,
        candidates: List[str],
        position: int = 0
    ) -> AttributionResult:
        """
        Apply post-processing rules.
        
        Args:
            speaker: Model's predicted speaker
            confidence: Model's confidence
            quote: The quote text
            context: Surrounding context
            candidates: List of candidate speakers
            position: Quote position in text
            
        Returns:
            AttributionResult with potentially modified speaker
        """
        original_speaker = speaker
        original_confidence = confidence
        rule_applied = None
        
        # CURSOR: Rule 1: Strong attribution pattern override
        if self.use_pattern_matching:
            pattern_speaker = self._check_attribution_patterns(context, candidates)
            if pattern_speaker and pattern_speaker in candidates:
                if confidence < 0.9:  # Only override if not very confident
                    speaker = pattern_speaker
                    confidence = max(confidence, 0.85)
                    rule_applied = "attribution_pattern"
        
        # CURSOR: Rule 2: Dialogue continuation
        if self.use_dialogue_tracking and rule_applied is None:
            if self._is_continuation(quote, context, position):
                last_speaker = self.dialogue_tracker.get_last_speaker()
                if last_speaker and last_speaker in candidates:
                    if confidence < self.confidence_threshold:
                        speaker = last_speaker
                        confidence = max(confidence, 0.7)
                        rule_applied = "dialogue_continuation"
        
        # CURSOR: Rule 3: Alternating dialogue
        if self.use_dialogue_tracking and rule_applied is None:
            if self._is_alternating_dialogue(context):
                alt_speaker = self.dialogue_tracker.get_alternating_speaker()
                if alt_speaker and alt_speaker in candidates:
                    if confidence < self.confidence_threshold:
                        speaker = alt_speaker
                        confidence = max(confidence, 0.65)
                        rule_applied = "alternating_dialogue"
        
        # CURSOR: Rule 4: Proximity fallback
        if rule_applied is None and confidence < 0.5:
            closest = self._find_closest_speaker(context, quote, candidates)
            if closest:
                speaker = closest
                confidence = 0.55
                rule_applied = "proximity_fallback"
        
        # Update dialogue tracker
        self.dialogue_tracker.add_quote(speaker, quote, position)
        
        return AttributionResult(
            speaker=speaker,
            confidence=confidence,
            original_speaker=original_speaker,
            original_confidence=original_confidence,
            rule_applied=rule_applied,
            was_modified=(speaker != original_speaker)
        )
    
    def _check_attribution_patterns(
        self,
        context: str,
        candidates: List[str]
    ) -> Optional[str]:
        """Check for strong attribution patterns."""
        for pattern in self.compiled_patterns:
            matches = pattern.findall(context)
            for match in matches:
                # Check if match is a candidate
                match_lower = match.lower()
                for cand in candidates:
                    if match_lower in cand.lower() or cand.lower() in match_lower:
                        return cand
        return None
    
    def _is_continuation(
        self,
        quote: str,
        context: str,
        position: int
    ) -> bool:
        """Check if quote is dialogue continuation."""
        context_lower = context.lower()
        
        # Check for continuation verbs
        for verb in self.CONTINUATION_VERBS:
            if verb in context_lower:
                return True
        
        # Check tracker
        return self.dialogue_tracker.is_continuation(quote, position)
    
    def _is_alternating_dialogue(self, context: str) -> bool:
        """Check for alternating dialogue pattern."""
        # Count quote marks in context
        quote_count = context.count('"') + context.count('"') + context.count('"')
        
        # Alternating pattern typically has multiple quotes
        return quote_count >= 4
    
    def _find_closest_speaker(
        self,
        context: str,
        quote: str,
        candidates: List[str]
    ) -> Optional[str]:
        """Find candidate closest to quote in context."""
        quote_pos = context.find(quote)
        if quote_pos < 0:
            quote_pos = len(context) // 2
        
        closest = None
        min_distance = float('inf')
        
        for cand in candidates:
            pos = context.find(cand)
            if pos >= 0:
                dist = abs(pos - quote_pos)
                if dist < min_distance:
                    min_distance = dist
                    closest = cand
        
        return closest
    
    def reset_context(self):
        """Reset dialogue tracker for new document."""
        self.dialogue_tracker.reset()
    
    def batch_process(
        self,
        predictions: List[Dict]
    ) -> List[AttributionResult]:
        """
        Process a batch of predictions.
        
        Args:
            predictions: List of prediction dicts with:
                - speaker, confidence, quote, context, candidates, position
                
        Returns:
            List of AttributionResults
        """
        self.reset_context()
        results = []
        
        for pred in predictions:
            result = self.process(
                speaker=pred['speaker'],
                confidence=pred['confidence'],
                quote=pred['quote'],
                context=pred['context'],
                candidates=pred['candidates'],
                position=pred.get('position', 0)
            )
            results.append(result)
        
        return results


def apply_post_processing(
    model_predictions: List[Dict],
    confidence_threshold: float = 0.6
) -> Tuple[List[str], Dict[str, int]]:
    """
    Apply post-processing to model predictions.
    
    Args:
        model_predictions: List of prediction dictionaries
        confidence_threshold: Threshold for applying rules
        
    Returns:
        (corrected_speakers, rule_statistics)
    """
    processor = PostProcessor(confidence_threshold=confidence_threshold)
    
    results = processor.batch_process(model_predictions)
    
    speakers = [r.speaker for r in results]
    
    stats = {
        'total': len(results),
        'modified': sum(1 for r in results if r.was_modified),
        'attribution_pattern': sum(1 for r in results if r.rule_applied == 'attribution_pattern'),
        'dialogue_continuation': sum(1 for r in results if r.rule_applied == 'dialogue_continuation'),
        'alternating_dialogue': sum(1 for r in results if r.rule_applied == 'alternating_dialogue'),
        'proximity_fallback': sum(1 for r in results if r.rule_applied == 'proximity_fallback'),
    }
    
    print(f"\nPost-processing statistics:")
    print(f"  Total predictions: {stats['total']}")
    print(f"  Modified: {stats['modified']} ({stats['modified']/max(stats['total'],1):.1%})")
    for rule in ['attribution_pattern', 'dialogue_continuation', 'alternating_dialogue', 'proximity_fallback']:
        if stats[rule] > 0:
            print(f"  {rule}: {stats[rule]}")
    
    return speakers, stats


# CURSOR: Export public API
__all__ = [
    'AttributionResult',
    'DialogueTracker',
    'PostProcessor',
    'apply_post_processing'
]



