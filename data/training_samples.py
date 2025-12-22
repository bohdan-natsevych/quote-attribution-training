"""
Training-sample preprocessing utilities for quote attribution.

CURSOR: This module centralizes candidate construction and shuffling so training
doesn't accidentally leak labels via candidate ordering.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


_TITLE = r"(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Lady|Lord)\.?"
_NAME = r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?"
_NAME_SEQ_RE = re.compile(rf"\b(?:{_TITLE}\s+)?{_NAME}(?:\s+{_NAME}){{0,2}}\b")

_ATTR_VERBS = ("said", "asked", "replied", "answered", "exclaimed", "whispered", "shouted", "murmured", "muttered")
_ATTR_PATTERNS = [
    re.compile(rf"\b({_NAME}(?:\s+{_NAME}){{0,2}})\s+(?:{'|'.join(_ATTR_VERBS)})\b"),
    re.compile(rf"\b(?:{'|'.join(_ATTR_VERBS)})\s+({_NAME}(?:\s+{_NAME}){{0,2}})\b"),
]

_STOPWORDS = {
    "The", "A", "An", "And", "But", "Or", "If", "Then", "This", "That", "These", "Those",
}


def _stable_int_hash(text: str) -> int:
    """CURSOR: Deterministic small hash for per-sample RNG seeding."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def derive_quote_span(
    text: str,
    quote: Optional[str],
    quote_start: Optional[int] = None,
    quote_end: Optional[int] = None,
) -> Tuple[int, int]:
    """CURSOR: Best-effort quote span discovery in `text`."""
    text = text or ""
    q = (quote or "").strip()
    if quote_start is not None and quote_end is not None and 0 <= quote_start < quote_end <= len(text):
        return int(quote_start), int(quote_end)
    if not q:
        return -1, -1

    pos = text.find(q)
    if pos >= 0:
        return pos, pos + len(q)

    lower_pos = text.lower().find(q.lower())
    if lower_pos >= 0:
        return lower_pos, lower_pos + len(q)

    return -1, -1


def derive_context_windows(
    text: str,
    quote: Optional[str],
    quote_start: Optional[int] = None,
    quote_end: Optional[int] = None,
    window_chars: int = 500,
) -> Tuple[str, str]:
    """CURSOR: Provide (context_before, context_after) for difficulty heuristics."""
    start, end = derive_quote_span(text, quote, quote_start, quote_end)
    if start < 0:
        mid = len(text) // 2
        before = text[max(0, mid - window_chars) : mid]
        after = text[mid : min(len(text), mid + window_chars)]
        return _norm_space(before), _norm_space(after)

    before = text[max(0, start - window_chars) : start]
    after = text[end : min(len(text), end + window_chars)]
    return _norm_space(before), _norm_space(after)


def extract_in_context_candidates(
    text: str,
    quote: Optional[str],
    quote_start: Optional[int] = None,
    quote_end: Optional[int] = None,
    max_candidates: int = 10,
) -> List[str]:
    """CURSOR: Heuristic candidate extraction from the sample context."""
    text = text or ""
    q_start, _ = derive_quote_span(text, quote, quote_start, quote_end)
    if q_start < 0:
        q_start = len(text) // 2

    candidates: Dict[str, int] = {}

    def consider(name: str, pos: int):
        name = _norm_space(name).strip(" ,.;:!?\"'()[]{}")
        if not name or name in _STOPWORDS:
            return
        # Drop obvious non-names.
        if len(name) < 2 or name.isdigit():
            return
        # Prefer closest occurrence to quote position.
        dist = abs(pos - q_start)
        prev = candidates.get(name)
        if prev is None or dist < prev:
            candidates[name] = dist

    for m in _NAME_SEQ_RE.finditer(text):
        consider(m.group(0), m.start())

    for pat in _ATTR_PATTERNS:
        for m in pat.finditer(text):
            consider(m.group(1), m.start(1))

    ranked = sorted(candidates.items(), key=lambda kv: kv[1])
    return [name for name, _ in ranked[: max(0, int(max_candidates))]]


def _add_random_negatives(
    existing: List[str],
    gold: str,
    pool: Sequence[str],
    k: int,
    rng: random.Random,
) -> List[str]:
    if k <= 0:
        return existing
    existing_set = {c.lower(): c for c in existing}
    gold_l = gold.lower()

    pool_unique = []
    for p in pool:
        p = _norm_space(p)
        if not p:
            continue
        pl = p.lower()
        if pl == gold_l:
            continue
        if pl in existing_set:
            continue
        pool_unique.append(p)

    rng.shuffle(pool_unique)
    return existing + pool_unique[:k]


def _shuffle_candidates(
    candidates: List[str],
    gold: str,
    rng: random.Random,
) -> Tuple[List[str], int]:
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    gold_idx = shuffled.index(gold)
    return shuffled, gold_idx


def finalize_candidate_sets(
    samples: List[Dict[str, Any]],
    seed: int,
    hard_negative_topk: int,
    max_candidates: int = 10,
    shuffle_candidates: bool = True,
) -> List[Dict[str, Any]]:
    """
    CURSOR: Ensure each sample has a realistic candidate set and a correct gold_index.

    - PDNC samples (already have candidates + gold_index) are preserved (optionally shuffled).
    - Other datasets get in-context candidates; remaining negatives come from a global speaker pool.
    """
    pool: List[str] = []
    for s in samples:
        gold = _norm_space(s.get("gold") or s.get("speaker") or "")
        if gold:
            pool.append(gold)
        for c in s.get("candidates") or []:
            c = _norm_space(c)
            if c:
                pool.append(c)
    pool = list(dict.fromkeys(pool))  # stable unique

    out: List[Dict[str, Any]] = []
    for s in samples:
        gold = _norm_space(s.get("gold") or s.get("speaker") or "")
        if not gold:
            continue

        qid = s.get("quote_id") or f"{s.get('source','unknown')}:{s.get('book_id','')}:{s.get('idx',0)}"
        rng = random.Random(int(seed) + _stable_int_hash(str(qid)))

        # PDNC-like: keep provided candidates if present.
        candidates = s.get("candidates")
        gold_index = s.get("gold_index")
        if isinstance(candidates, list) and candidates:
            candidates = [_norm_space(c) for c in candidates if _norm_space(c)]
            if gold not in candidates:
                candidates = [gold] + [c for c in candidates if c.lower() != gold.lower()]
            if not isinstance(gold_index, int) or not (0 <= gold_index < len(candidates)):
                gold_index = candidates.index(gold)
        else:
            # Build candidates from context.
            text = s.get("text") or ""
            quote = s.get("quote") or ""
            quote_start = s.get("quote_start")
            quote_end = s.get("quote_end")

            candidates = extract_in_context_candidates(
                text=text,
                quote=quote,
                quote_start=quote_start,
                quote_end=quote_end,
                max_candidates=max_candidates,
            )
            # Ensure gold is present and first before shuffling.
            candidates = [gold] + [c for c in candidates if c.lower() != gold.lower()]
            candidates = _add_random_negatives(
                existing=candidates,
                gold=gold,
                pool=pool,
                k=int(hard_negative_topk),
                rng=rng,
            )
            candidates = candidates[: max(2, int(max_candidates))]
            gold_index = 0

        # If we still don't have at least 2 candidates, skip sample (no supervision signal).
        if len(candidates) < 2:
            continue

        if shuffle_candidates:
            candidates, gold_index = _shuffle_candidates(candidates, gold, rng)

        s2 = dict(s)
        s2["gold"] = gold
        s2["candidates"] = candidates[: int(max_candidates)]
        s2["gold_index"] = int(gold_index)

        # Add derived context windows for curriculum heuristics.
        if "context_before" not in s2 or "context_after" not in s2:
            cb, ca = derive_context_windows(
                text=s2.get("text") or "",
                quote=s2.get("quote") or "",
                quote_start=s2.get("quote_start"),
                quote_end=s2.get("quote_end"),
            )
            s2["context_before"] = cb
            s2["context_after"] = ca

        out.append(s2)

    return out


__all__ = [
    "derive_quote_span",
    "derive_context_windows",
    "extract_in_context_candidates",
    "finalize_candidate_sets",
]


