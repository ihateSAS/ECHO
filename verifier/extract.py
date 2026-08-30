from __future__ import annotations

import re

_WINDOW = 60

_DEGREE_UNIT = r"(?:degrees?|°)"
_AZIMUTH_ADJACENT_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\s*{_DEGREE_UNIT}\s*(?:to the\s+)?(left|right)", re.IGNORECASE
)
_AZIMUTH_NUMBER_RE = re.compile(rf"(\d+(?:\.\d+)?)[\s-]*{_DEGREE_UNIT}", re.IGNORECASE)
_CENTRE_RE = re.compile(r"\bcent(er|re|red)\b", re.IGNORECASE)


def extract_azimuth(text: str) -> tuple[float, str, tuple[int, int]] | None:
    claims = extract_all_azimuths(text)
    if claims:
        magnitude, side, span = claims[-1]
        return magnitude, side, span

    centre = _CENTRE_RE.search(text)
    if centre is not None:
        return 0.0, "center", centre.span()
    return None


def extract_all_azimuths(text: str) -> list[tuple[float, str, tuple[int, int]]]:
    results: list[tuple[float, str, tuple[int, int]]] = []
    consumed: list[tuple[int, int]] = []

    for match in _AZIMUTH_ADJACENT_RE.finditer(text):
        span = match.span()
        results.append((float(match.group(1)), match.group(2).lower(), span))
        consumed.append(span)

    for match in _AZIMUTH_NUMBER_RE.finditer(text):
        span = match.span()
        if any(span[0] < end and start < span[1] for start, end in consumed):
            continue
        start = max(0, match.start() - _WINDOW)
        end = min(len(text), match.end() + _WINDOW)
        window = text[start:end]
        side_match = re.search(r"\b(left|right)\b", window, re.IGNORECASE)
        if side_match is None:
            continue
        results.append((float(match.group(1)), side_match.group(1).lower(), span))
        consumed.append(span)

    results.sort(key=lambda item: item[2][0])
    return results


_DB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*dB\b", re.IGNORECASE)


def extract_ild_db(text: str) -> tuple[float, tuple[int, int]] | None:
    for match in _DB_RE.finditer(text):
        start = max(0, match.start() - _WINDOW)
        end = min(len(text), match.end() + _WINDOW)
        window = text[start:end]
        if "louder" not in window.lower():
            continue
        side_match = re.search(r"\b(left|right)\b", window, re.IGNORECASE)
        if side_match is None:
            continue
        value = float(match.group(1))
        side = side_match.group(1).lower()
        return (value if side == "left" else -value), match.span()
    return None


_US_RE = re.compile(r"(\d+(?:\.\d+)?)\s*microseconds?\b", re.IGNORECASE)
_NO_DELAY_RE = re.compile(r"\bno\s+(?:inter-?channel\s+)?delay\b", re.IGNORECASE)


def extract_itd_us(text: str) -> tuple[float, tuple[int, int]] | None:
    no_delay = _NO_DELAY_RE.search(text)
    if no_delay is not None:
        return 0.0, no_delay.span()

    match = _US_RE.search(text)
    if match is None:
        return None
    value = float(match.group(1))
    start = max(0, match.start() - _WINDOW)
    end = min(len(text), match.end() + _WINDOW)
    window = text[start:end]
    side_match = re.search(r"\b(left|right)\b\s+channel\s+later", window, re.IGNORECASE)
    if side_match is not None:
        side = side_match.group(1).lower()
        value = value if side == "right" else -value
    return value, match.span()


_COHERENCE_NUMERIC_RE = re.compile(r"coherence\D{0,20}?(\d+(?:\.\d+)?)", re.IGNORECASE)
_COHERENCE_KEYWORDS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\bno coherence\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bnot coherent\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bdecorrelated\b", re.IGNORECASE), 0.0),
    (re.compile(r"\buncorrelated\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bhighly coherent\b", re.IGNORECASE), 1.0),
    (re.compile(r"\bperfectly coherent\b", re.IGNORECASE), 1.0),
    (re.compile(r"\bfully coherent\b", re.IGNORECASE), 1.0),
)


def extract_coherence(text: str) -> tuple[float, tuple[int, int]] | None:
    match = _COHERENCE_NUMERIC_RE.search(text)
    if match is not None:
        return float(match.group(1)), match.span()
    for pattern, value in _COHERENCE_KEYWORDS:
        match = pattern.search(text)
        if match is not None:
            return value, match.span()
    return None


_AMPLITUDE_RE = re.compile(
    r"amplitude[-\s]panned|amplitude panning|amplitude regime", re.IGNORECASE
)
_DELAYED_RE = re.compile(
    r"\bbinaural\b|genuinely delayed|time-delay|time delay|delayed regime", re.IGNORECASE
)


def extract_regime(text: str) -> tuple[str, tuple[int, int]] | None:
    match = _DELAYED_RE.search(text)
    if match is not None:
        return "delayed", match.span()
    match = _AMPLITUDE_RE.search(text)
    if match is not None:
        return "amplitude", match.span()
    return None


_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*Hz\b")


def extract_f0_hz(text: str) -> tuple[float, tuple[int, int]] | None:
    match = _HZ_RE.search(text)
    if match is None:
        return None
    return float(match.group(1)), match.span()


_PREMISE_R_RE = re.compile(
    r"r\s*=\s*10\^\(\s*(-?\d+(?:\.\d+)?)\s*/\s*20\s*\)\s*=\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def extract_premise_r(text: str) -> tuple[float, float, tuple[int, int]] | None:
    match = _PREMISE_R_RE.search(text)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2)), match.span()


_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
_DIGIT_RE = re.compile(r"\d")


def find_unparseable_sentences(
    text: str, consumed_spans: list[tuple[int, int]]
) -> tuple[str, ...]:
    unparseable: list[str] = []
    for match in _SENTENCE_RE.finditer(text):
        start, end = match.span()
        sentence = match.group().strip()
        if not _DIGIT_RE.search(sentence):
            continue
        if any(not (end <= s or start >= e) for s, e in consumed_spans):
            continue
        unparseable.append(sentence)
    return tuple(unparseable)
