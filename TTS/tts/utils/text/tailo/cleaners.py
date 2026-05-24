"""Text cleaners for Tâi-lô (台羅) input.

The cleaner output is fed directly into the Tâi-lô character vocabulary, so
its job is to: normalize to NFD, lowercase, strip residual CJK / other
out-of-set symbols, and map common variants (fullwidth punctuation,
exotic hyphens, dotless i) to their canonical form.
"""

import unicodedata

# Single-char remap table applied after NFD + lowercase.
# Maps to "" mean: drop the char.
_REMAP = {
    # Fullwidth → ASCII
    "，": ",",  # ，
    "．": ".",  # ．
    "？": "?",  # ？
    "！": "!",  # ！
    "：": ":",  # ：
    "；": ";",  # ；
    "／": "/",  # ／
    # Exotic hyphens / dashes → ASCII hyphen
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
    # Smart quotes → straight
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    # Turkish dotless i (occasionally appears as NFD of capital İ misread) → i
    "ı": "i",
    # Rare double-acute (suspected typo for vertical line tone 8) → drop
    # so the model treats the carrier letter as toneless rather than learn a
    # 1-in-1000 phantom symbol. Set to "" intentionally.
    "̋": "",
}


def tailo_basic_cleaner(text: str) -> str:
    """Normalize a Tâi-lô string for the TailoCharacters vocabulary.

    Steps:
        1. NFD-decompose so ``á`` → ``a`` + combining acute.
        2. Lowercase (case is non-phonetic in Tâi-lô — capitals only mark
           proper nouns and that's already implicit in pauses).
        3. Apply remap (fullwidth → ASCII, exotic dashes → ``-``, etc.).
        4. Drop any leftover CJK ideographs (residual Han from data leaks).
        5. Collapse runs of whitespace.
    """
    text = unicodedata.normalize("NFD", text).lower()
    out = []
    for ch in text:
        if ch in _REMAP:
            mapped = _REMAP[ch]
            if mapped:
                out.append(mapped)
            continue
        # Drop CJK / hiragana / katakana / hangul leakage.
        if "CJK" in unicodedata.name(ch, "") or unicodedata.category(ch) == "Lo":
            continue
        out.append(ch)
    cleaned = "".join(out)
    # Collapse whitespace.
    cleaned = " ".join(cleaned.split())
    return cleaned
