"""Tâi-lô (台羅) character set for Coqui-TTS.

Design notes
------------
Text is stored in NFD (Normalization Form D), so composed letters like
``á`` decompose into base ``a`` + combining acute ``́``. This keeps
the vocabulary small (~30 tokens vs ~100 in NFC) and lets the model
learn letter/tone combinations as independent features.

Frequency-derived from Common Voice nan-tw 25.0 (308k chars across
~24k utterances). Letters q, v, w, x, z don't occur in the corpus
but are kept for OOV loanword robustness.
"""

from TTS.tts.utils.text.characters import BaseCharacters

# Base Latin letters (lowercase). All 26 kept; some are loanword-only.
TAILO_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Combining tone diacritics (Mn category in NFD form):
#   ́  ́  acute             — tone 2  (e.g. á, í)
#   ̀  ̀  grave             — tone 3  (e.g. à, ì)
#   ̂  ̂  circumflex        — tone 5  (e.g. â, î)
#   ̄  ̄  macron            — tone 7  (e.g. ā, ī)
#   ̍  ̍  vertical line     — tone 8  入聲 (e.g. a̍)
#   ̌  ̌  caron             — tone 6  (rare, dialect)
#   ͘  ͘  dot above right   — POJ ``o͘`` legacy
TAILO_TONE_MARKS = "́̀̂̄̍̌͘"

# Standalone Tâi-lô marker:
#   ⁿ  ⁿ  superscript n     — nasalization
TAILO_NASAL = "ⁿ"

# Letters + diacritics + nasal
TAILO_CHARACTERS = TAILO_LETTERS + TAILO_TONE_MARKS + TAILO_NASAL

# Separators and sentence punctuation.
# `-` is a syllable boundary within a word (e.g. ``Tâi-uân``).
# ` ` (space) is a word boundary.
TAILO_PUNCTUATIONS = "-,.!?;: "

# Sentinels reuse the project-wide defaults.
TAILO_PAD = "<PAD>"
TAILO_EOS = "<EOS>"
TAILO_BOS = "<BOS>"
TAILO_BLANK = "<BLNK>"


class TailoCharacters(BaseCharacters):
    """Tâi-lô character vocabulary.

    Order produced by BaseCharacters:
        [PAD, EOS, BOS, BLANK, *letters, *tone_marks, *nasal, *punctuations]

    Typical vocab size with defaults: 4 + 26 + 7 + 1 + 8 = 46
    """

    def __init__(
        self,
        characters: str = TAILO_CHARACTERS,
        punctuations: str = TAILO_PUNCTUATIONS,
        pad: str = TAILO_PAD,
        eos: str = TAILO_EOS,
        bos: str = TAILO_BOS,
        blank: str = TAILO_BLANK,
        is_unique: bool = True,
        is_sorted: bool = False,  # preserve our intentional letter→tone order
    ) -> None:
        super().__init__(
            characters, punctuations, pad, eos, bos, blank, is_unique, is_sorted
        )

    @staticmethod
    def init_from_config(config):
        if config.characters is not None:
            return (
                TailoCharacters(
                    characters=config.characters.get("characters", TAILO_CHARACTERS),
                    punctuations=config.characters.get("punctuations", TAILO_PUNCTUATIONS),
                    pad=config.characters.get("pad", TAILO_PAD),
                    eos=config.characters.get("eos", TAILO_EOS),
                    bos=config.characters.get("bos", TAILO_BOS),
                    blank=config.characters.get("blank", TAILO_BLANK),
                ),
                config,
            )
        return TailoCharacters(), config
