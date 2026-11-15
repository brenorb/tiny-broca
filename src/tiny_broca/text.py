"""Brazilian Portuguese front end: text -> letter tokens with a stress feature.

The token alphabet is the Portuguese written form itself: lowercase letters
(including the accented ones that are distinct graphemes), punctuation, and
the usual special tokens. Two decisions, documented against the English
original:

* **Characters, not phonemes.** The original front end mapped text to eSpeak
  phonemes through the ``dizione`` package (CMUdict + G2P + rules). No
  permissive Portuguese equivalent exists: eSpeak's Portuguese voice falls
  back to English phonemes for most words, and no copyleft pronouncing
  dictionary is available. Portuguese graphemes map to phones through a small,
  regular set of context rules, so a character-level front end whose encoder
  learns the context is the standard, fully permissive design.
* **The number reader lives in the normaliser.** Digits, dates, money and
  identifiers are spelled in Brazilian conventions (dot thousands separator,
  comma decimal) before tokenisation, so "R$ 1.234,56" and "15/11/2026" reach
  the model as ordinary words. This is the feature the model is known for
  (reading numbers, dates and money from context) and it is language-specific
  either way.

The forced aligner (``tiny_broca.align``) is a Portuguese character-level CTC
recogniser, so every written letter is a label it can emit; punctuation is
pause-capable and is never forced.
"""

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass

FRONTEND_VERSION = 1
NORMALIZER_VERSION = 1
SEPARATOR = "_"
FRONTENDS = ("pt",)
FRONTEND_VARIABLE = "TINY_BROCA_FRONTEND"
_frontend = os.environ.get(FRONTEND_VARIABLE, "pt")
if _frontend not in FRONTENDS:
    raise ValueError(f"{FRONTEND_VARIABLE} must be one of {FRONTENDS}, not {_frontend!r}")

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…", "-")

LETTERS = (
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "à", "á", "â", "ã", "ç", "é", "ê", "í", "ó", "ô", "õ", "ú", "ü", "ũ",
)
VOWELS = ("a", "e", "i", "o", "u", "à", "á", "â", "ã", "é", "ê", "í", "ó", "ô", "õ", "ú", "ü", "ũ")
# Kept names of the English front end: the set of tokens the aligner can emit.
PHONES = LETTERS
PHONE_SET = frozenset(PHONES)
VOWEL_SET = frozenset(VOWELS)
# Unstressed consonants whose frames should not be voiced (quality checks).
VOICELESS = frozenset(("p", "t", "k", "f", "s", "x"))
# Append-only: changing the order or removing a symbol invalidates checkpoints.
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
LETTER_SET = frozenset(LETTERS)
STRESS_STRESSED = 1

_QUOTE_TRANSLATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "`": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u00ab": '"', "\u00bb": '"',
    "\u2013": "\u2014", "\u2015": "\u2014",
})
_WORD_CHARS = LETTER_SET | {"'"}   # apostrophes stay inside words (d'água)


def frontend():
    """Name of the program that produces the tokens: ``"pt"``."""
    return _frontend


def set_frontend(name):
    """Select the front end for this process (only ``"pt"`` exists)."""
    global _frontend
    if name not in FRONTENDS:
        raise ValueError(f"{FRONTEND_VARIABLE} must be one of {FRONTENDS}, not {name!r}")
    _frontend = name


def frontend_identity():
    return {"name": "pt", "version": FRONTEND_VERSION,
            "normalizer_version": NORMALIZER_VERSION, "symbols": list(SYMBOLS)}


def frontend_runtime():
    """What an annotation run records about the exact program behind its
    tokens. The normaliser is part of this package, so the identity covers it;
    the runtime records nothing extra."""
    return {"normalizer_version": NORMALIZER_VERSION}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Numbers in words (Brazilian conventions)
# ---------------------------------------------------------------------------

_UNITS = ("zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito",
          "nove", "dez", "onze", "doze", "treze", "catorze", "quinze", "dezesseis",
          "dezessete", "dezoito", "dezenove")
_TENS = ("", "dez", "vinte", "trinta", "quarenta", "cinquenta", "sessenta",
         "setenta", "oitenta", "noventa")
_HUNDREDS = ("", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos",
             "seiscentos", "setecentos", "oitocentos", "novecentos")
_ORDINALS = ("primeiro", "segundo", "terceiro", "quarto", "quinto", "sexto",
             "sétimo", "oitavo", "nono", "décimo", "undécimo", "duodécimo", "trezeno")
_MONTHS = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
           "agosto", "setembro", "outubro", "novembro", "dezembro")
_SCALE = ((10 ** 9, "bilhão", "bilhões"), (10 ** 6, "milhão", "milhões"),
          (10 ** 3, "mil", "mil"))


def _below_thousand(n):
    parts = []
    if n >= 100:
        hundreds, rest = divmod(n, 100)
        parts.append("cem" if n == 100 else _HUNDREDS[hundreds])
        if rest:
            parts.append("e")
    if n % 100:
        tens, unit = divmod(n % 100, 10)
        if tens == 1:
            parts.append(_UNITS[10 + unit])     # dez, onze, ..., dezenove
        elif tens:
            parts.append(_TENS[tens])
            if unit:
                parts.append("e")
                parts.append(_UNITS[unit])
        elif unit:
            parts.append(_UNITS[unit])
    return " ".join(p for p in parts if p)


def number_words(n):
    """Spell an integer 0..10**12 in Brazilian Portuguese (masculine form)."""
    n = int(abs(n))
    if n == 0:
        return "zero"
    if n < 1000:
        return _below_thousand(n)
    if n % 1000 == 0 and n < 10 ** 6:
        head = _below_thousand(n // 1000)
        return "mil" if head == "um" else head + " mil"
    parts = []
    for scale, singular, plural in _SCALE:
        if n >= scale:
            count, n = divmod(n, scale)
            # Portuguese says "mil", never "um mil".
            head = "" if (count == 1 and scale == 10 ** 3) else (
                _UNITS[count] if count == 1 else number_words(count))
            parts.append((head + " " if head else "") +
                         (singular if count == 1 else plural))
            # "e" joins the scales when the rest has no tens part:
            # "quatro mil e noventa", "mil e quinhentos", "mil e cinco", but
            # "mil cento e cinquenta". The larger scales take "e" for any rest
            # ("um milhão e quinhentos mil").
            if n and (scale > 10 ** 3 or n < 100 or n % 100 == 0):
                parts.append("e")
    if n:
        parts.append(_below_thousand(n) if n < 1000 else number_words(n))
    return " ".join(parts)


def ordinal_words(n):
    if n < len(_ORDINALS):
        return _ORDINALS[n - 1]
    decade, unit = divmod(n, 10)
    if decade < 20:
        base = number_words(decade).rstrip("s") if decade > 1 else ""
        return (base + " " if base else "") + number_words(unit).replace("um", "um") + "ésimo" if unit else _ORDINALS[9] if decade == 1 else number_words(n) + "ésimo"
    return number_words(n) + "ésimo"


def digits_words(text):
    return " ".join(_UNITS[int(c)] if int(c) < 10 else "dez" for c in text if c.isdigit())


# ---------------------------------------------------------------------------
# Text normalisation: digits and symbols -> words
# ---------------------------------------------------------------------------

_MONEY = re.compile(r"\b[rR]\$\s?(\d{1,3}(?:\.\d{3})*|\d+)(?!\d)(?:,(\d{1,2}))?")
_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\b")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_CLOCK = re.compile(r"\b(\d{1,2})h(\d{2})\b")
_PERCENT = re.compile(r"\b(\d+(?:[.,]\d+)?)\s?%")
_DEGREE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s?°")
_PHONE = re.compile(r"\(\d{2}\)\s?\d{5}-\d{4}")
_GROUPS = re.compile(r"\b\d{1,3}(?:[.-]\d{1,3}){2,}\b")     # CPF, CNPJ-like, CEP
_DECIMAL = re.compile(r"\b(\d+)[,](\d{1,2})\b")
_INTEGER = re.compile(r"\b(\d{1,3}(?:\.\d{3})+|\d{1,5})\b")
_YEAR = re.compile(r"\b(1[0-9]{3}|20\d{2})\b")


def _money(m):
    integer = m.group(1).replace(".", "")
    cents = m.group(2)
    words = number_words(int(integer)) + " real" if int(integer) == 1 else number_words(int(integer)) + " reais"
    if cents:
        words += " e " + number_words(int(cents)) + (" centavo" if int(cents) == 1 else " centavos")
    return words


def _date(m):
    day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
    year = "20" + year if len(year) == 2 else year
    if 1 <= month <= 12:
        return f"{number_words(day)} de {_MONTHS[month - 1]} de {number_words(int(year))}"
    return m.group(0)  # not a date: leave the digits (the integer rule reads them)


def _time(m):
    hour, minute = int(m.group(1)), int(m.group(2))
    if not 0 <= hour <= 23 or minute > 59:
        return m.group(0)
    if hour == 12 and minute == 0:
        return "meio dia"
    if hour == 0:
        return "meia noite" if minute == 0 else f"zero hora e {number_words(minute)}"
    if minute == 0:
        return f"{number_words(hour)}{' hora' if hour == 1 else ' horas'}"
    if minute == 30:
        return f"{number_words(hour)} e meia"
    return f"{number_words(hour)} horas e {number_words(minute)}"


def _clock(m):
    hour, minute = int(m.group(1)), int(m.group(2))
    if not 0 <= hour <= 23 or minute > 59:
        return m.group(0)
    if minute == 0:
        return f"{number_words(hour)}{' hora' if hour == 1 else ' horas'}"
    if minute == 30:
        return f"{number_words(hour)} e meia"
    return f"{number_words(hour)} e {number_words(minute)}"


def _decimal(m):
    return f"{number_words(int(m.group(1)))} vírgula {digits_words(m.group(2))}"


def _groups(m):
    return ", ".join(digits_words(g) for g in re.split(r"[.-]", m.group(0)))


def normalize_text(text):
    """NFKC + casefold + quote/dash unification + spelling out digits.

    Digits follow Brazilian conventions: "1.234" is a thousand and "3,14" a
    decimal; money uses reais and centavos; dates read day-month-year; times
    read in hours and minutes; document identifiers (three-digit groups) read
    digit by digit.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_TRANSLATION)
    text = text.replace("ª", " a").replace("º", " o").replace("×", " vezes ")
    # Two hyphens and three dots are the ASCII forms of the inventory symbols;
    # without this "..." would become three pause tokens.
    text = text.replace("--", "—").replace("...", "…")
    text = text.lower()
    text = _MONEY.sub(_money, text)
    text = _DATE.sub(_date, text)
    text = _TIME.sub(_time, text)
    text = _CLOCK.sub(_clock, text)
    text = _PERCENT.sub(lambda m: number_words(int(re.sub(r"[^0-9]", "", m.group(1)))) + " por cento", text)
    text = _DEGREE.sub(lambda m: number_words(int(re.sub(r"[^0-9]", "", m.group(1)))) + " graus", text)
    text = _PHONE.sub(lambda m: " ".join(digits_words(g) for g in re.findall(r"\d+", m.group(0))), text)
    text = _GROUPS.sub(_groups, text)
    text = _DECIMAL.sub(_decimal, text)
    text = _YEAR.sub(lambda m: number_words(int(m.group(1))), text)
    text = _INTEGER.sub(lambda m: number_words(int(m.group(1).replace(".", ""))), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_ACCENTED = frozenset(("á", "â", "é", "ê", "í", "ó", "ô", "õ", "ú", "ã", "õ"))


def _syllable_starts(word):
    """Start indices of the syllables of a word (one per vowel nucleus).

    A consonant run between two vowels splits so the consonants go with the
    following syllable, the Brazilian convention for single consonants; this
    is an approximation, which is fine for a prosody feature.
    """
    vowel = lambda c: c in VOWELS
    starts = [0]
    i, n = 0, len(word)
    while i < n:
        if vowel(word[i]) and i + 1 < n and not vowel(word[i + 1]):
            j = i + 1
            while j < n and not vowel(word[j]):
                j += 1
            if j < n and vowel(word[j]):      # a consonant run between two vowels
                run = j - i - 1
                starts.append(i + 1 if run == 1 else min(i + 2, j))
        i += 1
    return starts


def _stressed_range(word):
    """(start, end) of the stressed syllable in a word.

    An explicit accent (á â é ê í ó ô õ ú ã) wins; otherwise a word ending in
    a vowel is stressed on its penultimate syllable, a word ending in L, N, R
    or X on its last syllable, other endings on the penultimate.
    """
    for i, c in enumerate(word):
        if c in _ACCENTED:
            starts = _syllable_starts(word)
            start = max((s for s in starts if s <= i), default=0)
            index = starts.index(start)
            end = starts[index + 1] if index + 1 < len(starts) else len(word)
            return start, end
    starts = _syllable_starts(word)
    if word[-1] in ("l", "n", "r", "x"):
        start = starts[-1]
    else:
        start = starts[-2] if len(starts) >= 2 else starts[0]
    index = starts.index(start)
    end = starts[index + 1] if index + 1 < len(starts) else len(word)
    return start, end


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple

    def __len__(self):
        return len(self.ids)


def tokenize(text):
    """Build the token sequence from normalised text.

    Returns a :class:`Phonemized` whose symbols are letters, punctuation and
    word spaces, with a stress feature per token (1 on the letters of the
    stressed syllable of each word, 0 elsewhere) and the list of characters
    that fell outside the inventory.
    """
    symbols = [BOS]
    stress = [0]
    unknown = []
    word = []

    def flush_word():
        if not word:
            return
        text_ = "".join(word)
        start, end = _stressed_range(text_)
        for i, c in enumerate(word):
            symbols.append(c)
            stress.append(STRESS_STRESSED if start <= i < end else 0)
        word.clear()

    for c in text:
        if c == " ":
            flush_word()
            if symbols[-1] != SP:
                symbols.append(SP)
                stress.append(0)
        elif c in LETTER_SET or c in _WORD_CHARS:
            word.append(c)
        elif c in SYMBOL_TO_ID:
            flush_word()
            symbols.append(c)
            stress.append(0)
        else:
            flush_word()
            symbols.append(UNK)
            stress.append(0)
            unknown.append(c)
    flush_word()
    symbols.append(EOS)
    stress.append(0)
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols),
                       tuple(stress), tuple(unknown))


def phonemize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty text")
    return tokenize(normalize_text(text))


def is_phone(token_ids):
    """Boolean list/tensor helper: tokens that must be realised as audio."""
    return [int(i) in PHONE_IDS for i in token_ids]
