"""Q2.2 -- Shared, language-aware tokenizer for BM25.

MIND is English, EB-NeRD is Danish. A naive ASCII-stripping tokenizer mangles
Danish `æ ø å` and silently wrecks Danish retrieval quality
(see SPEC.md Q2.2) -- so tokenization is Unicode-aware (NFC-normalised,
`\\w` matched with re.UNICODE) and picks stopwords/stemmer by language.

NLTK's stopword corpora require a one-time download; ada may or may not have
internet access when this first runs, so we try the download once and fall
back to a small hardcoded stopword list per language if it's unavailable,
rather than crashing or (worse) silently retrieving with zero stopword
removal.
"""
from __future__ import annotations

import functools
import re
import unicodedata

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_FALLBACK_STOPWORDS = {
    "english": {
        "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on",
        "for", "with", "as", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "at", "by", "from", "it", "its", "he",
        "she", "they", "we", "you", "i", "not", "no", "do", "does", "did", "have",
        "has", "had", "will", "would", "can", "could", "should", "may", "might",
        "must", "shall", "about", "into", "over", "after", "before", "between",
        "out", "up", "down", "so", "than", "too", "very", "just", "also", "his",
        "her", "their", "our", "your", "my", "him", "them",
    },
    "danish": {
        "og", "i", "jeg", "det", "at", "en", "den", "til", "er", "som", "på",
        "de", "med", "han", "af", "for", "ikke", "der", "var", "mig", "sig", "men",
        "et", "har", "om", "vi", "min", "havde", "ham", "hun", "nu", "over", "da",
        "fra", "du", "ud", "sin", "dem", "os", "op", "man", "hans", "hvor",
        "eller", "hvad", "skal", "selv", "her", "alle", "vil", "blev", "kunne",
        "ind", "når", "være", "dog", "noget", "ville", "jo", "deres",
        "efter", "ned", "skulle", "denne", "end", "dette", "mit", "også",
        "under", "have", "dig", "anden", "hende", "mine", "alt", "meget", "sit",
        "sine", "vor", "mod", "disse", "hvis", "din", "nogle", "hos", "blive",
        "mange", "ad", "bliver", "hendes", "været", "thi", "jer",
    },
}


@functools.lru_cache(maxsize=None)
def get_stopwords(lang: str) -> frozenset[str]:
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            words = stopwords.words(lang)
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = stopwords.words(lang)
        return frozenset(words)
    except Exception:
        return frozenset(_FALLBACK_STOPWORDS[lang])


@functools.lru_cache(maxsize=None)
def get_stemmer(lang: str):
    from nltk.stem.snowball import SnowballStemmer

    return SnowballStemmer(lang)


def tokenize(text: str, lang: str = "english", stem: bool = False, remove_stopwords: bool = True) -> list[str]:
    """lang: 'english' or 'danish' (nltk's SnowballStemmer/stopwords language names)."""
    if not text:
        return []
    text = unicodedata.normalize("NFC", text).lower()
    tokens = _TOKEN_RE.findall(text)
    if remove_stopwords:
        stop = get_stopwords(lang)
        tokens = [t for t in tokens if t not in stop]
    if stem:
        stemmer = get_stemmer(lang)
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens


DATASET_LANG = {"mind": "english", "ebnerd": "danish"}
