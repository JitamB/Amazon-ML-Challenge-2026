"""Text normalisation: script transliteration, name/address cleaning, number
extraction and a phonetic consonant skeleton.

Everything here is country-agnostic on purpose: France has no training data,
so no rule may depend on the country label.
"""
import re
import unicodedata
from collections import Counter

import pandas as pd
from indic_transliteration import sanscript
from unidecode import unidecode

# ---------------------------------------------------------------- scripts
_INDIC_BLOCKS = [  # (first code point, last code point, sanscript scheme)
    (0x0900, 0x097F, sanscript.DEVANAGARI),
    (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA),
    (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU),
    (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
]
# Malayalam chillu letters are not handled by sanscript.
_CHILLU = str.maketrans({"ർ": "ര്", "ൻ": "ന്", "ൽ": "ല്", "ൾ": "ള്", "ൺ": "ണ്"})
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_INDIC_RUN = re.compile("[ऀ-ൿ]+")


def _scheme(ch):
    cp = ord(ch)
    for lo, hi, scheme in _INDIC_BLOCKS:
        if lo <= cp <= hi:
            return scheme
    return None


def _translit_run(m):
    run = m.group(0).translate(_CHILLU)
    scheme = _scheme(run[0])
    return transliterate(run, scheme) if scheme else run


def transliterate(text, scheme):
    out = sanscript.transliterate(text, scheme, sanscript.IAST)
    # Malayalam "ṟṟ" / "nṟ" are pronounced "tt" / "nt".
    return out.replace("ṟṟ", "tt").replace("nṟ", "nt") if scheme == sanscript.MALAYALAM else out


def to_latin(s):
    """Any script -> lowercase ASCII. Indic runs are transliterated, the rest unidecoded."""
    s = unicodedata.normalize("NFKC", s)
    s = _ZERO_WIDTH.sub("", s).replace("’", "'")
    if _INDIC_RUN.search(s):
        s = _INDIC_RUN.sub(_translit_run, s)
    return unidecode(s).lower()


def has_indic(s):
    return bool(_INDIC_RUN.search(s))


# ---------------------------------------------------------------- names
LEGAL = set("""
llc l l c inc incorporated corp corporation co company cos ltd limited pvt private pc p c lp llp pllc plc pa
sarl sas sasu sa eurl sci snc selarl eirl ei gie
the and of de du des la le les et
group groupe services service center centre shri sri smt dr mr mrs ms m s dba d b a nee aka a k a
""".split())
_WEB = re.compile(r"https?://|www\.|\.(com|net|org|in|co|fr|biz|info|us|io)\b")
# OCR-style digit->letter swaps, applied only inside tokens that also contain letters.
_OCR = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "8": "b"})


def _fix_ocr(tok):
    if tok.isdigit() or tok.isalpha():
        return tok
    return tok.translate(_OCR)


_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}")  # dotted acronyms: "e.u.r.l." -> "eurl", "l.l.c." -> "llc"


def name_tokens(latin_name):
    s = _DOTTED.sub(lambda m: m.group(0).replace(".", "") + " ", latin_name)
    s = _WEB.sub(" ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [_fix_ocr(t) for t in s.split()]
    return [t for t in toks if t not in LEGAL]


# ---------------------------------------------------------------- phonetics
# Consonant skeleton with voicing merged: robust to transliteration
# ("praivet limited" ~ "private limited") and to vowel typos.
_PH_MAP = str.maketrans({"b": "p", "d": "t", "g": "k", "q": "k", "c": "k", "v": "w",
                         "z": "s", "j": "s", "x": "s", "f": "p"})


def phonetic(tok):
    if not tok:
        return ""
    first = tok[0]
    body = re.sub(r"[aeiouyh]", "", tok[1:].translate(_PH_MAP))
    head = first.translate(_PH_MAP) if first not in "aeiouyh" else first
    return re.sub(r"(.)\1+", r"\1", head + body)


LEGAL_PHON = {phonetic(t) for t in LEGAL | {"private", "limited", "praivet", "limited"}}


# ---------------------------------------------------------------- addresses
ABBR = {
    # US / India
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "ln": "lane", "dr": "drive",
    "ct": "court", "blvd": "boulevard", "bd": "boulevard", "pl": "place", "cir": "circle",
    "hwy": "highway", "pkwy": "parkway", "ter": "terrace", "trl": "trail", "sq": "square",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "nr": "near", "opp": "opposite",
    # France
    "r": "rue", "ch": "chemin", "all": "allee", "imp": "impasse", "rte": "route", "fbg": "faubourg",
    "zi": "zone industrielle", "cc": "centre commercial", "ld": "lieu dit",
}
ADDR_STOP = {"null", "n", "a", "na", "none", "no", "h", "hno", "house", "number", "ndeg"}
# French S1 addresses end in the region, S2/S3 in the region and/or the department
# ("... bordeaux nouvelle aquitaine" vs "... bordeaux gironde"). Both carry no information
# beyond the city, so they are removed (France has no training labels to learn this from).
_FR_AREA = re.compile(r"\b(hauts de france|nouvelle aquitaine|pays de la loire|loire atlantique|pas de calais"
                      r"|gironde)\b|\bnord$")


def addr_tokens(latin_addr):
    s = re.sub(r"[^a-z0-9]+", " ", latin_addr).strip()
    s = _FR_AREA.sub(" ", s)
    out = []
    for t in s.split():
        if t in ADDR_STOP:
            continue
        out.extend(ABBR.get(t, t).split())
    return out


def numbers(latin_addr):
    """Distinct numeric tokens in order of appearance, leading zeros stripped."""
    seen = []
    for n in re.findall(r"\d+", latin_addr):
        n = n.lstrip("0") or "0"
        if n not in seen:
            seen.append(n)
    return seen


# ---------------------------------------------------------------- dataframe
def name_columns(latin_names):
    """(name, name_compact, phon) columns from lowercase-latin names."""
    ntoks = [name_tokens(x) for x in latin_names]
    ph = [[p for p in (phonetic(t) for t in toks) if p and p not in LEGAL_PHON] for toks in ntoks]
    return [" ".join(t) for t in ntoks], ["".join(t) for t in ntoks], [" ".join(p) for p in ph]


def prepare(df, source):
    """Add all normalised columns used by blocking and features."""
    out = pd.DataFrame({"rid": df["entity_id"].values, "country": df["country"].values,
                        "source": source})
    raw_name = df["business_name"].values
    raw_addr = df["business_address"].values
    lat_name = [to_latin(x) for x in raw_name]
    lat_addr = [to_latin(x) for x in raw_addr]
    atoks = [addr_tokens(x) for x in lat_addr]
    nums = [numbers(x) for x in lat_addr]
    out["name"], out["name_compact"], out["phon"] = name_columns(lat_name)
    out["addr"] = [" ".join(t) for t in atoks]
    out["addr_alpha"] = [" ".join(t for t in toks if t.isalpha() and len(t) >= 3) for toks in atoks]
    out["nums"] = [" ".join(n) for n in nums]
    out["hn"] = [n[0] if n else "" for n in nums]
    out["name_raw"] = [x.strip() for x in lat_name]
    out["name_indic"] = [has_indic(x) for x in raw_name]
    out["name_web"] = [bool(re.search(r"\.(com|net|org|in|fr)\b|^[#@]|www\.", x)) for x in lat_name]
    out["addr_empty"] = [len(t) == 0 for t in atoks]
    return out


# ---------------------------------------------------------------- learned transliteration
# Many Indic-script names are word-by-word transliterations of the English name
# ("balaji bisines praivet limited" <- "Balaji Business Private Limited"). From training
# pairs with equal word counts we learn transliterated word -> English word, then rewrite
# Indic-origin names with it (train and test alike; learned from train labels only).
def _raw_tokens(latin):
    return re.sub(r"[^a-z0-9]+", " ", latin).split()


def learn_translit(s1_names, rec_names, min_count=3, min_purity=0.6):
    pairs = Counter()
    for a, b in zip(s1_names, rec_names):
        ta, tb = _raw_tokens(a), _raw_tokens(b)
        if len(ta) == len(tb):
            pairs.update(zip(tb, ta))
    total, best = Counter(), {}
    for (src, dst), c in pairs.items():
        total[src] += c
        if c > best.get(src, ("", 0))[1]:
            best[src] = (dst, c)
    return {src: dst for src, (dst, c) in best.items()
            if c >= min_count and c / total[src] >= min_purity and src != dst}


def apply_translit(frame, mapping):
    """Rewrite name columns of Indic-origin names in place using the learned mapping."""
    rows = frame.index[frame["name_indic"].values]
    if len(rows) == 0 or not mapping:
        return frame
    mapped = [" ".join(mapping.get(t, t) for t in _raw_tokens(x)) for x in frame.loc[rows, "name_raw"]]
    name, compact, phon = name_columns(mapped)
    frame.loc[rows, "name"], frame.loc[rows, "name_compact"], frame.loc[rows, "phon"] = name, compact, phon
    frame.loc[rows, "name_raw"] = mapped
    return frame


# ---------------------------------------------------------------- learned address words
# Record addresses use transliterated or abbreviated forms of S1 words ("pascimavanga",
# "wb" <- "west bengal", "tg" <- "telangana"). For training pairs where exactly one alphabetic
# word on each side has no counterpart on the other, the two are paired; frequent, consistent
# pairs become a record-word -> S1-word map applied to all S2/S3 addresses.
def learn_addr_map(s1_addrs, rec_addrs, min_count=20, min_purity=0.6):
    pairs = Counter()
    for a, b in zip(s1_addrs, rec_addrs):
        sa = {t for t in a.split() if t.isalpha()}
        sb = {t for t in b.split() if t.isalpha()}
        only_a, only_b = sa - sb, sb - sa
        if len(only_a) == 1 and len(only_b) == 1:
            pairs[(next(iter(only_b)), next(iter(only_a)))] += 1
    total, best = Counter(), {}
    for (src, dst), c in pairs.items():
        total[src] += c
        if c > best.get(src, ("", 0))[1]:
            best[src] = (dst, c)
    return {src: dst for src, (dst, c) in best.items() if c >= min_count and c / total[src] >= min_purity}


def apply_addr_map(frame, mapping, s1):
    """Rewrite the addr / addr_alpha columns of S2/S3 records in place.

    The map is learned on India/US only. Per country, a rule is used only if it moves a word
    towards that country's S1 vocabulary (target word more frequent than the source word in
    its S1 addresses); otherwise French "de -> des" and "saint -> street" would break France.
    """
    if not mapping:
        return frame
    for country in frame["country"].unique():
        freq = Counter(t for x in s1["addr"].values[s1["country"].values == country] for t in x.split())
        m = {a: b for a, b in mapping.items() if freq[b] > freq[a]}
        rows = frame["country"].values == country
        for col in ("addr", "addr_alpha"):
            frame.loc[rows, col] = [" ".join(m.get(t, t) for t in x.split()) for x in frame.loc[rows, col].values]
    return frame
