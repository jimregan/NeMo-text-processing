# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2022, Jim O'Regan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pynini
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_LOWER,
    NEMO_SIGMA,
    NEMO_UPPER,
    TO_LOWER,
    delete_space,
)
from nemo_text_processing.text_normalization.ga.utils import get_abs_path, load_labels
from pynini.lib import pynutil

_UPPER_ECLIPSIS_LETTERS = pynini.union(
    pynini.cross("B", "mB"),
    pynini.cross("C", "gC"),
    pynini.cross("D", "nD"),
    pynini.cross("F", "bhF"),
    pynini.cross("G", "nG"),
    pynini.cross("P", "bP"),
    pynini.cross("T", "dT"),
    pynini.cross("A", "nA"),
    pynini.cross("E", "nE"),
    pynini.cross("I", "nI"),
    pynini.cross("O", "nO"),
    pynini.cross("U", "nU"),
    pynini.cross("Á", "nÁ"),
    pynini.cross("É", "nÉ"),
    pynini.cross("Í", "nÍ"),
    pynini.cross("Ó", "nÓ"),
    pynini.cross("Ú", "nÚ"),
)
UPPER_ECLIPSIS = pynini.cdrewrite(_UPPER_ECLIPSIS_LETTERS, "[BOS]", "", NEMO_SIGMA)

_LOWER_ECLIPSIS_LETTERS = pynini.union(
    pynini.cross("b", "mb"),
    pynini.cross("c", "gc"),
    pynini.cross("d", "nd"),
    pynini.cross("f", "bhf"),
    pynini.cross("g", "ng"),
    pynini.cross("p", "bp"),
    pynini.cross("t", "dt"),
    pynini.cross("a", "n-a"),
    pynini.cross("e", "n-e"),
    pynini.cross("i", "n-i"),
    pynini.cross("o", "n-o"),
    pynini.cross("u", "n-u"),
    pynini.cross("á", "n-á"),
    pynini.cross("é", "n-é"),
    pynini.cross("í", "n-í"),
    pynini.cross("ó", "n-ó"),
    pynini.cross("ú", "n-ú"),
)
LOWER_ECLIPSIS = pynini.cdrewrite(_LOWER_ECLIPSIS_LETTERS, "[BOS]", "", NEMO_SIGMA)

ECLIPSIS = pynini.union(UPPER_ECLIPSIS, LOWER_ECLIPSIS)

_S_FIXES = pynini.union(
    pynini.cross("shc", "sc"),
    pynini.cross("shf", "sf"),
    pynini.cross("shm", "sm"),
    pynini.cross("shp", "sp"),
    pynini.cross("sht", "st"),
)
S_FIXES = pynini.cdrewrite(_S_FIXES, "[BOS]", "", NEMO_SIGMA)
_LOWER_LENITION_LETTERS = pynini.union(
    pynini.cross("b", "bh"),
    pynini.cross("c", "ch"),
    pynini.cross("d", "dh"),
    pynini.cross("f", "fh"),
    pynini.cross("g", "gh"),
    pynini.cross("m", "mh"),
    pynini.cross("p", "ph"),
    pynini.cross("s", "sh"),
    pynini.cross("t", "th"),
)
_LOWER_LENITION = pynini.cdrewrite(_LOWER_LENITION_LETTERS, "[BOS]", "", NEMO_SIGMA)
LOWER_LENITION = _LOWER_LENITION @ S_FIXES

_LOWER_LENITION_NO_F_NO_S = pynini.union(
    pynini.cross("b", "bh"),
    pynini.cross("c", "ch"),
    pynini.cross("d", "dh"),
    pynini.cross("g", "gh"),
    pynini.cross("m", "mh"),
    pynini.cross("p", "ph"),
    pynini.cross("t", "th"),
)
LOWER_LENITION_NO_F_NO_S = pynini.cdrewrite(_LOWER_LENITION_NO_F_NO_S, "[BOS]", "", NEMO_SIGMA)

UPPER_VOWELS = pynini.union("A", "E", "I", "O", "U", "Á", "É", "Í", "Ó", "Ú").optimize()
LOWER_VOWELS = pynini.union("a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú").optimize()
EITHER_VOWELS = UPPER_VOWELS | LOWER_VOWELS
_UPPER_PONC = pynini.union("Ḃ", "Ċ", "Ḋ", "Ḟ", "Ġ", "Ṁ", "Ṗ", "Ṡ", "Ṫ").optimize()
_LOWER_PONC = pynini.union("ḃ", "ċ", "ḋ", "ḟ", "ġ", "ṁ", "ṗ", "ṡ", "ṫ").optimize()
UPPER_BASE = pynini.union(NEMO_UPPER, UPPER_VOWELS).optimize()
LOWER_BASE = pynini.union(NEMO_LOWER, LOWER_VOWELS).optimize()
UPPER_ALL = pynini.union(UPPER_BASE, _UPPER_PONC).optimize()
LOWER_ALL = pynini.union(LOWER_BASE, _LOWER_PONC).optimize()
UPPER_NO_H = (UPPER_BASE - "H").optimize()
LOWER_NO_H = (LOWER_BASE - "h").optimize()

_FADA_LOWER = pynini.union(*[pynini.cross(x, y) for x, y in zip(["Á", "É", "Í", "Ó", "Ú"], ["á", "é", "í", "ó", "ú"])])
_PONC_LOWER = pynini.union(
    *[
        pynini.cross(x, y)
        for x, y in zip(["Ḃ", "Ċ", "Ḋ", "Ḟ", "Ġ", "Ṁ", "Ṗ", "Ṡ", "Ṫ"], ["ḃ", "ċ", "ḋ", "ḟ", "ġ", "ṁ", "ṗ", "ṡ", "ṫ"])
    ]
)

GA_LOWER = pynini.union(TO_LOWER, _FADA_LOWER, _PONC_LOWER)
GA_ALPHA = pynini.union(UPPER_ALL, LOWER_ALL)

CHAR_NO_H = pynini.union(UPPER_NO_H, LOWER_NO_H).optimize()

_LOWERCASE_STARTS = pynini.union(
    pynini.cross("nA", "n-a"),
    pynini.cross("nE", "n-e"),
    pynini.cross("nI", "n-i"),
    pynini.cross("nO", "n-o"),
    pynini.cross("nU", "n-u"),
    pynini.cross("nÁ", "n-á"),
    pynini.cross("nÉ", "n-é"),
    pynini.cross("nÍ", "n-í"),
    pynini.cross("nÓ", "n-ó"),
    pynini.cross("nÚ", "n-ú"),
    pynini.cross("tA", "t-a"),
    pynini.cross("tE", "t-e"),
    pynini.cross("tI", "t-i"),
    pynini.cross("tO", "t-o"),
    pynini.cross("tU", "t-u"),
    pynini.cross("tÁ", "t-á"),
    pynini.cross("tÉ", "t-é"),
    pynini.cross("tÍ", "t-í"),
    pynini.cross("tÓ", "t-ó"),
    pynini.cross("tÚ", "t-ú"),
)
_DO_LOWER_STARTS = pynini.cdrewrite(_LOWERCASE_STARTS, "[BOS]", "", NEMO_SIGMA)
TOLOWER = (_DO_LOWER_STARTS @ pynini.closure(GA_LOWER | LOWER_BASE | "'" | "-")).optimize()

PREFIX_H = pynini.cdrewrite(pynutil.insert("h"), "[BOS]", EITHER_VOWELS, NEMO_SIGMA)
PREFIX_N = (
    pynini.cdrewrite(pynutil.insert("n"), "[BOS]", UPPER_VOWELS, NEMO_SIGMA)
    @ pynini.cdrewrite(pynutil.insert("n-"), "[BOS]", LOWER_VOWELS, NEMO_SIGMA)
)
PREFIX_T = (
    pynini.cdrewrite(pynutil.insert("t"), "[BOS]", UPPER_VOWELS, NEMO_SIGMA)
    @ pynini.cdrewrite(pynutil.insert("t-"), "[BOS]", LOWER_VOWELS, NEMO_SIGMA)
)

bos_or_space = pynini.union("[BOS]", " ")
eos_or_space = pynini.union("[EOS]", " ")

ensure_space = pynini.cross(pynini.closure(delete_space, 0, 1), " ")


def roman_to_int(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Alters given fst to convert Roman integers (lower and upper cased) into Arabic numerals. Valid for values up to 1000.
    e.g.
        "V" -> "5"
        "i" -> "1"

    Args:
        fst: Any fst. Composes fst onto Roman conversion outputs.
    """

    def _load_roman(file: str):
        roman = load_labels(get_abs_path(file))
        roman_numerals = [(x, y) for x, y in roman] + [(x.upper(), y) for x, y in roman]
        return pynini.string_map(roman_numerals)

    digit = _load_roman("data/roman/digit.tsv")
    ties = _load_roman("data/roman/ties.tsv")
    hundreds = _load_roman("data/roman/hundreds.tsv")
    thousands = _load_roman("data/roman/thousands.tsv")

    graph = (
        digit
        | ties + (digit | pynutil.add_weight(pynutil.insert("0"), 0.01))
        | (
            hundreds
            + (ties | pynutil.add_weight(pynutil.insert("0"), 0.01))
            + (digit | pynutil.add_weight(pynutil.insert("0"), 0.01))
        )
        | (
            thousands
            + (hundreds | pynutil.add_weight(pynutil.insert("0"), 0.01))
            + (ties | pynutil.add_weight(pynutil.insert("0"), 0.01))
            + (digit | pynutil.add_weight(pynutil.insert("0"), 0.01))
        )
    ).optimize()

    return graph @ fst
