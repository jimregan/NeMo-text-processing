# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import csv
import os


def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file

    Returns absolute path
    """
    return os.path.dirname(os.path.abspath(__file__)) + '/' + rel_path


def load_labels(abs_path):
    """
    loads relative path file as dictionary

    Args:
        abs_path: absolute path

    Returns dictionary of mappings
    """
    with open(abs_path, encoding="utf-8") as label_tsv:
        labels = list(csv.reader(label_tsv, delimiter="\t"))
        return labels


def adjective_inflection(word: str, compound: str = "") -> dict:
    """
    inflect adjectives based on their endings.
    This includes things like ordinals.
    """

    def fill_bare_template(stem, mi_sg, mp_pl, vowel, stem_b="", compound=""):
        if stem_b == "":
            stem_b = stem
        if compound == "":
            compound = stem_b + "o"

        # Soft stems have -ich; hard stems have -ëch.
        # Makùrôt: soft stems end in ń, cz, dż, sz, ż.
        pl_vowel = "i" if stem.endswith(("ń", "cz", "dż", "sz", "ż")) else "ë"

        return {
            "mi_sg_nom": mi_sg,
            "mi_sg_gen": stem + "égò",
            "mi_sg_dat": stem + "émù",
            "mi_sg_ins": stem + vowel + "m",

            "nt_sg_nom": stem + "é",

            "f_sg_nom": stem_b + "ô",
            "f_sg_gen": stem + vowel,
            "f_sg_ins": stem_b + "ą",

            "mp_pl_nom": mp_pl,

            "pl_nom": stem + "é",
            "pl_ins": stem + vowel + "ma",
            "pl_loc": stem + pl_vowel + "ch",

            "compound": compound,
        }

    stem_b = ""

    if word.endswith("czi"):
        # e.g. kaszëbsczi -> kaszëbskô
        #      dzyrsczi   -> dzyrskô
        stem = word[:-1]
        stem_b = word[:-3] + "k"
        mi_sg = word
        mp_pl = word
        vowel = "i"

    elif word.endswith("dżi"):
        # e.g. drëdżi -> drëgô
        #      wiôldżi -> wiôlgô
        stem = word[:-1]
        stem_b = word[:-3] + "g"
        mi_sg = word
        mp_pl = word
        vowel = "i"

    elif word.endswith("i"):
        stem = word[:-1]
        mi_sg = word
        mp_pl = word
        vowel = "i"

    elif word.endswith("y"):
        stem = word[:-1]
        mi_sg = word
        mp_pl = word
        vowel = "y"

    else:
        raise ValueError(f"Unrecognised adjective ending: {word}")

    forms = fill_bare_template(
        stem, mi_sg, mp_pl, vowel, stem_b, compound
    )

    return forms
