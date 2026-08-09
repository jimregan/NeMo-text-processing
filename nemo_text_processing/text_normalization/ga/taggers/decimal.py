# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, 2026, Jim O'Regan
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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimals, e.g.
        "1.7" -> decimal { integer_part: "a haon" fractional_part: "a seacht" }
        "-2.04" -> decimal { negative: "true" integer_part: "a dó" fractional_part: "a náid a ceathair" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        non_zero_integer = ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)) @ cardinal.graph
        integer = pynini.cross("0", "a náid") | non_zero_integer
        fractional_digits = cardinal.read_digits

        self.graph = fractional_digits
        self.graph_integer = pynutil.insert('integer_part: "') + integer + pynutil.insert('"')
        self.graph_fractional = (
            pynutil.insert('fractional_part: "') + fractional_digits + pynutil.insert('"')
        )

        decimal_separator = pynutil.delete(pynini.union(".", ","))
        integer_part = self.graph_integer | pynutil.insert('integer_part: "a náid"')
        self.final_graph_wo_sign = integer_part + decimal_separator + pynutil.insert(" ") + self.graph_fractional
        self.final_graph_wo_negative = self.final_graph_wo_sign
        self.final_graph_wo_negative_w_abbr = self.final_graph_wo_sign

        optional_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1
        )
        self.fst = self.add_tokens(optional_negative + self.final_graph_wo_sign).optimize()
