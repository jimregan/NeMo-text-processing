# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.csb.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.csb.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.csb.taggers.time import TimeFst
from nemo_text_processing.text_normalization.csb.verbalizers.time import TimeFst as VerbalizeTimeFst


class TestTime:
    tagger = TimeFst(CardinalFst(deterministic=False), OrdinalFst(deterministic=False), deterministic=False)
    verbalizer = VerbalizeTimeFst(deterministic=False)
    graph = pynini.compose(tagger.fst, verbalizer.fst).optimize()

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_minutes_after_hour(self):
        predictions = rewrite.top_rewrites("5:20", self.graph, 100)

        assert "piątô dwadzesce" in predictions
        assert "dwadzesce pò piąti" in predictions

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_minutes_before_half_hour(self):
        predictions = rewrite.top_rewrites("2:25", self.graph, 100)

        assert "drëgô dwadzesce piãc" in predictions
        assert "za piãc pół trzecy" in predictions
