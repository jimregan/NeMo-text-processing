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
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.csb.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.csb.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.csb.taggers.time import TimeFst
from nemo_text_processing.text_normalization.csb.verbalizers.time import TimeFst as VerbalizeTimeFst

from ..utils import get_test_cases_multiple


class TestTime:
    tagger = TimeFst(CardinalFst(deterministic=False), OrdinalFst(deterministic=False), deterministic=False)
    verbalizer = VerbalizeTimeFst(deterministic=False)
    graph = pynini.compose(tagger.fst, verbalizer.fst).optimize()

    @parameterized.expand(get_test_cases_multiple("csb/data_text_normalization/test_cases_normalize_with_audio.txt"))
    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_audio_normalization(self, test_input, expected):
        predictions = rewrite.top_rewrites(test_input, self.graph, 100)
        for option in expected:
            assert option in predictions
