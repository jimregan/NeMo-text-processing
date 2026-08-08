# Copyright (c) 2026, Jim O'Regan
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

from parameterized import parameterized
from pynini.lib.rewrite import one_top_rewrite

from nemo_text_processing.text_normalization.ga.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ga.taggers.ordinal import OrdinalFst
from tests.nemo_text_processing.utils import parse_test_case_file


class TestOrdinal:
    ordinal = OrdinalFst(cardinal=CardinalFst())

    @parameterized.expand(parse_test_case_file('ga/data_text_normalization/test_cases_ordinal.txt'))
    def test_ordinal(self, test_input, expected):
        assert one_top_rewrite(test_input, self.ordinal.fst) == f'ordinal {{ integer: "{expected}" }}'
