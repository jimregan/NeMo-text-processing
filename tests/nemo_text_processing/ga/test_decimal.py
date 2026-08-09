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

from pynini.lib.rewrite import one_top_rewrite, top_rewrites

from nemo_text_processing.text_normalization.ga.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ga.taggers.decimal import DecimalFst as DecimalTagger
from nemo_text_processing.text_normalization.ga.verbalizers.decimals import DecimalFst as DecimalVerbalizer


def _tagger(deterministic: bool = True) -> DecimalTagger:
    return DecimalTagger(cardinal=CardinalFst(deterministic=deterministic), deterministic=deterministic)


def test_deterministic_decimals():
    graph = _tagger().fst @ DecimalVerbalizer().fst
    cases = {
        '0.1': 'a náid ponc a haon',
        '7.55': 'a seacht ponc a cúig a cúig',
        '24.638': 'fiche a ceathair ponc a sé a trí a hocht',
        '-2.04': 'lúide a dó ponc a náid a ceathair',
        ',5': 'a náid ponc a cúig',
    }
    for written, spoken in cases.items():
        assert one_top_rewrite(written, graph) == spoken


def test_audio_decimal_separator_variants():
    graph = _tagger(deterministic=False).fst @ DecimalVerbalizer(deterministic=False).fst
    assert set(top_rewrites('2.5', graph, 10)) == {'a dó pointe a cúig', 'a dó ponc a cúig'}
