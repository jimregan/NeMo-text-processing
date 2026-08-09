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
from nemo_text_processing.text_normalization.ga.taggers.time import TimeFst as TimeTagger
from nemo_text_processing.text_normalization.ga.verbalizers.time import TimeFst as TimeVerbalizer


def _verbalize(tagged: str, deterministic: bool) -> str:
    return one_top_rewrite(tagged, TimeVerbalizer(deterministic=deterministic).fst)


def test_deterministic_half_past_by_dialect():
    cardinal = CardinalFst()
    expected = {
        'co': 'leathuair tar éis a sé',
        'gm': 'leathuair tréis a sé',
        'gc': 'leathuair théis a sé',
    }
    for dialect, spoken in expected.items():
        tagger = TimeTagger(cardinal=cardinal, deterministic=True, dialect=dialect)
        assert _verbalize(one_top_rewrite('6:30', tagger.fst), deterministic=True) == spoken


def test_audio_half_past_includes_all_known_dialects():
    tagger = TimeTagger(cardinal=CardinalFst(), deterministic=False, dialect='co')
    tagged = top_rewrites('6:30', tagger.fst, 10)
    spoken = {_verbalize(candidate, deterministic=False) for candidate in tagged}
    assert spoken == {'leathuair tar éis a sé', 'leathuair tréis a sé', 'leathuair théis a sé'}
