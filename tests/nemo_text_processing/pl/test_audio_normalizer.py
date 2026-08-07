# Copyright (c) 2026 Jim O'Regan
#
# Licensed under the Apache License, Version 2.0 (the "License");
import pynini
import pytest

from nemo_text_processing.text_normalization.pl.audio_normalizer import AudioNormalizerFst


class TestAudioNormalizer:
    normalizer = AudioNormalizerFst()

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_ngram_acceptor_selects_inflection(self):
        masculine = pynini.accep("Mam dwadzieścia dwa koty", weight=1)
        feminine = pynini.accep("Mam dwadzieścia dwie koty", weight=0)
        language_model = (masculine | feminine).optimize()
        assert self.normalizer.normalize("Mam 22 koty", language_model) == "Mam dwadzieścia dwie koty"

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_ngram_acceptor_selects_compound_tokenization(self):
        joined = pynini.accep("To dwudziestodwulatka", weight=1)
        spaced = pynini.accep("To dwudziesto dwu latka", weight=0)
        language_model = (joined | spaced).optimize()
        assert self.normalizer.normalize("To 22-latka", language_model) == "To dwudziesto dwu latka"
