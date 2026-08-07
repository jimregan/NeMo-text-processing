# Copyright (c) 2026 Jim O'Regan
#
# Licensed under the Apache License, Version 2.0 (the "License");
import pytest

from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio


class TestNormalizeWithAudio:
    normalizer = NormalizerWithAudio(input_case="cased", lang="pl", cache_dir=None, post_process=False)

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_cardinal_inflections_are_in_lattice(self):
        predictions = self.normalizer.normalize("Mam 22 koty", n_tagged=100, punct_post_process=False)
        assert "Mam dwadzieścia dwa koty" in predictions
        assert "Mam dwadzieścia dwie koty" in predictions
        assert "Mam dwudziestu dwóch koty" in predictions

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_ordinal_inflections_are_in_lattice(self):
        predictions = self.normalizer.normalize("To był 21. test", n_tagged=100, punct_post_process=False)
        assert "To był dwudziesty pierwszy test" in predictions
        assert "To był dwudziesta pierwsza test" in predictions
        assert "To był dwudziestego pierwszego test" in predictions
