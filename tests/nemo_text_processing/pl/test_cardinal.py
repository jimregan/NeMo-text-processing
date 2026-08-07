# Copyright (c) 2026 Jim O'Regan
#
# Licensed under the Apache License, Version 2.0 (the "License");
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.pl.taggers.cardinal import CardinalFst

from ..utils import parse_test_case_file


class TestCardinal:
    normalizer = Normalizer(input_case="cased", lang="pl", cache_dir=None, post_process=False)

    @parameterized.expand(parse_test_case_file("pl/data_text_normalization/test_cases_cardinal.txt"))
    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        prediction = self.normalizer.normalize(test_input, punct_post_process=False)
        assert prediction == expected

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_inflectional_graphs(self):
        cardinal = CardinalFst()
        assert rewrite.one_top_rewrite("1", cardinal.graphs["f_sg_nom"]) == "jedna"
        assert rewrite.one_top_rewrite("2", cardinal.graphs["mp_pl_nom"]) == "dwaj"
        assert rewrite.one_top_rewrite("22", cardinal.graphs["mp_pl_nom"]) == "dwudziestu dwóch"
        assert rewrite.one_top_rewrite("22", cardinal.graphs["f_pl_nom"]) == "dwadzieścia dwie"
        assert rewrite.one_top_rewrite("22", cardinal.graphs["pl_gen"]) == "dwudziestu dwóch"

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_compound_graph(self):
        cardinal = CardinalFst()
        assert rewrite.one_top_rewrite("22", cardinal.graphs["compound"]) == "dwudziestodwu"
        assert rewrite.one_top_rewrite("22-latka", cardinal.compound) == "dwudziestodwulatka"

        cardinal = CardinalFst(deterministic=False)
        alternatives = rewrite.top_rewrites("22", cardinal.graphs["compound"], 10)
        assert "dwudziestodwu" in alternatives
        assert "dwudziesto dwu" in alternatives
        alternatives = rewrite.top_rewrites("22-latka", cardinal.compound, 20)
        assert "dwudziestodwulatka" in alternatives
        assert "dwudziesto dwu latka" in alternatives
