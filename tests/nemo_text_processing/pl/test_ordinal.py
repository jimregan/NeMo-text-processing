# Copyright (c) 2026 Jim O'Regan
#
# Licensed under the Apache License, Version 2.0 (the "License");
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.pl.taggers.ordinal import OrdinalFst

from ..utils import parse_test_case_file


class TestOrdinal:
    normalizer = Normalizer(input_case="cased", lang="pl", cache_dir=None, post_process=False)

    @parameterized.expand(parse_test_case_file("pl/data_text_normalization/test_cases_ordinal.txt"))
    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        prediction = self.normalizer.normalize(test_input, punct_post_process=False)
        assert prediction == expected

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_inflectional_graphs(self):
        ordinal = OrdinalFst()
        assert rewrite.one_top_rewrite("21", ordinal.graphs["f_sg_nom"]) == "dwudziesta pierwsza"
        assert rewrite.one_top_rewrite("21", ordinal.graphs["mi_sg_gen"]) == "dwudziestego pierwszego"
