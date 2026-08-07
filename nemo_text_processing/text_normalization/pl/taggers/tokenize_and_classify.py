# Copyright (c) 2023 Jim O'Regan for Språkbanken Tal
#
# Licensed under the Apache License, Version 2.0 (the "License");
import os

import pynini
from nemo_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.pl.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.pl.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.pl.taggers.word import WordFst
from nemo_text_processing.utils.logging import logger
from pynini.lib import pynutil


class ClassifyFst(GraphFst):
    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"pl_tn_{deterministic}_{input_case}_tokenize.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            return

        self.cardinal = CardinalFst(deterministic=deterministic)
        self.ordinal = OrdinalFst(deterministic=deterministic)
        word = WordFst(deterministic=deterministic).fst
        punctuation = PunctuationFst(deterministic=deterministic).fst
        classify = (
            pynutil.add_weight(self.ordinal.fst, 1.09)
            | pynutil.add_weight(self.cardinal.fst, 1.1)
            | pynutil.add_weight(punctuation, 2.1)
            | pynutil.add_weight(word, 100)
        )
        token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
        graph = delete_space + token + pynini.closure(delete_extra_space + token) + delete_space
        self.fst = graph.optimize()
        if far_file:
            generator_main(far_file, {"tokenize_and_classify": self.fst})
            logger.info(f"ClassifyFst grammar was saved to {far_file}.")
