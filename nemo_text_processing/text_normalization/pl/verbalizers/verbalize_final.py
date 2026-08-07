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
from nemo_text_processing.text_normalization.en.verbalizers.word import WordFst
from nemo_text_processing.text_normalization.pl.verbalizers.verbalize import VerbalizeFst
from pynini.lib import pynutil


class VerbalizeFinalFst(GraphFst):
    def __init__(self, deterministic: bool = True, cache_dir: str = None, overwrite_cache: bool = False):
        super().__init__(name="verbalize_final", kind="verbalize", deterministic=deterministic)
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"pl_tn_{deterministic}_verbalizer.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["verbalize"]
            return

        types = VerbalizeFst(deterministic=deterministic).fst | WordFst(deterministic=deterministic).fst
        graph = (
            pynutil.delete("tokens")
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + types
            + delete_space
            + pynutil.delete("}")
        )
        self.fst = (delete_space + pynini.closure(graph + delete_extra_space) + graph + delete_space).optimize()
        if far_file:
            generator_main(far_file, {"verbalize": self.fst})
