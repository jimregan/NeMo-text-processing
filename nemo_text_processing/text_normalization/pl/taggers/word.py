# Copyright (c) 2023 Jim O'Regan for Språkbanken Tal
#
# Licensed under the Apache License, Version 2.0 (the "License");
import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_SPACE, GraphFst
from pynini.lib import pynutil


class WordFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="classify", deterministic=deterministic)
        self.fst = (
            pynutil.insert('name: "') + pynini.closure(NEMO_NOT_SPACE, 1) + pynutil.insert('"')
        ).optimize()
