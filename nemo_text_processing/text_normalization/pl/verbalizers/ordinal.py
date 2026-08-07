# Copyright (c) 2023 Jim O'Regan for Språkbanken Tal
#
# Licensed under the Apache License, Version 2.0 (the "License");
import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="ordinal", kind="verbalize", deterministic=deterministic)
        integer = pynutil.delete("integer:") + delete_space + pynutil.delete('"')
        integer += pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete('"')
        self.fst = self.delete_tokens(integer).optimize()
