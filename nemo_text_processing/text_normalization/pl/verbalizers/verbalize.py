# Copyright (c) 2023 Jim O'Regan for Språkbanken Tal
#
# Licensed under the Apache License, Version 2.0 (the "License");
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.pl.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.pl.verbalizers.ordinal import OrdinalFst


class VerbalizeFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)
        self.fst = CardinalFst(deterministic=deterministic).fst | OrdinalFst(deterministic=deterministic).fst
