# Copyright (c) 2026 Jim O'Regan
#
# Licensed under the Apache License, Version 2.0 (the "License");
from typing import Optional

import pynini
from nemo_text_processing.text_normalization.pl.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.text_normalization.pl.verbalizers.verbalize_final import VerbalizeFinalFst


class AudioNormalizerFst:
    """Builds and scores the non-deterministic Polish written-to-spoken lattice."""

    def __init__(self, input_case: str = "cased"):
        classifier = ClassifyFst(input_case=input_case, deterministic=False)
        verbalizer = VerbalizeFinalFst(deterministic=False)
        self.fst = (classifier.fst @ verbalizer.fst).optimize()

    def lattice(self, text: str) -> 'pynini.Fst':
        escaped_text = pynini.escape(text)
        lattice = pynini.compose(pynini.accep(escaped_text), self.fst)
        if lattice.start() == pynini.NO_STATE_ID:
            raise ValueError(f"Polish TN failed for input: {text}")
        return lattice

    def normalize(self, text: str, lm: Optional['pynini.FstLike'] = None) -> str:
        lattice = self.lattice(text)
        if lm is not None:
            lattice = pynini.compose(lattice, lm)
            if lattice.start() == pynini.NO_STATE_ID:
                raise ValueError("The language model rejected every Polish TN path")
        best = pynini.shortestpath(lattice, nshortest=1, unique=True).project("output")
        return best.string()
