import os
import numpy as np

from tokenizers import Tokenizer
from typing import List, Union

import sentencepiece

tokenizer_path = os.path.join(os.path.dirname(__file__), "word_tokenizer.json")
spm_path = os.path.join(os.path.dirname(__file__), "sp.model")


class LanguageFilter:
    def __init__(self, unk_threshold=0.05):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.unk_threshold = unk_threshold
        self.unk_id = self.tokenizer.token_to_id(self.tokenizer.model.unk_token)

    def get_unk_ratios(self, lines: List[str]) -> np.ndarray:
        """
        Get the unk ratio for each line in lines.
        """
        unk_ratios = np.array(
            [
                sum(1 for t in x.ids if t == self.unk_id) / len(x)
                for x in self.tokenizer.encode_batch(lines)
            ]
        )
        return unk_ratios

    def __call__(self, lines: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Filters out lines that have a unk ratio above the unk threshold.
        Returns a boolean array of the same length as lines. 
        Where True means valid document.
        """
        if isinstance(lines, list):
            lines = np.array(lines, dtype=object)

        unk_ratios = self.get_unk_ratios(lines)
        return lines[unk_ratios < self.unk_threshold]


class RedundancyFilter:
    def __init__(self, threshold=0.25):
        self.spm = sentencepiece.SentencePieceProcessor(spm_path)
        self.threshold = threshold

    def get_token_char_ratio(self, lines: List[str]) -> np.ndarray:
        """
        Get the token/character ratio for each line in lines.
        """
        ratios = np.array(
            [
                len(tokens) / len(chars)
                for tokens, chars in zip(self.spm.encode(lines), lines)
            ]
        )
        return ratios

    def __call__(self, lines: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Filters out lines that have a token/character ratio above the threshold.
        Returns a boolean array of the same length as lines. 
        Where True means valid document.
        """
        if isinstance(lines, list):
            lines = np.array(lines, dtype=object)

        ratios = self.get_token_char_ratio(lines)
        return lines[ratios < self.threshold]
