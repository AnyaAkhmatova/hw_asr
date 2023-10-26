from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        empty_tok_ind = self.char2ind[self.EMPTY_TOK]
        last_ind = empty_tok_ind
        new_inds = []
        for ind in inds:
            if ind == empty_tok_ind:
                last_ind = empty_tok_ind
            elif ind == last_ind:
                continue
            else:
                last_ind = ind
                new_inds.append(ind)
        return self.decode(new_inds)        

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        states = {('', self.EMPTY_TOK) : 1.0}
        new_states = {}
        char_length = min(char_length, probs_length)

        for i in range(char_length):
            for j in range(voc_size):
                for (text, last_char), proba in states.items():
                    new_last_char = self.ind2char[j]
                    new_text = text
                    if new_last_char != self.EMPTY_TOK and new_last_char != last_char:
                        new_text += new_last_char
                    if (new_text, new_last_char) not in new_states:
                        new_states[(new_text, new_last_char)] = proba * probs[i, j]
                    else:
                        new_states[(new_text, new_last_char)] += proba * probs[i, j]

            new_states = sorted(list(new_states.items()), key=lambda x: -x[1])[:beam_size]
            new_states = {k: v for k, v in new_states}
            states = dict(new_states)
            new_states = {}

        states = sorted(list(states.items()), key=lambda x: -x[1])
        hypos = [Hypothesis(text, proba) for (text, last_char), proba in states]
        return hypos
    
