"""Predecessor-conditioned low-rank corrections to a vocabulary shortlist."""

import torch

from .linear import UnquantizedProjection


class PreparedCandidateSelector:
    def __init__(self, hidden, weights, *, top_k, session, name):
        self.weights, self.top_k = weights, top_k
        self.projected = torch.zeros((*hidden.shape[:-1], weights["projection"].shape[0]),
                                     dtype=hidden.dtype, device=hidden.device)
        self.unary = torch.empty((*hidden.shape[:-1], top_k), dtype=torch.float32, device=hidden.device)
        self.ids = torch.empty(self.unary.shape, dtype=torch.int64, device=hidden.device)
        self.projection = UnquantizedProjection(hidden.flatten(0, -2), weights["projection"],
            self.projected.flatten(0, -2), session=session, name=name + ".hidden")

    def prepare(self, hidden):
        self.projection(hidden.flatten(0, -2))

    def prepare_shortlists(self, logits):
        unary, ids = logits.topk(self.top_k, dim=-1, sorted=False)
        self.unary.copy_(unary)
        self.ids.copy_(ids)

    def apply_shortlist(self, logits, predecessor, position):
        self._correct(logits, predecessor, position, self.unary[:, position], self.ids[:, position])

    def __call__(self, logits, predecessor, position):
        unary, ids = logits.topk(self.top_k, dim=-1, sorted=False)
        self._correct(logits, predecessor, position, unary, ids)

    def _correct(self, logits, predecessor, position, unary, ids):
        coded = self.weights["predecessor"][predecessor] * self.projected[:, position]
        successors = self.weights["successor"][ids]
        correction = torch.bmm(successors, coded.unsqueeze(-1)).squeeze(-1)
        logits.fill_(-torch.inf)
        logits.scatter_(1, ids, unary + correction.float())
