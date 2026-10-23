"""Functional probability snapshots used as an exact verification oracle."""

from ignition.components.speculation import (
    PreparedStochasticVerification, TARGET_RNG_DOMAIN, _live, _sliced, verify_chain,
)
import torch


class FunctionalVerificationReference(PreparedStochasticVerification):
    """Validate and mask complete tensors before copying into caller storage."""

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("chain verification requires prepare() before capture")
        out = self.output
        safe, live = _live(self.state, self.slots, self.generations, self.live_rows, self.rows)
        if self.token_limits is None:
            valid = live & self.draft_valid.all(-1) & self.target_valid.all(-1)
            valid &= ((self.drafts >= 0) & (self.drafts < self.state.vocab_size)).all(-1)
            draft_probs, target_probs, ids = self.draft_probs, self.target_probs, self.drafts
        else:
            target_used = self.columns[None, :] < self.token_limits[:, None]
            draft_used = self.columns[None, :-1] < self.token_limits[:, None] - 1
            valid = live & (self.draft_valid | ~draft_used).all(-1) & (self.target_valid | ~target_used).all(-1)
            valid &= (((self.drafts >= 0) & (self.drafts < self.state.vocab_size)) | ~draft_used).all(-1)
            valid &= (self.token_limits >= 1) & (self.token_limits <= self.target_probs.shape[1])
            draft_probs = torch.where(draft_used[:, :, None], self.draft_probs, self.dummy)
            target_probs = torch.where(target_used[:, :, None], self.target_probs, self.dummy)
            ids = torch.where(draft_used, self.drafts, 0)
        for probabilities in (draft_probs, target_probs):
            slices = _sliced(probabilities)
            entries = (torch.isfinite(slices) & (slices >= 0)).all(-1).all(-1)
            mass = slices.sum(-1).sum(-1)
            valid &= (entries & torch.isfinite(mass) & ((mass - 1).abs() <= 1e-4)).all(-1)
        self.safe_draft_probs.copy_(torch.where(valid[:, None, None], draft_probs, self.dummy))
        self.safe_target_probs.copy_(torch.where(valid[:, None, None], target_probs, self.dummy))
        self.safe_ids.copy_(torch.where(valid[:, None], ids, 0))
        self.seed_bits.copy_(self.state.seed.view(torch.int64)[safe] ^ TARGET_RNG_DOMAIN)
        self.offset.copy_(torch.where(self.rng.generation[safe] == self.generations, self.rng.target_offset[safe], 0))
        self.independent_accepted.zero_()
        self.emitted_drafts.zero_()
        tokens, _, _ = verify_chain(self.safe_draft_probs, self.safe_ids, self.safe_target_probs,
            self.independent_accepted, self.emitted_drafts, seed=self.seed_bits, offset=self.offset)
        accepted = self.emitted_drafts
        if self.token_limits is None:
            out.offered_drafts.fill_(self.drafts.shape[1])
        else:
            # The fixed-width library may accept dummy columns beyond a short
            # tail. Only the real prefix belongs to the commit contract.
            out.offered_drafts.copy_((self.token_limits - 1).clamp(0, self.drafts.shape[1]))
            accepted = torch.minimum(accepted, out.offered_drafts)
        out.token_ids.copy_(tokens)
        out.accepted_drafts.copy_(torch.where(valid, accepted, 0))
        out.proposed_count.copy_(torch.where(valid, accepted + 1, 0))
        out.valid.copy_(valid)
        out.rng_increment.copy_(torch.where(valid, self.draw_reservation, 0))
        out.token_ids.masked_fill_(~valid[:, None], -1)
        out.survival.fill_(-1.)
        prefix = torch.ones_like(self.offset, dtype=torch.float32)
        for depth in range(self.drafts.shape[1]):
            index = self.safe_ids[:, depth:depth + 1].long()
            q = self.safe_draft_probs[:, depth].gather(1, index).squeeze(1)
            p = self.safe_target_probs[:, depth].gather(1, index).squeeze(1)
            prefix = prefix * torch.where(q > 0, torch.minimum(p / q, torch.ones_like(p)), (p > 0).float())
            out.survival[:, depth].copy_(torch.where(valid, prefix, -1.))
        if self.broadcast is not None:
            self.broadcast()
        return out
