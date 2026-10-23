"""Sample from GLM's cached final normalized hidden output without token consumption."""

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.resume import PreparedResumeState
from ignition.components.sampling import PreparedCompletion, PreparedSampling
from ignition.components.vocabulary import PreparedVocabularyProjection


class PreparedResume:
    def __init__(self, weights, requests, pool, inputs, records, frame, sequence, *, rows,
                 runtime, vocabulary_runtime, coordinator, session, name, replicated_sampling=False):
        self.state = PreparedResumeState(requests, pool.last_hidden, inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows], inputs.live_rows, world_size=weights.shard.world_size, state_rows=inputs.request_state_rows)
        self.validation = PreparedAllReduce(self.state.validation_input, self.state.validation_sum,
            runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.validation")
        self.local_logits = torch.empty((rows, 154880 // weights.shard.world_size),
            dtype=torch.bfloat16, device=self.state.hidden.device)
        self.logits = torch.empty((rows, 154880), dtype=torch.bfloat16, device=self.state.hidden.device)
        self.projection = PreparedVocabularyProjection(self.state.hidden, weights.head,
            self.local_logits, self.logits, runtime=vocabulary_runtime, coordinator=coordinator,
            session=session, name=f"{name}.lm_head")
        arguments = (requests, inputs.row_slot_ids[:rows], inputs.row_generations[:rows], inputs.live_rows,
                     records, frame, sequence)
        self.completion = PreparedCompletion(*arguments) if weights.shard.rank != 0 and not replicated_sampling else PreparedSampling(self.logits, *arguments)
        self.completion.prepare()
        if replicated_sampling:
            self.completion.prepare_broadcast(runtime=vocabulary_runtime, session=session,
                coordinator=coordinator, name=name + ".sampling")
        self.compilation_variant = "sample" if weights.shard.rank == 0 or replicated_sampling else "consume"
        self.replicated_sampling = replicated_sampling

    def __call__(self):
        self.validation(self.state.validate())
        hidden = self.state.gather()
        self.projection(hidden)
        self.completion()
