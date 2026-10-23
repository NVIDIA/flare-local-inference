"""Run one rank's share of rank zero's direct steps: copy inputs, then replay a graph or call its forward.

A step is a plain message: the graph's name, the input byte ranges to write
(the same partial uploads Helios writes, leaving every other byte of the
resident input buffers unchanged), and the frame index and submission sequence
the graph's completion record carries. Rank zero decides every step and sends
it to each peer over the rank's executor connection before running it itself,
so every rank executes the same graphs in the same order; the collectives
inside each graph join the ranks' GPU work. A rank finishes each step before
taking the next, which keeps its pinned staging and the next step's uploads
ordered without events, then reports the step's graph time to rank zero.
Graph time spans the graph alone, from after its uploads until it completes,
as Helios's frame events do, and a step's time is its slowest rank's.

With `eager`, a rank calls the forward each graph captured, compiled where
preparation compiled it, on the same buffers instead of replaying the graph.
Eager execution releases the captured graphs first so their memory pool can
hold the forwards' temporaries.
"""

import math
import struct

import torch

from ignition.preparation.host_ranks import receive_json, receive_object, send_json, send_object

_RESERVED = ("frame_index", "submission_sequence")


class StepRunner:
    """Execute steps on one rank's prepared model."""

    def __init__(self, model, *, eager):
        if eager and model.forwards is None:
            raise RuntimeError("this recipe retains no prepared forwards; --eager requires them")
        self.model, self.eager = model, eager
        self.device = torch.device("cuda", model.device)
        self.inputs = {name: frozenset((*signature.inputs, *_RESERVED))
                       for name, signature in model.signatures.items()}
        self.packed = frozenset(name for name, signature in model.signatures.items()
                                if signature.kind == "packed_speculative")
        self.proposals = {}
        for name, signature in model.signatures.items():
            if signature.kind == "verification" and signature.token_capacity > 1:
                candidates = [(s.token_capacity, n) for n, s in model.signatures.items()
                              if s.kind == "proposal" and s.batch_capacity == signature.batch_capacity]
                self.proposals[name] = max(candidates)[1]
        self.staging = {}
        # Collective runtimes that report transport failures only when asked, after each step.
        health, seen = [], set()
        for resource in model.resources:
            if callable(getattr(resource, "check_health", None)) and id(resource) not in seen:
                seen.add(id(resource))
                health.append(resource)
        self.health = tuple(health)
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)
        if eager:
            for graph in model.graphs.values():
                graph.reset()
            torch.cuda.empty_cache()
        self.stream = torch.cuda.Stream(self.device)
        self.started = torch.cuda.Event(enable_timing=True)
        self.finished = torch.cuda.Event(enable_timing=True)

    def _upload(self, name, offset, data, *, staging_key=None):
        target = self.model.buffers[name].view(-1)
        size = target.element_size()
        if offset % size or len(data) % size or offset + len(data) > target.numel() * size or not data:
            raise ValueError(f"misaligned or out-of-range upload to {name}")
        key = name if staging_key is None else staging_key
        staging = self.staging.get(key)
        if staging is None:
            staging = self.staging[key] = torch.empty(target.numel(), dtype=target.dtype, pin_memory=True)
        start, count = offset // size, len(data) // size
        staging[start:start + count].copy_(torch.frombuffer(bytearray(data), dtype=target.dtype))
        target[start:start + count].copy_(staging[start:start + count], non_blocking=True)

    def run(self, step):
        """Execute one step and return its graph nanoseconds, from after its uploads through the graph.

        The interval starts where Helios's frame `started` event does, so input
        staging is not graph time. An eager step's interval is its forward's,
        Python launches included.
        """
        name = step["graph"]
        allowed = self.inputs.get(name)
        if allowed is None:
            raise ValueError(f"step names unknown graph {name!r}")
        with torch.cuda.stream(self.stream), torch.no_grad():
            for buffer, offset, data in step["uploads"]:
                if buffer not in allowed or buffer in _RESERVED:
                    raise ValueError(f"graph {name} does not bind input {buffer}")
                self._upload(buffer, offset, data)
            self._upload("frame_index", 0, struct.pack("<i", step["frame"]))
            proposal = self.proposals.get(name)
            self._upload("submission_sequence", 0, struct.pack("<q", step["sequence"] - bool(proposal)),
                         staging_key="proposal_sequence" if proposal is not None else None)
            self.started.record(self.stream)
            if proposal is not None:
                if self.eager:
                    self.model.forwards[proposal]()
                else:
                    self.model.graphs[proposal].replay()
                self._upload("submission_sequence", 0, struct.pack("<q", step["sequence"]))
            if self.eager:
                self.model.forwards[name]()
            else:
                self.model.graphs[name].replay()
            self.finished.record(self.stream)
        self.finished.synchronize()
        for runtime in self.health:
            runtime.check_health()
        # Helios rounds the event interval to nanoseconds half away from zero.
        return int(math.floor(self.started.elapsed_time(self.finished) * 1e6 + .5))

    def records(self, frame):
        """The completion records the last step wrote to `frame`, one row per request slot."""
        return self.model.buffers["completion_records"][frame].cpu().numpy()

    def close(self):
        torch.cuda.synchronize(self.device)


class RankSteps:
    """Rank zero's side: send every step to each peer, then run it here."""

    def __init__(self, model, peers, *, eager):
        self.runner = StepRunner(model, eager=eager)
        self.peers = tuple(peers)
        self.stopped = False

    def run(self, step):
        """Execute a step on every rank; returns its graph nanoseconds on the slowest rank, as Helios takes them."""
        for peer in self.peers:
            send_object(peer, step)
        nanos = self.runner.run(step)
        for rank, peer in enumerate(self.peers, 1):
            try:
                reply = receive_json(peer)
            except ConnectionError as error:
                raise RuntimeError(f"rank {rank} stopped during direct step {step['sequence']}") from error
            nanos = max(nanos, int(reply["graph_ns"]))
        return nanos

    def records(self, frame):
        return self.runner.records(frame)

    def confidence(self, graph):
        if graph not in self.runner.proposals and graph not in self.runner.packed:
            return None
        return self.runner.model.proposal_confidence.tolist()

    def stop(self):
        """End every peer's step loop; peers then close their models with this rank."""
        if self.stopped:
            return
        self.stopped = True
        for peer in self.peers:
            send_object(peer, None)
        self.runner.close()


def serve_peer(model, connection, *, eager):
    """Execute rank zero's steps, in order, reporting each one's graph time, until it sends the stop message."""
    runner = StepRunner(model, eager=eager)
    while (step := receive_object(connection)) is not None:
        send_json(connection, {"graph_ns": runner.run(step)})
    runner.close()
