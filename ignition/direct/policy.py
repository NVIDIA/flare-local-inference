"""Helios's rectangular decode-round selection, mirrored for direct execution.

Direct execution keeps its own fixed choice of step kinds, but chooses every
decode round's graph as Helios's `RequestScheduler::select_plan`
(helios/src/scheduler/policy.rs) chooses among that kind's candidates: every
prepared batch capacity up to the concurrency's power of two, at every loaded
verification width (only the fixed width when verification is not adaptive),
carries its highest-valued requests (`best_cohort`), and the graph with the best
expected value per graph time wins. A request's value is its expected output
tokens, from Helios's per-depth acceptance estimate, priced by its token and
delivery fairness, plus its output age (`LaneValue`, `lane_values`). Graph times
are Helios's expiring estimates (`ActionCosts`): unmeasured until a step of that
shape runs, then a 7/8 moving average of observed graph time, forgotten after
30 s without an observation. An unmeasured or forgotten graph is explored before
any measured one, by value alone (`ValueRate::compare`), so every shape is tried
at startup and neglected shapes are tried again every 30 s.

Prices follow Helios's service accounting (helios/src/scheduler/service.rs):
initialized when a step is committed, charged the graph time of every completed
step, and credited with committed tokens and published deliveries. Each
rectangular selection function names the Helios code it mirrors; floating-point
operations follow the same order so those paths choose identically from identical
inputs. Packed candidates use the complete captured shape and causal prefix
forecasts from the latest completed conditional confidence. They do not reproduce
Helios's retained-proposal forecasts or regrouping.
"""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass, field

# helios/src/scheduler/policy.rs PolicyConfig::default.
OUTPUT_TOKEN_VALUE = 64.
TOKEN_FAIRNESS = 16.
DELIVERY_FAIRNESS = 4.
DECODE_AGE_PRICE = .1
ADAPTATION_MS = 100.
ESTIMATE_TTL_NS = 30_000_000_000
# helios/src/execution/manifest.rs.
BATCH_CAPACITIES = (1, 2, 4, 8, 16, 32, 64)
REQUEST_LANES = 64

_KINDS = {"init": "init", "restore": "restore", "prefill": "prefill", "resume": "resume", "snapshot": "snapshot"}


class GraphCosts:
    """Expiring durations by complete captured graph shape; zero is indeterminate."""

    def __init__(self, shapes, ttl_ns=ESTIMATE_TTL_NS):
        self.entries = {shape: [0, 0] for shape in shapes}
        self.now_ns, self.ttl_ns = 0, ttl_ns

    def _fresh(self, entry):
        return entry[0] != 0 and max(0, self.now_ns - entry[1]) < self.ttl_ns

    def duration(self, kind, batch, tokens, packed=0, drafts=0):
        """None for an unprepared shape, 0 for an unmeasured or expired one, else nanoseconds."""
        key = (kind, batch, tokens)
        entry = self.entries.get(key + (packed, drafts) if packed else key)
        if entry is None:
            return None
        return entry[0] if self._fresh(entry) else 0

    def observe(self, shape, nanos):
        if nanos is None or nanos <= 0:
            return
        entry = self.entries.get(shape)
        if entry is not None:
            entry[0] = (7 * entry[0] + nanos + 4) // 8 if self._fresh(entry) else nanos
            entry[1] = self.now_ns

    def estimates(self):
        """The last estimate of every measured shape, fresh or not."""
        return {shape: nanos for shape, (nanos, _) in self.entries.items() if nanos}


class ServicePrice:
    """A request's marginal service price over its fairness coefficient, in log form. Mirrors utility.rs."""

    __slots__ = ("log_inverse_rate",)

    def __init__(self, log_inverse_rate):
        self.log_inverse_rate = log_inverse_rate

    @classmethod
    def initial(cls, rate):
        assert math.isfinite(rate) and rate > 0
        return cls(-math.log(rate))

    def copy(self):
        return ServicePrice(self.log_inverse_rate)

    def log_price(self, coefficient):
        return math.log(coefficient) + self.log_inverse_rate

    def observe(self, duration_ms, delivered, maximum_rate, adaptation_ms):
        previous = self.log_inverse_rate
        if delivered > 0.:
            normalized = delivered / maximum_rate
            decay = normalized / adaptation_ms
            previous -= decay
            incoming = math.log(-math.expm1(-decay)) + math.log(duration_ms) - math.log(delivered)
        else:
            incoming = math.log(duration_ms) - math.log(adaptation_ms) - math.log(maximum_rate)
        maximum = max(previous, incoming)
        self.log_inverse_rate = maximum + math.log(math.exp(previous - maximum) + math.exp(incoming - maximum))


@dataclass
class Accounting:
    """Helios's `RequestAccounting` and `ServiceAccounting` for one request; times are scheduler nanoseconds."""

    arrival_ns: int = 0
    last_progress_ns: int = 0
    last_output_ns: int | None = None
    last_output_fence: int | None = None
    published_deliveries: int = 0
    pending_publication_fence: int | None = None
    tokens: ServicePrice | None = None
    deliveries: ServicePrice | None = None
    pending_delivery: tuple | None = None


@dataclass
class LaneValue:
    """Mirrors utility.rs `LaneValue`; values share one scaling factor within a decision."""

    prefill_token: float = 0.
    output_token: float = 0.
    delivery: float = 0.
    age_price: float = 0.
    age_ms: float = 0.
    reference_delivery_ms: float = 0.

    def decode(self, expected_output, duration_ms):
        return (self.output_token * expected_output + self.delivery
                + self.age_price * self.reference_delivery_ms * (self.age_ms + duration_ms))


def value_rate(duration_ns, value, age_price):
    """utility.rs `ValueRate`, as a key whose order is `ValueRate::compare`: unmeasured first, then score."""
    duration_ms = duration_ns / 1e6
    score = value if duration_ns == 0 else value / duration_ms - age_price * duration_ms / 2.
    return duration_ns == 0, score, -duration_ns


def best_cohort(offers, capacity):
    """utility.rs `best_cohort`: the mask of the `capacity` highest values, lower slots first among equals."""
    limit = min(capacity, REQUEST_LANES)
    selected = []
    for slot, value in offers:
        insertion = next((index for index, (existing, score) in enumerate(selected)
                          if value > score or (value == score and slot < existing)), len(selected))
        if insertion < limit:
            selected.insert(insertion, (slot, value))
            del selected[limit:]
    mask = 0
    for slot, _ in selected:
        mask |= 1 << slot
    return mask


@dataclass
class DecodeRound:
    batch: int
    width: int
    requests: list
    duration_ns: int
    kind: str = "speculative"
    packed_tokens: int = 0
    draft_tokens: int = 0

    @property
    def shape(self):
        key = (self.kind, self.batch, self.width)
        return key + (self.packed_tokens, self.draft_tokens) if self.packed_tokens else key


@dataclass
class Participants:
    """Rates a committed step's requests had, which its completion charges; service.rs `ServiceParticipant`."""

    rows: list = field(default_factory=list)


class Policy:
    """Helios's graph costs, service prices and decode-round choice over direct execution's requests.

    Requests are direct `Request`s: `phase`, `prompt`, `consumed`, `generated`,
    `max_new_tokens`, `acceptance`, `accounting`, `slot`, `last_use` and
    `finished_early` stand for the `LiveRequest` fields Helios reads.
    """

    def __init__(self, layout):
        self.layout = layout
        self.costs = GraphCosts(layout.graphs)
        self.speculative = layout.speculative
        self.capacity = 1 << (layout.max_active - 1).bit_length()
        self.batches = tuple(batch for batch in BATCH_CAPACITIES if batch <= self.capacity)
        # speculation.rs `graph_catalog`, restricted to loaded shapes as `speculative_graphs` restricts it.
        self.catalog = {batch: tuple(width for width in sorted(layout.widths)
                                     if ("speculative", batch, width) in layout.graphs
                                     and (layout.fixed_width in (None, width)
                                          or (width == 1 and ("speculative", batch, layout.fixed_width) not in layout.graphs)))
                        for batch in self.batches}
        self.packed_catalog = tuple(key for key in layout.packed_shapes
                                    if key[1] <= self.capacity and layout.fixed_width in (None, key[2]))
        self.prefill_catalog = tuple(layout.capacities("prefill"))
        self.now_ns = 0

    def advance_to(self, now_ns):
        if now_ns < self.now_ns:
            raise ValueError("scheduler clock must be monotonic")
        self.now_ns = self.costs.now_ns = now_ns

    # Request views of Helios's `LiveRequest`.

    def kind(self, request):
        """policy.rs `request_kind`."""
        if request.phase == "decode":
            return "speculative" if self.speculative else "decode"
        return _KINDS.get(request.phase)

    @staticmethod
    def terminal(request):
        """policy.rs `terminal_cleanup`: a prompt snapshot after which the request finishes."""
        return request.phase == "snapshot" and request.finished_early

    def speculative_tokens(self, request, width):
        """scheduler.rs `speculative_tokens` for model proposals without host stops or grammar."""
        return min(width, self.layout.context - request.consumed, request.max_new_tokens - len(request.generated))

    @staticmethod
    def prefill_work(request, capacity):
        """scheduler.rs `prefill_work` without system boundaries or image spans."""
        return min(len(request.prompt), request.consumed + capacity) - request.consumed

    def eligible(self, request):
        """policy.rs `eligible` for a decoding request; direct execution reserves every page at admission."""
        return request.consumed < self.layout.context and len(request.generated) < request.max_new_tokens

    # service.rs.

    def service_rates(self, request, expected):
        """service.rs `service_rates`: the best tokens and deliveries per millisecond any measured graph gives."""
        token_rate = delivery_rate = 0.
        costs = self.costs

        def consider(kind, batch, capacity, tokens, delivery, packed=0, drafts=0):
            nonlocal token_rate, delivery_rate
            ns = costs.duration(kind, batch, capacity, packed, drafts)
            if ns:
                per_ms = 1e6 / ns
                token_rate = max(token_rate, tokens * per_ms)
                if delivery:
                    delivery_rate = max(delivery_rate, per_ms)

        prompt, generated = len(request.prompt), len(request.generated)
        if self.terminal(request):
            for batch in self.batches:
                consider("snapshot", batch, 1, 0., True)
        elif request.consumed < prompt:
            for capacity in self.prefill_catalog:
                work = self.prefill_work(request, capacity)
                if work == 0:
                    continue
                first = request.consumed + work == prompt
                consider("prefill", 1, capacity, work + float(first), False)
        else:
            for batch in self.batches:
                if generated == 0:
                    consider("resume", batch, 1, 1., True)
                elif generated < request.max_new_tokens:
                    if not self.speculative:
                        consider("decode", batch, 1, 1., True)
                    else:
                        for width in self.catalog[batch]:
                            tokens = self.speculative_tokens(request, width)
                            if tokens > 0:
                                consider("speculative", batch, width, expected(request, tokens), True)
            if generated and generated < request.max_new_tokens:
                for kind, batch, width, packed, drafts in self.packed_catalog:
                    tokens = self.speculative_tokens(request, min(width, drafts + 1) if drafts else width)
                    if tokens > 0:
                        consider(kind, batch, width, expected(request, min(tokens, packed)), True, packed, drafts)
        return (token_rate if token_rate > 0. else None, delivery_rate if delivery_rate > 0. else None)

    def _family(self, request):
        if request.phase == "decode":
            return "speculative" if self.speculative else "decode"
        if request.phase == "resume":
            return "resume"
        if request.phase == "snapshot" and self.terminal(request):
            return "snapshot"
        return None

    def delivery_references(self, requests, rates):
        """service.rs `delivery_reference_intervals` over each request's delivery rate."""
        count = float(sum(rate is not None for rate in rates.values()))
        references = {slot: None if rate is None else count / rate for slot, rate in rates.items()}
        for kind in ("decode", "speculative", "resume", "snapshot"):
            members = [r.slot for r in requests if rates.get(r.slot) is not None and self._family(r) == kind]
            streams = len(members)
            if streams == 0:
                continue
            per_delivery_ms = math.inf
            for batch in self.batches:
                for width in (self.catalog[batch] if kind == "speculative" else (1,)):
                    ns = self.costs.duration(kind, batch, width)
                    if ns:
                        per_delivery_ms = min(per_delivery_ms, ns / 1e6 / min(batch, streams))
            if kind == "speculative":
                for packed_kind, batch, width, packed, drafts in self.packed_catalog:
                    ns = self.costs.duration(packed_kind, batch, width, packed, drafts)
                    if ns:
                        per_delivery_ms = min(per_delivery_ms, ns / 1e6 / min(batch, streams))
            if math.isfinite(per_delivery_ms):
                for slot in members:
                    references[slot] = count * per_delivery_ms
        return references

    def participants(self, active, rates):
        """service.rs `service_participants`: initialize missing prices and record each request's rates."""
        references = self.delivery_references(active, {r.slot: rates[r.slot][1] for r in active})
        count = len(active)
        rows = []
        for request in active:
            token_rate, delivery_rate = rates[request.slot]
            accounting = request.accounting
            if token_rate is not None and accounting.tokens is None:
                accounting.tokens = ServicePrice.initial(token_rate / count)
            if references[request.slot] is not None and accounting.deliveries is None:
                accounting.deliveries = ServicePrice.initial(1. / references[request.slot])
            rows.append((request, token_rate, delivery_rate))
        return Participants(rows)

    def observe_service(self, sequence, participants, useful, active, duration_ns):
        """service.rs `observe_service`: charge a completed step's graph time to every participant."""
        for request in active:
            request.accounting.pending_delivery = None
        if duration_ns is None or duration_ns <= 0 or participants is None:
            return
        duration_ms = duration_ns / 1e6
        for request, token_rate, delivery_rate in participants.rows:
            accounting = request.accounting
            tokens = useful.get(request.slot, 0)
            output = accounting.pending_publication_fence == sequence and tokens > 0
            if accounting.tokens is not None and token_rate is not None:
                accounting.tokens.observe(duration_ms, float(tokens), token_rate, ADAPTATION_MS)
            if accounting.deliveries is not None and delivery_rate is not None:
                if output:
                    accounting.pending_delivery = (sequence, accounting.deliveries.copy(), duration_ms, delivery_rate)
                accounting.deliveries.observe(duration_ms, 0., delivery_rate, ADAPTATION_MS)
            elif output and accounting.deliveries is None:
                rate = 1. / duration_ms
                accounting.pending_delivery = (sequence, ServicePrice.initial(rate), duration_ms, rate)

    def published(self, request, sequence):
        """scheduler `published_output`: one delivery per completed step that emitted visible output."""
        accounting = request.accounting
        if accounting.last_output_fence != sequence:
            accounting.published_deliveries += 1
            accounting.last_output_fence = sequence
            if accounting.pending_delivery is not None:
                _, before, duration_ms, maximum_rate = accounting.pending_delivery
                accounting.pending_delivery = None
                price = before.copy()
                price.observe(duration_ms, 1., maximum_rate, ADAPTATION_MS)
                accounting.deliveries = price
        accounting.last_output_ns = self.now_ns

    # policy.rs.

    def rates(self, active):
        """Each live request's service rates, from one expected-token table per request."""
        tables = {}

        def expected(request, tokens):
            table = tables.get(request.slot)
            if table is None:
                table = tables[request.slot] = request.acceptance.expectations(self.layout.max_width)
            return table[tokens]

        return {r.slot: self.service_rates(r, expected) for r in active}, expected

    def lane_values(self, active, rates):
        """policy.rs `lane_values` with no submission in flight (a zero-length window)."""
        live = [r for r in active if self.kind(r) is not None]
        count = max(len(live), 1)
        references = self.delivery_references(live, {r.slot: rates[r.slot][1] for r in live})
        values, token_log, delivery_log = {}, {}, {}
        scale = max(0., math.log(OUTPUT_TOKEN_VALUE), math.log(DECODE_AGE_PRICE))
        for request in live:
            slot, accounting = request.slot, request.accounting
            tokens, deliveries = rates[slot]
            value = values[slot] = LaneValue()
            token_log[slot] = delivery_log[slot] = -math.inf
            if tokens is not None:
                price = accounting.tokens or ServicePrice.initial(tokens / count)
                token_log[slot] = price.log_price(TOKEN_FAIRNESS)
            reference = references[slot]
            if reference is not None and deliveries is not None:
                price = accounting.deliveries or ServicePrice.initial(1. / reference)
                delivery_log[slot] = price.log_price(DELIVERY_FAIRNESS)
                value.reference_delivery_ms = reference
                value.age_price = DECODE_AGE_PRICE
                last = accounting.last_output_ns
                value.age_ms = max(0, self.now_ns - (accounting.last_progress_ns if last is None else last)) / 1e6
            scale = max(scale, token_log[slot], delivery_log[slot])
        unit = math.exp(-scale)
        age_price = 0.
        for request in live:
            slot, value = request.slot, values[request.slot]
            value.prefill_token = unit + math.exp(token_log[slot] - scale)
            value.output_token = OUTPUT_TOKEN_VALUE * unit + math.exp(token_log[slot] - scale)
            value.delivery = math.exp(delivery_log[slot] - scale)
            value.age_price *= unit
            age_price += value.age_price
        return values, age_price

    def decode_round(self, decoding, active, rates, expected):
        """The decode candidate `select_plan` prefers: graph, width and requests; None when none is prepared.

        Mirrors `candidates`/`add_candidate` for the decode kind and `select_plan`'s
        order: value rate, more requests, older last use, then smaller batch, width
        and slot mask.
        """
        values, age_price = self.lane_values(active, rates)
        kind = "speculative" if self.speculative else "decode"
        ready = [r for r in decoding if self.eligible(r)]
        best = best_key = None
        for batch in self.batches:
            for width in (self.catalog[batch] if self.speculative else (1,)):
                duration_ns = self.costs.duration(kind, batch, width)
                if duration_ns is None or not ready:
                    continue
                duration_ms = duration_ns / 1e6
                offers = [(r.slot, values[r.slot].decode(
                    expected(r, self.speculative_tokens(r, width)) if self.speculative else 1., duration_ms))
                    for r in ready]
                mask = best_cohort(offers, batch)
                value = sum(v for slot, v in offers if mask & (1 << slot))
                members = [r for r in ready if mask & (1 << r.slot)]
                key = (value_rate(duration_ns, value, age_price), len(members),
                       -sum(r.last_use for r in members), -batch, -width, -mask)
                if best_key is None or key >= best_key:
                    best, best_key = DecodeRound(batch, width, members, duration_ns, kind), key
        for packed_kind, batch, width, packed, drafts in self.packed_catalog:
            duration_ns = self.costs.duration(packed_kind, batch, width, packed, drafts)
            limit = min(width, drafts + 1) if drafts else width
            cohort = self.packed_cohort(ready, batch, limit, packed, values, duration_ns, expected)
            if cohort is None:
                continue
            members, value = cohort
            mask = sum(1 << r.slot for r in members)
            key = (value_rate(duration_ns, value, age_price), len(members),
                   -sum(r.last_use for r in members), -batch, -width, -mask)
            if best_key is None or key > best_key:
                best, best_key = DecodeRound(batch, width, members, duration_ns, packed_kind, packed, drafts), key
        return best

    def packed_lengths(self, requests, width, capacity):
        """Predict causal lengths using completed confidence; current proposals remain unread."""
        limits = [self.speculative_tokens(r, width) for r in requests]
        lengths = [1] * len(requests)
        frontier = [(-r.acceptance.depths[0][0], 0, r.slot, row)
                    for row, (r, limit) in enumerate(zip(requests, limits)) if limit > 1]
        heapq.heapify(frontier)
        for _ in range(min(capacity - len(requests), sum(limits) - len(requests))):
            probability, depth, slot, row = heapq.heappop(frontier)
            lengths[row] += 1
            if lengths[row] < limits[row]:
                heapq.heappush(frontier, (probability * requests[row].acceptance.depths[depth + 1][0],
                                         depth + 1, slot, row))
        return lengths

    def packed_cohort(self, ready, batch, width, capacity, values, duration_ns, expected):
        """Price complete physical rounds using bounded causal forecasts and immutable cohorts."""
        if not ready:
            return None
        duration_ms = duration_ns / 1e6
        base = {r.slot: values[r.slot].decode(1., duration_ms) for r in ready}
        full = {r.slot: values[r.slot].decode(expected(r, min(self.speculative_tokens(r, width), capacity)),
                                             duration_ms) for r in ready}
        best = best_key = None
        for ordering in range(3):
            def priority(r):
                return (full[r.slot] if ordering == 0 else base[r.slot] if ordering == 1
                        else full[r.slot] / self.speculative_tokens(r, width))
            ordered = sorted(ready, key=lambda r: (-priority(r), r.slot))
            for count in range(1, min(batch, capacity, len(ordered)) + 1):
                members = sorted(ordered[:count], key=lambda r: r.slot)
                lengths = self.packed_lengths(members, width, capacity)
                value = sum(values[r.slot].decode(expected(r, length), duration_ms)
                            for r, length in zip(members, lengths))
                key = (value, count, -sum(1 << r.slot for r in members))
                if best_key is None or key > best_key:
                    best, best_key = (members, value), key
        return best
