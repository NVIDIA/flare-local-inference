"""Recipe declarations: qualified checkpoints, serving ranges, and memory costs.

Declarations import no model code, so planning and model listing run without
PyTorch.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
import math
from types import MappingProxyType

from ignition.components.capacity import request_capacity

@dataclass(frozen=True)
class SamplingRange:
    """Inclusive bounds from the model author's sampling recommendations."""

    min: float | int
    max: float | int


@dataclass(frozen=True)
class SamplingDefaults:
    """Complete serving parameters manually verified against the model card."""

    temperature: float
    top_p: float
    top_k: int
    seed: int = 0
    min_p: float = 0.
    repetition_penalty: float = 1.
    presence_penalty: float = 0.
    frequency_penalty: float = 0.
    ranges: Mapping[str, SamplingRange] = field(default_factory=dict)

    def __post_init__(self):
        values = (self.temperature, self.top_p, self.min_p, self.repetition_penalty,
                  self.presence_penalty, self.frequency_penalty)
        if (any(type(v) not in (int, float) or not math.isfinite(v)
                or abs(v) > float.fromhex("0x1.fffffep+127") for v in values)
                or any(v < float.fromhex("0x1p-126") for v in
                       (self.temperature, self.top_p, self.repetition_penalty))
                or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1
                or type(self.top_k) is not int or not -1 <= self.top_k < 2**31
                or type(self.seed) is not int or not 0 <= self.seed < 2**64):
            raise ValueError("invalid recipe sampling defaults")
        names = {item.name for item in fields(self)} - {"seed", "ranges"}
        for name, bounds in self.ranges.items():
            if (name not in names or not isinstance(bounds, SamplingRange)
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           for v in (bounds.min, bounds.max))
                    or not bounds.min <= getattr(self, name) <= bounds.max):
                raise ValueError(f"invalid sampling range for {name}")
            for value in (bounds.min, bounds.max):
                replace(self, **{name: value}, ranges={})
        object.__setattr__(self, "ranges", MappingProxyType(dict(self.ranges)))

    def as_dict(self):
        result = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "ranges"}
        if self.ranges:
            result["ranges"] = {name: asdict(bounds) for name, bounds in self.ranges.items()}
        return result


@dataclass(frozen=True)
class Checkpoint:
    """One qualified checkpoint revision and where its stored tensors reside."""

    alias: str
    repository: str
    revision: str
    architecture: str
    stored_bytes: int
    device_bytes: int
    mapped_host_bytes: int = 0
    # The share of `device_bytes` that holds the draft model, loaded only for learned drafts.
    draft_bytes: int = 0
    # Device bytes that loading adds beyond the stored tensors, such as block scales
    # expanded from a compressed encoding.
    expansion_bytes: int = 0
    # Estimated resident device weights after optional lossless expert-scale compression.
    scale_compressed_device_bytes: Mapping[str, int] = field(default_factory=dict)
    # Per-rank costs replacing the recipe's `ranks` and `plain_ranks` when this checkpoint's draft or state
    # layout differs, such as a draft model that only this checkpoint carries.
    ranks: Mapping[int, "RankCosts"] | None = None
    # Bytes of weights one decode step reads for one request across every rank: dense weights
    # plus the experts one token activates, at their stored precision. Zero leaves the serving
    # contract without the value, and dashboards then show no weight bandwidth.
    decode_weight_bytes: int = 0
    plain_ranks: Mapping[int, "RankCosts"] | None = None


@dataclass(frozen=True)
class Speculation:
    """Multi-token speculation: how many draft tokens a round verifies and where drafts come from.

    A round verifies the pending input and up to `draft_tokens` drafts.
    Preparation captures width one at every request bucket and wider widths
    within the target-token capacity. `adaptive` sets the serving default: choose
    widths from measured costs and conditional acceptance estimates when true, or use the fixed maximum when
    false. Flare can reduce both capacities when loading the package. Drafts
    come from the checkpoint's MTP module or, with the `copy` source, from
    continuations from shared decoded history; `history_drafts` adds those
    continuations to MTP drafts.
    """

    draft_tokens: int
    adaptive: bool = True
    proposal_source: str = "mtp"
    history_drafts: bool = False
    token_capacity: int | None = None
    packed_verification: bool = False
    draft_prefix_selection: str = "cpu"
    graph_budget: int | None = None

    def __post_init__(self):
        if self.graph_budget is not None:
            if type(self.graph_budget) is not int or self.graph_budget < 1:
                raise ValueError("speculative graph budget must be a positive integer")
            if (not self.packed_verification or not self.adaptive or self.draft_tokens < 1
                    or self.proposal_source != "mtp" or self.history_drafts):
                raise ValueError("speculative graph budget requires packed adaptive learned speculation")
        if self.draft_prefix_selection not in ("cpu", "gpu"):
            raise ValueError("draft prefix selection must be cpu or gpu")
        if self.draft_prefix_selection == "gpu" and (
                not self.packed_verification or not self.adaptive or self.draft_tokens < 1
                or self.proposal_source != "mtp" or self.history_drafts):
            raise ValueError("GPU draft prefix selection requires packed adaptive learned speculation")

    @property
    def learned(self) -> bool:
        return self.proposal_source == "mtp"

    @property
    def widths(self) -> tuple[int, ...]:
        """Verification widths: each draft count from zero through `draft_tokens`, plus the pending input."""
        return tuple(range(1, self.draft_tokens + 2))

    @property
    def fixed_width(self) -> int | None:
        return None if self.adaptive else self.draft_tokens + 1


@dataclass(frozen=True)
class RankCosts:
    """Device bytes one tensor-parallel rank spends per unit of capacity."""

    state_row_bytes: int
    page_bytes: int
    graph_bytes: int
    # The share of `page_bytes` that stores the draft model's pages, present only with learned drafts.
    draft_page_bytes: int = 0
    # Global positions per page when ranks share each page's context; the recipe's page size otherwise.
    page_tokens: int | None = None


@dataclass(frozen=True)
class RankUsage:
    """What one rank allocated and left free after preparation."""

    pages: int
    free_bytes: int


@dataclass(frozen=True)
class Recipe:
    """One model family's qualified checkpoints, serving range, costs, and bootstrap.

    `prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress)` loads,
    tunes, and captures the plan's serving catalog and returns the prepared model.
    `usage(prepared)` reports each rank's device memory in rank order.
    `served_model_name` is the canonical model name advertised by the API,
    independent of checkpoint selection aliases and storage formats.

    `ranks` holds each tensor-parallel size's costs as the recipe serves by
    default. A recipe that declares `speculation` drafts with those settings
    unless a plan overrides them, and `plain_ranks` holds its costs with
    speculation disabled. Each active request has one execution state row per
    selected verification position unless `verification_rows` gives another count
    of extra rows. `history_copy` states that drafts can be copied from shared decoded history.
    `stream_overlap` states that captured graphs run independent branches on an
    auxiliary stream, which a plan can serialize onto the main stream.
    `online_mxfp8` states that the recipe names BF16 projections a plan may store as
    MXFP8 while loading, trading checkpoint precision for decode bandwidth.
    `language_model_only` states that a plan may prepare the recipe without its vision
    encoder, rejecting images and leaving the encoder's memory to KV pages.
    `mla_compute_modes` declares the attention arithmetic constraints a plan may select,
    independently of KV storage encoding and kernel tuning.
    `export_repository` names the Hugging Face destination and the default local snapshot
    namespace. Export writes locally; it does not upload or select a signing identity.
    `b12x_environment` holds the b12x settings the recipe prepares with. b12x reads them
    only from its environment, so Ignition sets them for the preparing process.
    """

    name: str
    served_model_name: str
    checkpoints: tuple[Checkpoint, ...]
    context: int
    page_tokens: int
    concurrency: int
    max_concurrency: int
    max_draft_tokens: int
    prefill_chunk: int
    ranks: Mapping[int, RankCosts]
    prepare: Callable
    usage: Callable
    sampling_defaults: SamplingDefaults
    mapped_host_name: str = ""
    max_images: int = 0
    image_bytes: int = 0
    speculation: Speculation | None = None
    plain_ranks: Mapping[int, RankCosts] = field(default_factory=lambda: MappingProxyType({}))
    history_copy: bool = False
    verification_rows: int | None = None
    stream_overlap: bool = False
    mixed_prefill: bool = False
    online_mxfp8: bool = False
    language_model_only: bool = False
    b12x_environment: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    mla_compute_modes: tuple[str, ...] = ("auto",)
    export_repository: str = ""

    def mixed_shapes(self, concurrency, chunk, speculation):
        from ignition.components.capacity import mixed_shapes

        if not self.mixed_prefill:
            return ()
        modes = (True, False) if speculation is not None else (False,)
        return tuple((enabled, b, w, p) for enabled in modes
            for b, w, p in mixed_shapes(concurrency, chunk, speculation.widths if enabled else (1,),
                                        speculation.token_capacity if enabled else None))

    def __post_init__(self):
        if self.export_repository:
            import re
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*', self.export_repository) or '..' in self.export_repository:
                raise ValueError('export_repository must be a Hugging Face namespace/repository')
        if any(not name.startswith("B12X_") or not isinstance(value, str)
               for name, value in self.b12x_environment.items()):
            raise ValueError("b12x settings name B12X_ variables with string values")
        if self.speculation is not None and set(self.plain_ranks) != set(self.ranks):
            raise ValueError("a recipe that speculates declares its costs without speculation for every rank count")
        for checkpoint in self.checkpoints:
            if any(costs is not None and set(costs) != set(self.ranks)
                   for costs in (checkpoint.ranks, checkpoint.plain_ranks)):
                raise ValueError(f"{checkpoint.alias} declares costs for other tensor-parallel sizes than its recipe")

    @property
    def tensor_parallel(self) -> tuple[int, ...]:
        return tuple(sorted(self.ranks))

    def costs(self, world_size: int, speculation: Speculation | None, checkpoint: Checkpoint | None = None) -> RankCosts:
        """Return one rank's costs under the given speculation settings, for `checkpoint` when it declares its own."""
        ranks = self.ranks if checkpoint is None or checkpoint.ranks is None else checkpoint.ranks
        plain = self.plain_ranks if checkpoint is None or checkpoint.plain_ranks is None else checkpoint.plain_ranks
        if speculation is None:
            return (plain or ranks)[world_size]
        costs = ranks[world_size]
        return costs if speculation.learned else replace(
            costs, page_bytes=costs.page_bytes - costs.draft_page_bytes, draft_page_bytes=0)

    def weights_bytes(self, checkpoint: Checkpoint, world_size: int, speculation: Speculation | None,
                      expert_scale_compression: str | None = None) -> int:
        """Estimate one rank's device weights from the checkpoint's stored tensors and their expansion."""
        learned = speculation is not None and speculation.learned
        device_bytes = (checkpoint.device_bytes if expert_scale_compression is None else
                        checkpoint.scale_compressed_device_bytes[expert_scale_compression])
        return -(-(device_bytes + checkpoint.expansion_bytes
                   - (0 if learned else checkpoint.draft_bytes)) // world_size)

    def live_rows(self, concurrency: int, speculation: Speculation | None) -> int:
        """Permanently backed null, live, and speculative execution rows."""
        slots = request_capacity(concurrency)
        speculative = (0 if speculation is None else speculation.draft_tokens if self.verification_rows is None
                       else self.verification_rows)
        return 1 + slots * (1 + speculative)
