"""Prepare a selected graph catalog and report complete GPU round latency."""

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from ignition.errors import FlareError, PreparationError, UsageError

from . import environment
from .config import Configuration
from .preparation import Host, context_parallel, derive, fetch, resolve_model, speculation, stream_overlap


def _integers(text):
    try:
        values = tuple(int(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected distinct comma-separated integers")
    return values


def arguments(parser):
    options = parser.add_argument_group("GPU graph latency benchmark",
        "Qwen3.8-27B DFlash2 only. Measures complete prepared speculative graph rounds; "
        "excludes preparation, context prefill, state restoration, scheduling, and HTTP.")
    options.add_argument("--batches", type=_integers, default=(1, 2, 4), metavar="LIST",
                         help="request batches selected from 1,2,4, comma-separated (default: 1,2,4)")
    options.add_argument("--contexts", type=_integers, default=(512, 2048, 8192, 16384), metavar="LIST",
                         help="consumed context tokens, positive multiples of 128 (default: 512,2048,8192,16384)")
    options.add_argument("--verify-width", type=int, metavar="N",
                         help="target tokens per request including pending token; this benchmark requires 8")
    options.add_argument("--warmup", type=int, default=5, metavar="N",
                         help="positive untimed restored rounds per case and seed (default: 5)")
    options.add_argument("--repetitions", type=int, default=20, metavar="N",
                         help="timed restored rounds per case and seed (default: 20)")
    options.add_argument("--seeds", type=_integers, default=(0, 1, 2), metavar="LIST",
                         help="independent input and sampling seeds, comma-separated (default: 0,1,2)")
    options.add_argument("--output", type=Path, metavar="PATH",
                         help="write JSON configuration, per-case statistics, and every raw latency sample")


@dataclass(frozen=True)
class Settings:
    batches: tuple[int, ...]
    contexts: tuple[int, ...]
    width: int
    warmup: int
    repetitions: int
    seeds: tuple[int, ...]

    @classmethod
    def from_args(cls, args):
        width = args.verify_width
        if width is None:
            width = 8 if args.mtp_draft_tokens is None else args.mtp_draft_tokens + 1
        if args.mtp_draft_tokens is not None and args.mtp_draft_tokens + 1 != width:
            raise UsageError("--verify-width must equal --mtp-draft-tokens + 1")
        if width != 8:
            raise UsageError("ignition bench requires --verify-width 8 and --mtp-draft-tokens 7")
        if any(value not in (1, 2, 4) for value in args.batches):
            raise UsageError("--batches must select request counts from 1,2,4")
        if any(value < 128 or value % 128 for value in args.contexts):
            raise UsageError("--contexts must contain positive multiples of 128")
        if args.warmup < 1:
            raise UsageError("--warmup must be positive")
        if args.repetitions < 1:
            raise UsageError("--repetitions must be positive")
        if any(not 0 <= seed < 2**63 for seed in args.seeds):
            raise UsageError("--seeds must contain integers from 0 through 2^63 - 1")
        if args.no_mtp:
            raise UsageError("ignition bench requires DFlash2 state; --no-mtp does not apply")
        if args.mixed_prefill or args.prompt_logprobs:
            raise UsageError("ignition bench measures speculative rounds; --mixed-prefill and --prompt-logprobs do not apply")
        if args.official_source is not None:
            raise UsageError("--official-source applies to package publishing; ignition bench exports no package")
        return cls(tuple(sorted(args.batches)), tuple(sorted(args.contexts)), width,
                   args.warmup, args.repetitions, args.seeds)


def table(result):
    gpu = result["device"]
    lines = [f"GPU graph round latency · {gpu['name']} · SM{gpu['capability'][0]}{gpu['capability'][1]}",
             " B   W    P   Context    Median ms       p25–p75 ms"]
    for case in result["cases"]:
        lines.append(f"{case['batch']:2d}  {case['width']:2d}  {case['packed_tokens']:3d}  "
                     f"{case['context_tokens']:8d}  {case['median_ms']:11.4f}  "
                     f"{case['p25_ms']:8.4f}–{case['p75_ms']:.4f}")
    lines.append(f"Equal-weight mean of case medians: {result['aggregate_ms']:.4f} ms (lower is better)")
    lines.append("GPU class scores are separate. This measures graph latency, not serving throughput.")
    return "\n".join(lines)


def run(args):
    settings = Settings.from_args(args)
    configuration = Configuration.load(args.config)
    model = resolve_model(configuration, args.model or configuration.model or "qwen3.8-27b")
    if model.recipe.name != "qwen38_dense":
        raise UsageError("ignition bench supports only the Qwen3.8-27B DFlash2 recipe")
    from ignition.preparation.selection import BenchmarkSelection

    selection = BenchmarkSelection(batches=settings.batches, contexts=settings.contexts, width=settings.width)
    concurrency = configuration.max_concurrency(model.checkpoint.alias, max(settings.batches), args.max_concurrency)
    if concurrency.value < max(settings.batches):
        raise UsageError("--max-concurrency must cover every selected --batches value")
    args = argparse.Namespace(**vars(args))
    args.mtp_draft_tokens = settings.width - 1
    plan = derive(configuration, model, Host.discover(), gpus=args.gpus, concurrency=concurrency,
        speculation=speculation(args, model, configuration), prefill_chunk=args.prefill_chunk_size,
        context_parallel=context_parallel(args, model), stream_overlap=stream_overlap(args, model),
        compile_workers=args.compile_workers, tuning_gpus=args.tuning_gpus,
        autotune_effort=args.autotune_effort, collective_tuning=args.collective_tuning, collective_ladder=args.collective_ladder,
        lock_gpu_clocks=args.lock_gpu_clocks, native_dense_activations=args.native_dense_activations, packed_dense_w4a16=args.packed_dense_w4a16,
        shard_hyperconnections=args.shard_hyperconnections, joint_tuning=args.joint_tuning,
        image_concurrency=args.image_concurrency, mixed_prefill=args.mixed_prefill,
        prompt_logprobs=args.prompt_logprobs, expert_scale_compression=args.expert_scale_compression,
        online_mxfp8=args.online_mxfp8, language_model_only=args.language_model_only,
        mla_compute_mode=args.mla_compute_mode)
    plan = replace(plan, benchmark_selection=selection)
    print(json.dumps({"plan": plan.to_dict()}), file=sys.stderr, flush=True)
    progress = _progress(args.verbose)
    hub_cache = configuration.directory("hub_cache")
    if not model.cached(hub_cache):
        if model.directory is not None:
            raise UsageError(f"{model.directory} lacks tensor files named by its safetensors index")
        from ignition.progress import phase

        with phase(progress, "Download checkpoint"):
            fetch(model, hub_cache, progress_bars=False)
            model.verify(hub_cache)
    environment.apply(configuration, plan.gpus, plan.recipe.b12x_environment)
    result = _measure(model, plan, configuration, settings, progress, compile_forwards=not args.no_compile, output=args.output)
    write_result(result, args.output)
    return 0


def write_result(result, output):
    print(table(result), flush=True)
    save_result(result, output)


def save_result(result, output):
    if output is not None:
        try:
            output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        except OSError as error:
            raise UsageError(f"cannot write benchmark JSON to {output}: {error}") from error


def _progress(verbose):
    from ignition.progress import StartupProgress

    class Progress(StartupProgress):
        def __call__(self, message):
            super().__call__(message)
            if verbose:
                self.console.print(f"  {message}", style="dim", highlight=False)

    return Progress()


def _measure(model, plan, configuration, settings, progress, *, compile_forwards, output=None):
    import torch
    from tokenizers import Tokenizer
    from ignition.bench.runner import BenchmarkCase, run_benchmark
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tuning_replicas import prepare_recipe

    snapshot = model.snapshot(configuration.directory("hub_cache"))
    details = {}
    gpu = plan.gpus[0]
    details["device"] = {"name": gpu.name, "uuid": gpu.uuid, "capability": list(gpu.capability),
                        "total_bytes": gpu.total_bytes}
    preparation = plan.to_dict()
    details["configuration"] = {"model": model.model_id, "checkpoint_directory": str(snapshot),
        "repository": model.checkpoint.repository, "revision": model.checkpoint.revision,
        "compiled_forwards": compile_forwards, "torch_version": torch.__version__,
        "requested_max_concurrency": plan.concurrency, "prepared_request_capacity": max(settings.batches),
        "preparation": {key: preparation[key] for key in (
            "tensor_parallel", "prefill_chunk", "stream_overlap", "context_parallel",
            "native_dense_activations", "packed_dense_w4a16", "autotune_effort", "collective_tuning", "collective_ladder", "compile_workers",
            "speculative_token_capacity", "speculation", "expert_scale_compression", "mla_compute_mode",
            "online_mxfp8", "language_model_only")}}

    def record(result):
        result.update(details)
        save_result(result, output)
        if result["status"] == "running":
            case = result["cases"][-1]
            print(f"Measured B={case['batch']} W={case['width']} P={case['packed_tokens']} "
                  f"context={case['context_tokens']} seed={case['samples'][-1]['seed']} "
                  f"({len(case['samples'])}/{len(settings.seeds)} seeds)", file=sys.stderr, flush=True)

    torch.cuda.set_device(0)
    storage = PackageStorage(world=plan.world_size)
    try:
        with storage:
            prepared = prepare_recipe(model.recipe.prepare, snapshot, plan, compile_forwards=compile_forwards,
                kernel_cache=configuration.directory("kernel_cache"), progress=progress, package_storage=storage)
        with prepared:
            tokenizer = Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))
            scenarios = tuple(BenchmarkCase(batch=batch, width=settings.width, packed_tokens=batch * settings.width,
                context_tokens=context) for batch in settings.batches for context in settings.contexts)
            result = run_benchmark(prepared, tokenizer, scenarios=scenarios, warmup=settings.warmup,
                repetitions=settings.repetitions, seeds=settings.seeds, progress=record)
    except FlareError:
        raise
    except Exception as error:
        raise PreparationError(f"GPU graph benchmark failed: {type(error).__name__}: {error}") from error
    finally:
        torch.cuda.synchronize()
        for region in storage.host_regions:
            region.close()
    result.update(details)
    return result
