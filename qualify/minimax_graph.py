"""Replay a selected MiniMax verification graph with real untimed draft setup."""

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory


def _snapshot(tensors):
    return [(value, value.clone()) for value in tensors]


def _restore(saved):
    for value, original in saved:
        value.copy_(original)


def _fixture(model, texts, history, chunk, seed):
    import torch
    from tokenizers import Tokenizer
    from ignition.components.sampling import RequestAdmission

    functions = next(value for value in model.resources if isinstance(value, dict) and "verification8x6" in value)
    verification = functions["verification8x6"]
    owner = verification.target.owner
    pool, requests, inputs = owner.pool, owner.requests, owner.inputs
    tokenizer = Tokenizer.from_file(str(Path(model.server_config["assets"]["tokenizer"]["path"])))
    pages_per_request = math.ceil((history + 6) / 128)
    prefill = functions[f"prefill{chunk}"]
    pool.clear_for_preparation()
    for value in verification.storage.buffers().values():
        value.zero_()
    for value in (owner.offset, owner.generation, owner.frame, owner.sequence, owner.records):
        value.zero_()
    for name in owner.rng.__dataclass_fields__:
        getattr(owner.rng, name).zero_()
    inputs.request_state_rows.zero_()
    inputs.page_tables.fill_(-1)
    for slot, text in enumerate(texts):
        tokens = tokenizer.encode(text).ids
        if not tokens:
            raise ValueError("each fixture text must tokenize to nonempty input")
        tokens = (tokens * math.ceil(history / len(tokens)))[:history]
        requests.admit(slot, RequestAdmission(generation=slot + 1, seed=seed + slot,
            max_new_tokens=history + 128, temperature=1., top_p=.95, top_k=-1), tokens)
        inputs.request_state_rows[slot, 0] = slot + 1
        inputs.page_tables[slot, :pages_per_request] = torch.arange(
            1 + slot * pages_per_request, 1 + (slot + 1) * pages_per_request, device=model.device)
        inputs.row_slot_ids.fill_(-1)
        inputs.row_slot_ids[0] = slot
        inputs.row_generations[0] = slot + 1
        inputs.live_rows.fill_(1)
        inputs.sample_enabled.zero_()
        for start in range(0, history, chunk):
            segment = tokens[start:start + chunk]
            inputs.query_lengths[0] = len(segment)
            inputs.token_ids[:len(segment)] = torch.tensor(segment, device=model.device)
            inputs.sample_enabled[0] = int(start + len(segment) == history)
            prefill()
        if requests.status[slot].item() != 1 or requests.consumed_tokens[slot].item() != history:
            raise RuntimeError(f"prefill slot {slot}: status={requests.status[slot].item()} "
                f"consumed={requests.consumed_tokens[slot].item()} error={requests.error_code[slot].item()}")
    inputs.row_slot_ids.copy_(torch.arange(8, device=model.device))
    inputs.row_generations.copy_(torch.arange(1, 9, device=model.device))
    inputs.live_rows.fill_(8)
    inputs.query_lengths.fill_(6)
    inputs.sample_enabled.fill_(1)
    inputs.token_ids[:8].copy_(requests.pending_token)
    if not (requests.status == 1).all().item() or not (requests.error_code == 0).all().item():
        raise RuntimeError("fixture admission or prefill left a request inactive or in error")
    if not (requests.consumed_tokens == history).all().item():
        raise RuntimeError("fixture prefill did not consume the complete history")
    if not (requests.generation == inputs.row_generations).all().item():
        raise RuntimeError("fixture request and packed row generations disagree")
    if not (requests.pending_token >= 0).all().item() or not (inputs.sample_enabled == 1).all().item():
        raise RuntimeError("fixture requires a sampled pending token for every verification row")
    owner.sequence.fill_(1)
    functions["proposal8x6"]()
    owner.sequence.fill_(2)
    torch.cuda.synchronize(model.device)
    if not verification.storage.ready.all().item():
        proposal = functions["proposal8x6"].source
        details = dict(ready=verification.storage.ready.cpu().tolist(),
            status=requests.status.cpu().tolist(), error=requests.error_code.cpu().tolist(),
            generations=requests.generation.cpu().tolist(),
            row_generations=inputs.row_generations.cpu().tolist(),
            pending=requests.pending_token.cpu().tolist(), tokens=inputs.token_ids[:8].cpu().tolist(),
            consumed=requests.consumed_tokens.cpu().tolist(),
            sequence=owner.sequence.cpu().tolist(), active=proposal.active.cpu().tolist(),
            before=proposal.before.cpu().tolist(),
            draft_context_consumed=pool.region("consumed_tokens")[:9].cpu().tolist())
        raise RuntimeError(f"actual DSpark proposal did not publish all eight request rows: {details}")
    tensors = [pool.backing[:int(pool.memory_capacity[0].item())]]
    tensors.extend(value[:int(pool.memory_capacity[1].item())] for value in pool.pages.values())
    tensors.extend(getattr(requests, name) for name in requests.__dataclass_fields__)
    tensors.extend(inputs.buffers().values())
    tensors.extend(verification.storage.buffers().values())
    tensors.extend((owner.frame, owner.sequence, owner.records, owner.offset, owner.generation))
    tensors.extend(getattr(owner.rng, name) for name in owner.rng.__dataclass_fields__)
    return verification, tensors



def _check_parity(model, forward, tensors, saved, *, graph=None):
    import torch

    graph = model.graphs["verification8x6"] if graph is None else graph
    _restore(saved)
    forward()
    torch.cuda.synchronize(model.device)
    expected = [value.clone() for value in tensors]
    expected_logits = forward.target.decoder.scratch.logits.clone()
    expected_active = forward.target.active.clone()
    expected_lengths = forward.target.metadata.lengths.clone()
    _restore(saved)
    graph.replay()
    torch.cuda.synchronize(model.device)
    for index, (value, reference) in enumerate(zip(tensors, expected, strict=True)):
        if not torch.equal(value.view(torch.uint8), reference.view(torch.uint8)):
            raise AssertionError(f"eager/captured mutable tensor differs: {index}")
    torch.testing.assert_close(forward.target.active, expected_active, rtol=0, atol=0)
    torch.testing.assert_close(forward.target.metadata.lengths, expected_lengths, rtol=0, atol=0)
    width = forward.target.width
    actual_logits = forward.target.decoder.scratch.logits
    if expected_active.all().item() and (expected_lengths == width).all().item():
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    else:
        offered = expected_active[:, None] & (torch.arange(width, device=model.device)[None, :] < expected_lengths[:, None])
        actual_logits = actual_logits.view(-1, width, expected_logits.shape[-1])
        torch.testing.assert_close(actual_logits[offered], expected_logits.view_as(actual_logits)[offered], rtol=0, atol=0)



def benchmark_replays(model, saved, *, warmup=10, repeats=50, graph=None):
    """Restore live state outside CUDA events and return rank-maximum replay milliseconds."""
    import torch
    import torch.distributed as dist

    graph = model.graphs["verification8x6"] if graph is None else graph
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    allocated_before = torch.cuda.memory_allocated(model.device)
    for iteration in range(warmup + repeats):
        _restore(saved)
        torch.cuda.synchronize(model.device)
        dist.barrier()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(start.elapsed_time(end), dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if iteration >= warmup:
            samples.append(elapsed.item())
    if torch.cuda.memory_allocated(model.device) != allocated_before:
        raise AssertionError("repeated replay changed allocated tensor bytes")
    return samples



def _rank(rank, args, rendezvous):
    import torch
    import torch.distributed as dist
    from ignition.preparation.package_storage import PackageStorage
    from ignition.recipes.minimax_m3.bootstrap import PreparationConfig, prepare_rank

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    storage = PackageStorage(rank=rank, world=4, device=rank)
    pages = 1 + 8 * math.ceil((args.history + 6) / 128)
    config = PreparationConfig(devices=(0, 1, 2, 3), compile_workers=args.compile_workers,
        context=args.context, max_active=8, expert_scale_compression=args.expert_scale_compression,
        prefill_chunk_size=args.prefill_chunk, vision=False, dspark=True,
        speculative_widths=(1, 6), speculative_token_capacity=48,
        speculative_fixed_width=6, retain_graphs=True, kernel_cache=args.kernel_cache,
        diagnostic_graphs=("verification8x6",),
        diagnostic_support_graphs=(f"prefill{args.prefill_chunk}", "proposal8x6"),
        diagnostic_cache_pages=max(pages, math.ceil(args.prefill_chunk / 128) + 16))
    with storage:
        model = prepare_rank(args.checkpoint, config, rank=rank, group=dist.group.WORLD,
            package_storage=storage, progress=lambda message: print(f"rank={rank} {message}", flush=True))
    try:
        texts = json.loads(args.texts.read_text())
        if not isinstance(texts, list) or len(texts) != 8 or len(set(texts)) != 8:
            raise ValueError("--texts must contain a JSON array of eight distinct actual texts")
        forward, tensors = _fixture(model, texts, args.history, args.prefill_chunk, args.seed)
        saved = _snapshot(tensors)
        graph = model.graphs["verification8x6"]
        _check_parity(model, forward, tensors, saved)
        if not forward.target.active.all().item() or not (forward.target.metadata.lengths == 6).all().item():
            raise AssertionError("primary verification must execute all 48 offered target positions")
        samples = benchmark_replays(model, saved, warmup=args.warmup, repeats=args.repeats)
        if (forward.target.owner.requests.error_code != 0).any().item():
            raise AssertionError("verification returned request errors")
        if rank == 0:
            print(json.dumps(dict(graph="verification8x6", context=args.context, history=args.history, concurrency=8,
                draft_tokens=5, target_rows=48, expert_scale_compression=args.expert_scale_compression,
                all_rank_max_ms=samples, parity="bitwise")), flush=True)
        if args.holdout_texts is not None:
            holdout = json.loads(args.holdout_texts.read_text())
            if len(holdout) != 8 or len(set(holdout)) != 8 or set(holdout) & set(texts):
                raise ValueError("holdout requires eight distinct texts separate from tuning texts")
            del saved
            forward, tensors = _fixture(model, holdout, args.history, args.prefill_chunk, args.seed + 10000)
            saved = _snapshot(tensors)
            _check_parity(model, forward, tensors, saved)
            _restore(saved)
            inputs = forward.target.owner.inputs
            inputs.live_rows.fill_(7)
            inputs.row_slot_ids[7] = -1
            inputs.query_lengths.copy_(torch.tensor([1, 2, 3, 4, 5, 6, 1, 0], device=rank))
            ragged = _snapshot(tensors)
            _check_parity(model, forward, tensors, ragged)
            if rank == 0:
                print("holdout and ragged/inactive replay parity passed", flush=True)
    finally:
        import sys

        pending_error = sys.exception()
        try:
            torch.cuda.synchronize(rank)
            dist.barrier()
            model.close()
            dist.destroy_process_group()
        except BaseException as cleanup_error:
            if pending_error is None:
                raise
            pending_error.add_note(f"rank cleanup could not drain: {cleanup_error}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--expert-scale-compression", choices=("csf",), default=None,
        help="Compress routed-expert scales using the selected checkpoint-preserving representation.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gpus", default="0-3")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--holdout-texts", type=Path)
    parser.add_argument("--context", type=int, default=1048576)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--prefill-chunk", type=int, default=4096)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--kernel-cache", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.history < 1 or args.prefill_chunk > args.history or args.history + 6 > args.context:
        parser.error("history must cover prefill chunk and verification must fit context")
    from ignition.cli.config import Configuration
    from ignition.cli.preparation import Host
    from ignition.cli.environment import apply
    from ignition.planning import parse_gpus

    configuration = Configuration.load(args.config)
    values = {key: dict(value) for key, value in configuration.values.items()}
    values.setdefault("paths", {})["kernel_cache"] = str(args.kernel_cache)
    configuration = Configuration(configuration.path, values, configuration.exists)
    host = Host.discover()
    indices = parse_gpus(args.gpus)
    gpus = tuple(gpu for index in indices for gpu in host.gpus if gpu.index == index)
    if len(gpus) != 4:
        parser.error("--gpus must select exactly four visible GPUs")
    apply(configuration, gpus)
    import torch.multiprocessing as mp
    with TemporaryDirectory(prefix="flare-graph-rendezvous-") as temporary:
        mp.spawn(_rank, args=(args, "file://" + temporary + "/group"), nprocs=4, join=True)
