"""Command output and exit codes with injected GPUs and hub cache."""

from pathlib import Path

import pytest

from ignition import planning
from ignition.cli import _parser, main, preparation
from ignition.errors import UsageError
from ignition.recipes import resolve

GIB = 1 << 30


def gpu(index, free_gib):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 * GIB, free_gib * GIB, (12, 0))


@pytest.fixture
def host(monkeypatch, tmp_path):
    """Give commands four free GPUs, an empty hub cache, and no configuration file."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus = [gpu(index, 95) for index in range(4)]
    monkeypatch.setattr(planning, "discover_gpus", lambda: tuple(gpus))
    return gpus


def configure(tmp_path, text):
    path = tmp_path / ".config/flare/config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def export(*arguments):
    return main(["export", *arguments])


def failure(capsys):
    lines = capsys.readouterr().err.splitlines()
    return lines[next(index for index, line in enumerate(lines) if line.startswith("error: ")):]


def test_export_uses_the_configured_model(host, tmp_path, capsys):
    assert export() == 2
    assert "no MODEL given and defaults.model is not set" in failure(capsys)[0]
    configure(tmp_path, '[defaults]\nmodel = "nope"\n')
    assert export() == 2
    assert "unknown model 'nope'" in failure(capsys)[0]


@pytest.mark.parametrize("arguments, code, message", [
    (["export", "nope"], 2, "supported models: qwen3.8-flash, qwen3.8-27b, glm-5.3-flash"),
    (["export", "qwen3.8-flash", "--gpus", "x"], 2, "--gpus expects indices"),
    (["export", "qwen3.8-flash", "--gpus", "9"], 2, "does not have"),
    (["export", "qwen3.8-flash", "--max-concurrency", "65"], 2, "1 through 64"),
    (["export", "glm-5.3-flash", "--gpus", "0-2"], 4, "needs 1 or 2 or 4 GPUs; 3 selected; use --gpus 0-3"),
    (["export", "glm-5.3-flash", "--mixed-prefill"], 2,
     "--mixed-prefill: glm-5.3-flash does not capture mixed prefill/decode graphs"),
])
def test_failures_use_the_documented_exit_codes(host, capsys, arguments, code, message):
    assert (export(*arguments[1:]) if arguments[0] == "export" else main(arguments)) == code
    assert message in failure(capsys)[0]


def test_capacity_errors_exit_four_with_aligned_continuation_lines(host, tmp_path, capsys):
    configure(tmp_path, '[memory]\nreserve = "18G"\n')
    assert export("qwen3.8-flash", "--gpus", "0") == 4
    lines = failure(capsys)
    assert lines[0].startswith("error: needs ")
    assert all(line.startswith("       ") for line in lines[1:])
    assert lines[-1].lstrip().startswith("Options: ")


def test_unknown_commands_flags_and_abbreviations_are_usage_errors(host, capsys):
    for arguments in (["export", "qwen3.8-flash", "--context", "4096"], ["export", "qwen3.8-flash", "--gpu", "0"],
                      ["export", "qwen3.8-flash", "--max", "2"], ["join", "10.0.0.1:29600"],
                      ["join", "10.0.0.1:29600", "--rank", "1", "--ranks", "17"], ["serve", "--ranks", "17"],
                      ["run", "qwen3.8-flash"], ["plan", "qwen3.8-flash"],
                      ["prepare", "qwen3.8-flash"], ["models"], ["pull"], ["status"], ["bench"], ["config"], ["skill"]):
        with pytest.raises(SystemExit) as exit:
            main(arguments)
        assert exit.value.code == 2
    capsys.readouterr()


def test_multi_host_ranks_parse_for_rank_zero_and_its_peers():
    rank_zero = _parser().parse_args(["serve", "glm-5.3-flash", "--ranks", "4", "--rendezvous", "10.0.0.1:29600"])
    assert (rank_zero.model, rank_zero.ranks, rank_zero.rendezvous) == ("glm-5.3-flash", 4, "10.0.0.1:29600")
    peer = _parser().parse_args(["join", "10.0.0.1:29600", "--rank", "3", "--ranks", "4"])
    assert (peer.rendezvous, peer.rank, peer.ranks, peer.gpu) == ("10.0.0.1:29600", 3, 4, 0)


def test_runtime_memory_flags_do_not_select_preparation_or_diagnostic_behavior(capsys):
    for command in (["export", "--output", "model.safetensors"], ["serve"]):
        for flags in (["--kv-cache-limit", "context"], ["--memory-limit", "30G"]):
            with pytest.raises(SystemExit) as error:
                main([*command, "qwen3.8-flash", *flags])
            assert error.value.code == 2
    capsys.readouterr()


def test_served_model_name_is_a_runtime_override(capsys):
    for command in (["export", "--output", "model.safetensors"], ["serve"]):
        with pytest.raises(SystemExit) as error:
            main([*command, "qwen3.8-flash", "--served-model-name", "local-model"])
        assert error.value.code == 2
    capsys.readouterr()


def test_stream_overlap_flag_is_a_preparation_flag_that_overlapping_recipes_accept(host, tmp_path, capsys):
    for command in (["export", "--output", str(tmp_path / "package")], ["serve"]):
        assert _parser().parse_args([*command, "glm-5.3-flash", "--no-stream-overlap"]).no_stream_overlap
        assert not _parser().parse_args([*command, "glm-5.3-flash"]).no_stream_overlap
    for model in ("qwen3.8-flash", "glm-5.3-flash", "deepseek-v4-flash", "deepseek-v4.1-flash"):
        serial = _parser().parse_args(["export", model, "--no-stream-overlap"])
        assert not preparation.stream_overlap(serial, resolve(model))
        assert preparation.stream_overlap(_parser().parse_args(["export", model]), resolve(model))
    # MiMo composes no concurrent branches, so serializing them is rejected rather than ignored.
    assert export("mimo-v2.6-pro-mopd", "--no-stream-overlap") == 2
    assert "--no-stream-overlap: mimo-v2.6-pro-mopd runs no branches concurrently" in failure(capsys)[0]


def test_direct_execution_is_an_explicit_serve_diagnostic(host, capsys):
    arguments = _parser().parse_args(["serve", "glm-5.3-flash", "--direct", "--eager"])
    assert arguments.direct and arguments.eager
    arguments = _parser().parse_args(["serve", "glm-5.3-flash"])
    assert not arguments.direct and not arguments.eager
    assert main(["serve", "glm-5.3-flash", "--eager"]) == 2
    assert "--eager applies only to --direct" in failure(capsys)[0]
    arguments = _parser().parse_args(["serve", "glm-5.3-flash", "--direct", "--graph-times", "times.json"])
    assert arguments.graph_times == Path("times.json")
    assert main(["serve", "glm-5.3-flash", "--graph-times", "times.json"]) == 2
    assert "--graph-times applies only to --direct" in failure(capsys)[0]
    assert main(["serve", "glm-5.3-flash", "--direct", "--eager", "--graph-times", "/nonexistent/times.json"]) == 2
    assert "graph replay with --graph-times saves it" in failure(capsys)[0]
    for arguments in (["export", "glm-5.3-flash", "--direct"], ["export", "glm-5.3-flash", "--output", "x", "--eager"],
                      ["join", "10.0.0.1:29600", "--rank", "1", "--ranks", "2", "--direct"]):
        with pytest.raises(SystemExit) as exit:
            main(arguments)
        assert exit.value.code == 2
    capsys.readouterr()


def speculation(*arguments, model="qwen3.8-flash"):
    """Resolve the speculation settings `export` would prepare for a command line."""
    return preparation.speculation(_parser().parse_args(["export", model, *arguments]), resolve(model))


def test_speculation_flags_override_the_recipe():
    default = speculation()
    assert default == resolve("qwen3.8-flash").recipe.speculation
    assert (default.draft_tokens, default.adaptive, default.widths, default.fixed_width) == (
        5, True, (1, 2, 3, 4, 5, 6), None)
    assert (default.proposal_source, default.history_drafts) == ("mtp", False)
    assert speculation("--no-mtp") is None
    # Verification adapts among every count up to the draft-token setting unless it is fixed.
    reduced = speculation("--mtp-draft-tokens", "3")
    assert (reduced.draft_tokens, reduced.adaptive, reduced.widths, reduced.fixed_width) == (
        3, True, (1, 2, 3, 4), None)
    fixed = speculation("--mtp-draft-tokens", "5", "--no-mtp-adaptive-verification")
    assert (fixed.draft_tokens, fixed.adaptive, fixed.fixed_width) == (5, False, 6)
    assert speculation("--no-mtp-adaptive-verification").fixed_width == 6
    assert speculation("--mtp-draft-tokens", "0").widths == (1,)
    assert speculation("--mtp-draft-tokens", "7").draft_tokens == 7
    copied = speculation("--mtp-proposal-source", "copy")
    assert copied.proposal_source == "copy" and not copied.learned and not copied.history_drafts
    assert speculation("--history-drafts").history_drafts
    assert speculation(model="glm-5.3-flash").draft_tokens == 5
    assert speculation("--no-mtp", model="glm-5.3-flash") is None


def test_deepseek_defaults_to_five_adaptive_drafts_and_accepts_explicit_overrides():
    default = speculation(model="deepseek-v4.1-flash")
    assert (default.draft_tokens, default.adaptive, default.fixed_width) == (5, True, None)
    assert default.widths == tuple(range(1, 7))
    reduced = speculation("--mtp-draft-tokens", "3", model="deepseek-v4.1-flash")
    assert (reduced.draft_tokens, reduced.adaptive, reduced.fixed_width) == (3, True, None)
    assert reduced.packed_verification and reduced.draft_prefix_selection == "gpu"
    with pytest.raises(UsageError, match="requires adaptive verification"):
        speculation("--mtp-draft-tokens", "3", "--no-mtp-adaptive-verification",
                    model="deepseek-v4.1-flash")
    extended = speculation("--mtp-draft-tokens", "7", model="deepseek-v4.1-flash")
    assert extended.draft_tokens == 7 and extended.widths == tuple(range(1, 9))
    assert speculation("--mtp-draft-tokens", "0", model="deepseek-v4.1-flash").widths == (1,)
    assert speculation("--no-mtp", model="deepseek-v4.1-flash") is None


@pytest.mark.parametrize("arguments, message", [
    (["--no-mtp", "--mtp-draft-tokens", "2"], "--no-mtp disables speculation; --mtp-draft-tokens cannot"),
    (["--no-mtp", "--no-mtp-adaptive-verification"],
     "--no-mtp disables speculation; --no-mtp-adaptive-verification cannot accompany it"),
    (["--no-mtp", "--history-drafts"], "--no-mtp disables speculation; --history-drafts cannot accompany it"),
    (["--mtp-draft-tokens", "8"], "--mtp-draft-tokens must be 0 through 7; --no-mtp disables speculation"),
    (["--mtp-draft-tokens", "-1"], "--mtp-draft-tokens must be 0 through 7"),
])
def test_invalid_speculation_flags_are_usage_errors(host, capsys, arguments, message):
    assert export("qwen3.8-flash", *arguments) == 2
    assert message in failure(capsys)[0]


def test_speculation_flags_have_one_spelling(host, capsys):
    for arguments in (["--mtp-draft-tokens", "1,3"], ["--mtp-max-draft-tokens", "5"],
                      ["--mtp-fixed-draft-tokens", "2"], ["--mtp-draft-tokens", "x"]):
        with pytest.raises(SystemExit) as exit:
            main(["export", "qwen3.8-flash", *arguments])
        assert exit.value.code == 2
    capsys.readouterr()


def test_history_copy_flags_apply_only_to_recipes_that_support_them(host, capsys):
    assert export("glm-5.3-flash", "--mtp-proposal-source", "copy") == 2
    assert "glm-5.3-flash drafts only with its MTP module; --mtp-proposal-source copy does not apply" in (
        failure(capsys)[0])
    assert export("glm-5.3-flash", "--history-drafts") == 2
    assert "--history-drafts does not apply" in failure(capsys)[0]
    with pytest.raises(SystemExit):
        main(["export", "qwen3.8-flash", "--mtp-proposal-source", "ngram"])
    capsys.readouterr()


def test_fetch_downloads_the_pinned_revision_into_the_hub_cache(host, tmp_path, monkeypatch):
    import huggingface_hub
    from ignition.errors import FetchError

    calls = []

    def download(repository, *, revision, cache_dir):
        calls.append((repository, revision, cache_dir))
        return "/snapshots/x"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    model = resolve("glm-5.3-flash")
    cache = tmp_path / ".cache/huggingface/hub"
    assert preparation.fetch(model, cache) == "/snapshots/x"
    assert calls == [("nvidia/GLM-5.3-Flash-FP4-Local",
                      "76348ddb2be2e35e39ecfc00e75dc07703f4f3d5", tmp_path / ".cache/huggingface/hub")]

    def fail(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    with pytest.raises(FetchError, match="cannot download.*network unreachable"):
        preparation.fetch(model, cache)


def test_deepseek_v4_vision_uses_only_checkpoint_integrated_dspark():
    from ignition.errors import UsageError

    default = speculation(model="deepseek-v4-flash")
    assert default.draft_tokens == 5 and default.adaptive
    assert default.proposal_source == "mtp" and default.history_drafts is False
    assert speculation("--mtp-draft-tokens", "0", model="deepseek-v4-flash").widths == (1,)
    assert speculation("--no-mtp", model="deepseek-v4-flash") is None
    for flags in (("--mtp-draft-tokens", "6"), ("--mtp-proposal-source", "copy"), ("--history-drafts",)):
        with pytest.raises(UsageError):
            speculation(*flags, model="deepseek-v4-flash")


def test_roce_ring_requires_multiple_hosts_before_model_or_gpu_work(tmp_path, capsys):
    for arguments in (["export", "--output", str(tmp_path / "model")], ["serve"]):
        assert main([*arguments, "qwen3.8-flash", "--roce-ring"]) == 2
        assert "--roce-ring requires at least two hosts" in failure(capsys)[0]
    args = _parser().parse_args(["export", "qwen3.8-flash", "--output", str(tmp_path / "model"),
                               "--roce-ring", "--ranks", "4"])
    assert args.roce_ring
    from ignition.cli.serve import _checkpoint_fabric
    assert _checkpoint_fabric(args)["streams"] == 8
    for count in (1, 4, 8, 16, 32, 64):
        args = _parser().parse_args(["export", "glm-5.3-flash", "--roce-ring", "--ranks", "4",
                                    "--transfer-streams", str(count)])
        assert _checkpoint_fabric(args)["streams"] == count
    for count in (0, 65):
        with pytest.raises(SystemExit):
            _parser().parse_args(["export", "--transfer-streams", str(count)])
    args = _parser().parse_args(["export", "--transfer-streams", "8"])
    from ignition.errors import UsageError
    with pytest.raises(UsageError, match="requires multiple host ranks"):
        _checkpoint_fabric(args)


@pytest.mark.parametrize("ring", [False, True])
def test_fabric_streams_do_not_require_orbit_configuration(ring):
    from ignition.cli.serve import _checkpoint_fabric

    args = _parser().parse_args(["export", "glm-5.3-flash", "--ranks", "4", "--transfer-streams", "16",
                                *(["--roce-ring"] if ring else [])])
    assert _checkpoint_fabric(args) == dict(ring=ring, chain=False, streams=16)


def test_fabric_transfers_follow_the_chain_shape():
    from ignition.cli.serve import _checkpoint_fabric

    args = _parser().parse_args(["export", "glm-5.3-flash", "--ranks", "4", "--roce-chain"])
    assert _checkpoint_fabric(args) == dict(ring=False, chain=True, streams=8)
