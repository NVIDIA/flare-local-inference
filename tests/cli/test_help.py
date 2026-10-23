"""Recipe discovery and preparation syntax without GPU initialization."""

import subprocess
import sys
from pathlib import Path

import pytest

from ignition.cli import _parser, main
from ignition.errors import UsageError
from ignition.recipes import aliases, recipes, resolve


@pytest.mark.parametrize('flags', [['--history-drafts'], ['--no-history-drafts'], ['--mtp-proposal-source', 'copy']])
def test_export_rejects_history_preferences_and_copy_only_preparation(flags):
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(['export', 'qwen3.8-flash', *flags])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["export", "serve"])
def test_mla_arithmetic_flag_defaults_and_override(command):
    arguments = [command, "deepseek-v4.1-flash", *(["--output", "model.safetensors"] if command == "export" else [])]
    default = _parser().parse_args(arguments)
    explicit = _parser().parse_args([*arguments, "--mla-compute-mode", "bf16"])
    assert default.mla_compute_mode is None
    assert explicit.mla_compute_mode == "bf16"


@pytest.mark.parametrize("model", ["deepseek-v4.1-flash", "deepseek-v4-flash"])
@pytest.mark.parametrize("flags", [["--draft-head-quantization", "nvfp4"],
                                   ["--markov-head-quantization", "nvfp4"],
                                   ["--draft-head-quantization", "nvfp4", "--markov-head-quantization", "nvfp4"]])
def test_independent_draft_precision_flags_require_learned_proposals(model, flags):
    from ignition.cli.preparation import speculation

    args = _parser().parse_args(["export", model, "--output", "model.safetensors", *flags])
    result = speculation(args, resolve(model))
    assert bool(result.draft_head_quantization) == ("--draft-head-quantization" in flags)
    assert bool(result.markov_head_quantization) == ("--markov-head-quantization" in flags)
    args.no_mtp = True
    with pytest.raises(UsageError, match="disables speculation"):
        speculation(args, resolve(model))
    args.no_mtp = False
    args.mtp_draft_tokens = 0
    with pytest.raises(UsageError, match="positive learned draft depth"):
        speculation(args, resolve(model))


@pytest.mark.parametrize("command", [[], ["export"], ["serve"]])
def test_help_lists_every_recipe(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([*command, "--help"])
    assert error.value.code == 0
    captured = capsys.readouterr()
    output = captured.out
    assert not captured.err
    for alias in aliases():
        assert alias in output
    assert "Drafts (default/max)" in output
    assert "--packed-verification" not in output
    assert "--draft-prefix-selection" not in output


@pytest.mark.parametrize("command", ["export", "serve"])
def test_model_help_explains_step5_limits_and_compression(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([command, "step5-preview-nvfp4", "--help"])
    assert error.value.code == 0
    captured = capsys.readouterr()
    output = captured.out
    assert "--gpus: select 4, 8, 16 GPUs" in output
    assert "--mtp-draft-tokens: 0..3; default 3" in output
    assert "--max-concurrency: 1..64; recipe default 32" in output
    assert "--expert-scale-compression [MODE]" in output
    assert "estimated 6% less resident weight storage" in output
    assert "MODE: csf" in output
    assert "batch * width <= N" in output
    assert not captured.err


def test_help_for_all_aliases_and_repositories_needs_no_model_dependencies():
    script = """
import contextlib
import io
import sys
from ignition.cli import main
from ignition.recipes import recipes
with contextlib.redirect_stdout(io.StringIO()):
    for recipe in recipes():
        for checkpoint in recipe.checkpoints:
            for model in (checkpoint.alias, checkpoint.repository):
                try:
                    main(['export', model, '--help'])
                except SystemExit as error:
                    assert error.code == 0
assert not {'torch', 'transformers', 'flare_heliopause', 'huggingface_hub', 'pynvml'} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("command", ["export", "serve"])
def test_deepseek41_help_reports_packed_gpu_recipe_defaults(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([command, "deepseek-v4.1-flash", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--max-concurrency: 1..64; recipe default 32" in output
    assert "--prefill-chunk-size: 64..1048576, multiple of 64; default 4096" in output
    assert "--mtp-draft-tokens: 0..7; default 7" in output
    assert "--image-concurrency: 1..max concurrency; default 4" in output
    assert "--mla-compute-mode: auto, bf16; default bf16" in output
    assert "--expert-scale-compression: native, csf; default csf" in output
    assert "--speculative-token-capacity: at least drafts + 1; default 128" in output
    assert "--speculative-graph-budget: positive integer; default 64" in output
    assert "excludes mandatory width-one and ordinary graphs" in output
    assert "Prepared verification: packed rows with GPU prefix selection" in output
    assert "--packed-verification" not in output
    assert "--draft-prefix-selection" not in output
    assert "Target rows are shared across retained request prefixes" in output
    assert "Packed rounds need more target rows than active requests" in output


@pytest.mark.parametrize("command", ["export", "serve"])
def test_qwen_help_reports_concurrency_and_target_row_defaults(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([command, "qwen3.8-flash", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--max-concurrency: 1..64; recipe default 32" in output
    assert "--speculative-token-capacity: at least drafts + 1; default 128" in output
    assert "Prepared verification: rectangular rows" in output


def test_optional_features_follow_the_selected_recipe(capsys):
    for recipe in recipes():
        checkpoint = recipe.checkpoints[0]
        with pytest.raises(SystemExit):
            main(["export", checkpoint.alias, "--help"])
        details = capsys.readouterr().out.split("Optional features for this recipe:")[1]
        assert ("--mixed-prefill" in details) == recipe.mixed_prefill
        assert "--history-drafts" not in details
        assert ("--no-stream-overlap" in details) == recipe.stream_overlap
        assert ("--expert-scale-compression" in details) == bool(checkpoint.scale_compressed_device_bytes)


@pytest.mark.parametrize("command", ["export", "serve"])
def test_all_recipe_help_reports_concurrency_and_speculative_capacity_defaults(command, capsys):
    for recipe in recipes():
        for checkpoint in recipe.checkpoints:
            with pytest.raises(SystemExit) as error:
                main([command, checkpoint.alias, "--help"])
            assert error.value.code == 0
            output = capsys.readouterr().out
            assert f"--max-concurrency: 1..{recipe.max_concurrency}; recipe default {recipe.concurrency}" in output
            if recipe.speculation is None:
                assert "Multi-token speculation: unavailable" in output
            elif recipe.speculation.token_capacity is None:
                assert "--speculative-token-capacity: at least drafts + 1; default is the larger of" in output
                assert "power-of-two padded concurrency and drafts + 1." in output
            else:
                assert (f"--speculative-token-capacity: at least drafts + 1; default {recipe.speculation.token_capacity}"
                        in output)


@pytest.mark.parametrize("flag", [["--expert-scale-compression"], ["--expert-scale-compression", "csf"],
                                  ["--expert-scale-compression=csf"]])
def test_export_command_accepts_bare_and_explicit_compression(flag):
    args = _parser().parse_args([
        "export", "step5-preview-nvfp4", *flag, "--prefill-chunk-size", "4096",
        "--mtp-draft-tokens", "3", "--no-mtp-adaptive-verification",
        "--speculative-token-capacity", "16", "--max-concurrency", "16", "--output", "model.safetensors"])
    assert args.expert_scale_compression == "csf"
    assert (args.prefill_chunk_size, args.mtp_draft_tokens, args.speculative_token_capacity,
            args.max_concurrency) == (4096, 3, 16, 16)
    assert _parser().parse_args(["export", "step5-preview-nvfp4"]).expert_scale_compression is None


def test_invalid_compression_is_rejected_with_the_supported_encoding(capsys):
    with pytest.raises(SystemExit) as error:
        main(["export", "step5-preview-nvfp4", "--expert-scale-compression", "zip", "--output", "model"])
    assert error.value.code == 2
    assert "invalid choice: 'zip' (choose from csf, native)" in capsys.readouterr().err


def test_mistyped_alias_suggests_step5():
    with pytest.raises(UsageError, match="did you mean 'step5-preview-nvfp4'"):
        resolve("step4-preview-nvfp4")


@pytest.mark.parametrize("command", ["export", "serve"])
@pytest.mark.parametrize("effort", ["off", "low", "medium", "high", "max"])
def test_autotune_effort_flag(command, effort):
    args = [command, "deepseek-v4.1-flash"]
    assert _parser().parse_args(args).autotune_effort is None
    assert _parser().parse_args([*args, "--autotune-effort", effort]).autotune_effort == effort
    with pytest.raises(SystemExit):
        _parser().parse_args([*args, "--autotune-effort", "exhaustive"])
