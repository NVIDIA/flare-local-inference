"""Recipe presets and retained CLI controls for packed GPU prefix selection."""

from dataclasses import replace

import pytest

from ignition.cli import _parser, preparation, display
from ignition.cli.config import Configuration
from ignition.errors import UsageError
from ignition.recipes import resolve
from ignition.recipes.declaration import Speculation


DEFAULT = Speculation(5, token_capacity=128, packed_verification=True, draft_prefix_selection="gpu", graph_budget=64)


def settings(*flags, alias="deepseek-v4.1-flash", configuration=None):
    args = _parser().parse_args(["export", alias, *flags])
    return preparation.speculation(args, resolve(alias), configuration)


def test_deepseek41_uses_packed_gpu_prefix_selection_without_flags():
    assert settings() == DEFAULT
    assert settings("--mtp-draft-tokens", "3") == replace(DEFAULT, draft_tokens=3)


def test_layout_selection_is_not_a_cli_argument():
    args = _parser().parse_args(["export", "deepseek-v4.1-flash"])
    assert not hasattr(args, "packed_verification")
    assert not hasattr(args, "draft_prefix_selection")


@pytest.mark.parametrize("command", ("export", "serve"))
@pytest.mark.parametrize("flags", [
    ("--packed-verification",), ("--packed-verification=true",),
    ("--draft-prefix-selection", "cpu"), ("--draft-prefix-selection=gpu",),
])
def test_removed_layout_flags_are_unrecognized(command, flags, capsys):
    with pytest.raises(SystemExit) as error:
        _parser().parse_args([command, "deepseek-v4.1-flash", *flags])
    assert error.value.code == 2
    assert "unrecognized arguments:" in capsys.readouterr().err


@pytest.mark.parametrize("alias", ("qwen3.8-flash", "glm-5.3-flash", "step5-preview-nvfp4"))
def test_other_recipes_preserve_cpu_and_rectangular_defaults(alias):
    model = resolve(alias)
    assert settings(alias=alias) == model.recipe.speculation
    assert settings(alias=alias).draft_prefix_selection == "cpu"
    assert not settings(alias=alias).packed_verification
    assert not settings("--no-mtp-adaptive-verification", alias=alias).adaptive


def test_gpu_prefix_selection_is_printed():
    assert "GPU prefix selection" in display.speculation(settings())
    assert "GPU prefix selection" in display.speculation(settings(), brief=True)


@pytest.mark.parametrize("flags,message", [
    (("--no-mtp-adaptive-verification",), "requires adaptive verification"),
    (("--mtp-proposal-source", "copy"), "drafts only with its MTP module"),
    (("--history-drafts",), "drafts only with its MTP module"),
])
def test_incompatible_selection_fails_before_preparation(flags, message):
    with pytest.raises(UsageError, match=message):
        settings(*flags)


def test_width_one_and_disabled_speculation_preserve_their_meaning():
    assert settings("--no-mtp") is None
    assert settings("--mtp-draft-tokens", "0") == Speculation(0, token_capacity=128)
    assert settings("--mtp-draft-tokens", "0", "--no-mtp-adaptive-verification") == Speculation(
        0, adaptive=False, token_capacity=128)


@pytest.mark.parametrize("flags", [
    ("--mtp-draft-tokens", "0"), ("--speculative-token-capacity", "128"),
    ("--no-mtp-adaptive-verification",), ("--history-drafts",),
])
def test_no_mtp_rejects_explicit_speculation_flags(flags):
    with pytest.raises(UsageError, match="cannot accompany"):
        settings("--no-mtp", *flags)


def test_capacity_configuration_and_explicit_flags_override_recipe_defaults(tmp_path):
    path = tmp_path / "flare.toml"
    path.write_text('[defaults]\nspeculative_token_capacity=64\n'
                    '[models."deepseek-v4.1-flash"]\nspeculative_token_capacity=96\n')
    configuration = Configuration.load(path)
    assert settings(configuration=configuration) == replace(DEFAULT, token_capacity=96)
    assert settings("--speculative-token-capacity", "192", configuration=configuration) == replace(
        DEFAULT, token_capacity=192)
    path.write_text("")
    assert settings(configuration=Configuration.load(path)) == DEFAULT


def test_plan_and_rebuild_metadata_record_prefix_selection():
    from types import SimpleNamespace
    from tests.cli.test_planning import derive
    from ignition.preparation import rebuild_manifest, runtime_package_pb2 as p

    plan = derive(speculation=settings())
    assert plan.speculative_token_capacity == 128
    assert plan.to_dict()["speculation"]["draft_prefix_selection"] == "gpu"
    encoded = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
    assert encoded["speculation"]["draft_prefix_selection"] == "gpu"
    assert encoded["speculation"]["token_capacity"] == 128
    legacy = p.RebuildSpeculation.FromString(b"\x08\x05\x10\x01\x1a\x03mtp\x30\x01")
    assert legacy.draft_prefix_selection == ""
    assert legacy.draft_tokens == 5 and legacy.packed_verification
