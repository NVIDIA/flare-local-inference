"""Packed graph-count budgets and their independent preparation settings."""

from dataclasses import replace

import pytest

from ignition.cli import _parser, preparation
from ignition.cli.config import Configuration
from ignition.errors import UsageError
from ignition.recipes import resolve
from ignition.recipes.declaration import Speculation


def selected(*flags, alias="deepseek-v4.1-flash", configuration=None):
    args = _parser().parse_args(["export", alias, *flags])
    return preparation.speculation(args, resolve(alias), configuration)


@pytest.mark.parametrize("command", ["export", "serve"])
def test_graph_budget_flag_parses_independently_of_token_capacity(command):
    args = _parser().parse_args([command, "deepseek-v4.1-flash", "--speculative-graph-budget", "48",
                                 "--speculative-token-capacity", "96"])
    assert (args.speculative_graph_budget, args.speculative_token_capacity) == (48, 96)
    assert _parser().parse_args([command]).speculative_graph_budget is None


def test_graph_budget_precedence_and_token_capacity_are_independent(tmp_path):
    path = tmp_path / "flare.toml"
    default = resolve("deepseek-v4.1-flash").recipe.speculation
    assert default.graph_budget == 64
    assert selected() == default
    path.write_text('[defaults]\nspeculative_graph_budget=48\nspeculative_token_capacity=96\n')
    config = Configuration.load(path)
    assert selected(configuration=config) == replace(default, graph_budget=48, token_capacity=96)
    path.write_text(path.read_text() + '[models."deepseek-v4.1-flash"]\nspeculative_graph_budget=32\n')
    config = Configuration.load(path)
    assert selected(configuration=config) == replace(default, graph_budget=32, token_capacity=96)
    assert selected("--speculative-graph-budget", "24", configuration=config) == replace(
        default, graph_budget=24, token_capacity=96)
    assert selected("--speculative-token-capacity", "192", configuration=config) == replace(
        default, graph_budget=32, token_capacity=192)


@pytest.mark.parametrize("alias", ["qwen3.8-flash", "glm-5.3-flash", "step5-preview-nvfp4", "qwen3.8-27b"])
def test_explicit_graph_budget_rejects_recipes_without_packed_rounds(alias):
    with pytest.raises(UsageError, match="--speculative-graph-budget"):
        selected("--speculative-graph-budget", "32", alias=alias)


@pytest.mark.parametrize("flags,message", [
    (("--no-mtp",), "cannot accompany"),
    (("--mtp-draft-tokens", "0"), "positive draft depth"),
    (("--no-mtp-adaptive-verification",), "requires adaptive verification"),
])
def test_explicit_graph_budget_rejects_incompatible_modes(flags, message):
    with pytest.raises(UsageError, match=message):
        selected(*flags, "--speculative-graph-budget", "32")


@pytest.mark.parametrize("budget", ["0", "-1"])
def test_graph_budget_requires_a_positive_integer(budget):
    with pytest.raises(UsageError, match="positive integer"):
        selected("--speculative-graph-budget", budget)


def test_global_graph_budget_is_scoped_to_active_packed_speculation(tmp_path):
    path = tmp_path / "flare.toml"
    path.write_text('[defaults]\nspeculative_graph_budget=48\n')
    config = Configuration.load(path)
    assert selected(configuration=config).graph_budget == 48
    assert selected("--mtp-draft-tokens", "0", configuration=config).graph_budget is None
    assert selected("--no-mtp", configuration=config) is None
    for alias in ("qwen3.8-flash", "qwen3.8-27b"):
        assert selected(alias=alias, configuration=config) == resolve(alias).recipe.speculation


@pytest.mark.parametrize("alias,flags", [
    ("qwen3.8-flash", ()), ("qwen3.8-27b", ()),
    ("deepseek-v4.1-flash", ("--no-mtp",)),
    ("deepseek-v4.1-flash", ("--mtp-draft-tokens", "0")),
])
def test_model_graph_budget_is_an_explicit_selection(tmp_path, alias, flags):
    path = tmp_path / "flare.toml"
    path.write_text(f'[models."{alias}"]\nspeculative_graph_budget=32\n')
    with pytest.raises(UsageError, match="--speculative-graph-budget"):
        selected(*flags, alias=alias, configuration=Configuration.load(path))


def test_plan_serializes_graph_budget_only_for_applicable_speculation():
    from tests.cli.test_planning import derive

    packed = selected("--speculative-graph-budget", "48")
    plan = derive(speculation=packed)
    assert plan.to_dict()["speculation"]["graph_budget"] == 48
    assert plan.speculative_token_capacity == 128
    assert "graph_budget" not in derive().to_dict()["speculation"]
    assert derive(speculation=None).to_dict()["speculation"] is None


def test_planning_resolves_missing_graph_budget_from_recipe_without_overriding_explicit_values():
    from tests.cli.test_planning import SOLO, derive

    default = selected()
    recipe = replace(SOLO, speculation=default)
    assert derive(model=recipe, speculation=replace(default, graph_budget=None)).speculation.graph_budget == 64
    assert derive(model=recipe, speculation=replace(default, graph_budget=32)).speculation.graph_budget == 32
    assert derive(model=recipe, speculation=Speculation(0)).speculation.graph_budget is None


def test_planning_rejects_graph_budget_below_required_endpoints():
    from tests.cli.test_planning import derive

    settings = selected("--speculative-graph-budget", "11")
    with pytest.raises(UsageError, match="--speculative-graph-budget: .*at least 12"):
        derive(concurrency=32, speculation=settings)
    assert derive(concurrency=32, speculation=replace(settings, graph_budget=12)).speculation.graph_budget == 12


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_speculation_declaration_validates_graph_budget_type_and_range(budget):
    with pytest.raises(ValueError, match="positive integer"):
        Speculation(5, packed_verification=True, graph_budget=budget)


@pytest.mark.parametrize("options", [
    {"packed_verification": False}, {"adaptive": False}, {"draft_tokens": 0},
    {"proposal_source": "copy"}, {"history_drafts": True},
])
def test_speculation_declaration_requires_packed_adaptive_learned_rounds(options):
    values = {"draft_tokens": 5, "packed_verification": True, "graph_budget": 64, **options}
    with pytest.raises(ValueError, match="requires packed adaptive learned speculation"):
        Speculation(**values)
