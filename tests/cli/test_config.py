# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Configuration file validation and precedence."""

import pytest

from ignition.cli.config import Configuration, Setting, parse_size
from ignition.errors import UsageError


def load(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return Configuration.load(path)


def test_absent_default_file_yields_built_in_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    configuration = Configuration.load()
    assert not configuration.exists and configuration.path == tmp_path / ".config/flare/config.toml"
    assert configuration.get("server.host") == Setting("127.0.0.1", "default")
    assert configuration.get("server.port") == Setting(8000, "default")
    assert configuration.reserve_bytes == 0 and configuration.model is None
    assert configuration.directory("hub_cache") == tmp_path / ".cache/huggingface/hub"
    assert configuration.directory("kernel_cache") == tmp_path / ".cache/flare/kernels"
    assert configuration.directory("cuda") is None


def test_an_explicit_missing_file_is_a_usage_error(tmp_path):
    with pytest.raises(UsageError, match="does not exist"):
        Configuration.load(tmp_path / "absent.toml")


def test_flags_override_the_file_which_overrides_the_recipe(tmp_path):
    configuration = load(tmp_path, """
[server]
host = "0.0.0.0"
port = 9000

[defaults]
model = "qwen3.8-flash"
max_concurrency = 6
speculative_token_capacity = 32

[memory]
reserve = "16G"

[models."qwen3.8-flash"]
max_concurrency = 2
speculative_token_capacity = 8
""")
    assert configuration.get("server.host") == Setting("0.0.0.0", "[server]")
    assert configuration.get("server.port") == Setting(9000, "[server]")
    assert configuration.get("server.port", 7000) == Setting(7000, "flag")
    assert configuration.model == "qwen3.8-flash" and configuration.reserve_bytes == 16 << 30

    assert configuration.max_concurrency("qwen3.8-flash", 4) == Setting(2, '[models."qwen3.8-flash"]')
    assert configuration.max_concurrency("glm-5.3-flash", 4) == Setting(6, "[defaults]")
    assert configuration.max_concurrency("qwen3.8-flash", 4, 9) == Setting(9, "flag")
    assert configuration.model_setting("qwen3.8-flash", "speculative_token_capacity", None).value == 8
    assert configuration.model_setting("glm-5.3-flash", "speculative_token_capacity", None).value == 32
    assert configuration.model_setting("qwen3.8-flash", "speculative_token_capacity", None, 16).value == 16


def test_prefix_cache_settings_validate(tmp_path):
    configuration = load(tmp_path, "[prefix_cache]\nmemory = \"4G\"\n")
    assert configuration.get("prefix_cache.memory") == Setting("4G", "[prefix_cache]")
    assert configuration.directory("prefix_cache") is None
    with pytest.raises(UsageError, match="prefix_cache disk_limit"):
        load(tmp_path, "[prefix_cache]\ndisk_limit = \"lots\"\n")
    for transport in ("nvme", "staged", "direct"):
        with pytest.raises(UsageError, match="disk_transport"):
            load(tmp_path, f'[prefix_cache]\ndisk_transport = "{transport}"\n')


def test_recipe_supplies_concurrency_when_the_file_is_silent(tmp_path):
    configuration = load(tmp_path, "[server]\nport = 8001\n")
    assert configuration.max_concurrency("qwen3.8-flash", 4) == Setting(4, "recipe")


def test_image_concurrency_configuration_is_scoped_to_vision_and_flags_win(tmp_path):
    from types import SimpleNamespace
    from ignition.cli.preparation import image_capacity_setting

    config = load(tmp_path, '[defaults]\nimage_concurrency = 4\n[models."vision"]\nimage_concurrency = 2\n')
    def model(alias, images):
        return SimpleNamespace(checkpoint=SimpleNamespace(alias=alias), recipe=SimpleNamespace(max_images=images))
    assert image_capacity_setting(config, model("vision", 4)) == 2
    assert image_capacity_setting(config, model("vision", 4), 1) == 1
    assert image_capacity_setting(config, model("another-vision", 4)) == 4
    assert image_capacity_setting(config, model("text", 0)) is None
    assert image_capacity_setting(config, model("text", 0), 1) == 1
    with pytest.raises(UsageError, match="at least 1"):
        load(tmp_path, '[defaults]\nimage_concurrency = 0\n')


def test_compiler_worker_flag_overrides_configured_host_budget(tmp_path):
    from ignition.cli.preparation import compiler_workers

    config = load(tmp_path, "[preparation]\ncompile_workers = 24\n")
    assert compiler_workers(config) == 24
    assert compiler_workers(config, 32) == 32
    assert compiler_workers(load(tmp_path, "")) is None
    for value in (0, -1):
        with pytest.raises(UsageError, match="positive integer"):
            compiler_workers(config, value)
    with pytest.raises(UsageError, match="at least 1"):
        load(tmp_path, "[preparation]\ncompile_workers = 0\n")


def test_compile_headroom_flag_overrides_configured_size(tmp_path):
    from ignition.cli.preparation import compiler_headroom
    from ignition.preparation.host_memory import COMPILE_HEADROOM_BYTES

    assert compiler_headroom(load(tmp_path, "")) == COMPILE_HEADROOM_BYTES == 8 << 30
    config = load(tmp_path, '[preparation]\ncompile_headroom = "6G"\n')
    assert compiler_headroom(config) == 6 << 30
    assert compiler_headroom(config, "12GiB") == 12 << 30
    assert compiler_headroom(config, "0") == 0
    for value in ("-1G", "", "lots", "6X"):
        with pytest.raises(UsageError, match="--compile-headroom"):
            compiler_headroom(config, value)
    for text, message in (('"-6G"', "compile_headroom in \\[preparation\\]"), ('""', "compile_headroom"),
                          ("6", "must be str")):
        with pytest.raises(UsageError, match=message):
            load(tmp_path, f"[preparation]\ncompile_headroom = {text}\n")


def test_initial_file_documents_the_compile_headroom(tmp_path):
    configuration = Configuration(tmp_path / "config.toml", {}, False)
    configuration.write_initial("qwen3.8-flash", 4)
    lines = configuration.path.read_text().splitlines()
    example = next(line for line in lines if line.startswith("# compile_headroom"))
    written = load(tmp_path, "[preparation]\n" + example.removeprefix("# ") + "\n")
    assert written.get("preparation.compile_headroom") == Setting("8G", "[preparation]")


@pytest.mark.parametrize("text, message", [
    ("[memory]\nfraction = 0.9\n", "unknown key fraction"),
    ("[engine]\npages = 4\n", "unknown section"),
    ("[server]\nport = \"8000\"\n", "port in .* must be int"),
    ("[server]\nport = 70000\n", "1 through 65535"),
    ("[defaults]\nmax_concurrency = true\n", "must be int"),
    ("[defaults]\nmax_concurrency = 0\n", "at least 1"),
    ("[defaults]\nspeculative_token_capacity = 0\n", "at least 1"),
    ("[defaults]\nspeculative_graph_budget = 0\n", "at least 1"),
    ("[defaults]\nspeculative_graph_budget = true\n", "must be int"),
    ('[models."deepseek-v4.1-flash"]\nspeculative_graph_budget = -1\n', "at least 1"),
    ("[memory]\nconversation_cache = 16\n", "unknown key conversation_cache"),
    ("[memory]\nreserve = \"lots\"\n", "memory size"),
    ("[memory]\nreserve = 16\n", "must be str"),
    ("[models.\"qwen3.8-flash\"]\ncontext = 4096\n", "unknown key context"),
    ("server = 3\n", "must be a table"),
    ("[server\n", "config.toml"),
])
def test_invalid_files_are_usage_errors(tmp_path, text, message):
    with pytest.raises(UsageError, match=message):
        load(tmp_path, text)


@pytest.mark.parametrize("text, expected", [
    ("0", 0), ("4096", 4096), ("16G", 16 << 30), ("16g", 16 << 30), ("1.5G", 3 << 29), ("512M", 512 << 20),
    ("8 GiB", 8 << 30), ("2T", 2 << 40), ("64K", 64 << 10),
])
def test_sizes_use_binary_units(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "G", "-1G", "16X", "1e3"])
def test_malformed_sizes_are_usage_errors(text):
    with pytest.raises(UsageError):
        parse_size(text)


def test_shard_size_is_not_a_publishing_setting(tmp_path):
    with pytest.raises(UsageError, match='unknown key.*shard_size'):
        load(tmp_path, '[publishing]\nshard_size = "2G"\n')


@pytest.mark.parametrize("effort", ["off", "default", "audit"])
def test_autotune_effort_configuration_and_flag_precedence(tmp_path, effort):
    from ignition.cli.preparation import tuning_effort

    assert tuning_effort(load(tmp_path, "")) == "default"
    config = load(tmp_path, f'[preparation]\nautotune_effort = "{effort}"\n')
    assert tuning_effort(config) == effort
    assert tuning_effort(config, "off") == "off"
    assert tuning_effort(config, "audit") == "audit"
    for legacy in ("exhaustive", "medium", "max"):
        with pytest.raises(UsageError, match="autotune_effort"):
            load(tmp_path, f'[preparation]\nautotune_effort = "{legacy}"\n')


@pytest.mark.parametrize("recipe_mode", ["static", "runtime"])
@pytest.mark.parametrize("configured", [None, "static", "runtime"])
@pytest.mark.parametrize("flag", [None, "static", "runtime"])
def test_collective_tuning_resolves_flag_configuration_and_recipe(tmp_path, recipe_mode, configured, flag):
    from types import SimpleNamespace
    from ignition.cli.preparation import collective_tuning_mode

    config = load(tmp_path, "" if configured is None else f'[preparation]\ncollective_tuning = "{configured}"\n')
    model = SimpleNamespace(recipe=SimpleNamespace(collective_tuning=recipe_mode))
    assert collective_tuning_mode(config, model, flag) == (flag or configured or recipe_mode)


def test_collective_tuning_rejects_unknown_policy(tmp_path):
    with pytest.raises(UsageError, match="collective_tuning must be static or runtime"):
        load(tmp_path, '[preparation]\ncollective_tuning = "dynamic"')


@pytest.mark.parametrize("value", ["on", "off", "0", "90s", "1.5m", "20ms", "2h", "0.001"])
def test_shutdown_drain_configuration(tmp_path, value):
    config = load(tmp_path, f'[prefix_cache]\ndrain_on_shutdown = "{value}"\n')
    assert config.get("prefix_cache.drain_on_shutdown").value == value
    assert config.get("prefix_cache.drain_on_shutdown", "off").value == "off"


@pytest.mark.parametrize("value", ["-1", "NaN", "inf", "1e30h", "later", "0s"])
def test_invalid_shutdown_drain_configuration(tmp_path, value):
    with pytest.raises(UsageError, match="drain_on_shutdown"):
        load(tmp_path, f'[prefix_cache]\ndrain_on_shutdown = "{value}"\n')
