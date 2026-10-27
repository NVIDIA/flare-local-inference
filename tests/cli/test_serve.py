"""Package export and development serving with injected preparation resources."""

import json
import os
import socket
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from ignition import planning
from ignition.cli import environment, main, serve
from ignition.cli.config import Configuration
from ignition.recipes import ResolvedModel, recipes, resolve
from ignition.recipes.declaration import RankUsage

GIB = 1 << 30
FREE = 95 * GIB


class Prepared:
    def __init__(self, plan, *, extra=0, pages=None):
        self.plan, self.extra, self.pages = plan, extra, pages
        self.closed = False
        self.started = None
        self.max_active = plan.concurrency
        self.server_config = {"snapshots": plan.snapshots, "pages": pages or plan.pages,
            "model_contract": {"graphs": [{"state_rows_per_request": plan.live_rows // plan.concurrency}]}}
        self.buffers = {"memory_capacity": types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(
            tolist=lambda: [plan.live_rows + plan.snapshots + 1, pages or plan.pages]))}

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.closed = True

    def usage(self):
        plan = self.plan
        state = plan.live_rows * plan.costs.state_row_bytes
        held = plan.weights_bytes + plan.graph_bytes + self.extra + state + plan.page_bytes
        return (RankUsage(self.pages or plan.pages, FREE - held),)

    def start_server(self, **kwargs):
        raise AssertionError("serving must start in the native executable")


@pytest.fixture
def host(monkeypatch, tmp_path):
    from ignition.preparation import module_images, package_storage, model_package

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus = tuple(planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 * GIB, FREE, (12, 0))
                 for index in range(2))
    monkeypatch.setattr(planning, "discover_gpus", lambda: gpus)
    monkeypatch.setattr(environment, "apply", lambda configuration, selected, b12x=(): calls.update(gpus=selected, b12x=b12x))
    monkeypatch.setattr(ResolvedModel, "cached", lambda self, hub_cache: True)
    calls = {"behavior": {}, "exports": 0, "signals": types.SimpleNamespace(entered=0, exited=0)}

    class CliSignals:
        def __enter__(self):
            calls["signals"].entered += 1
        def __exit__(self, *exception):
            calls["signals"].exited += 1

    class Scope:
        def __init__(self, **kwargs):
            self.host_regions = []
        def __enter__(self):
            return self
        def __exit__(self, *exception):
            pass

    class OutOfMemoryError(RuntimeError):
        pass

    import flare_heliopause
    monkeypatch.setitem(sys.modules, "flare_heliopause", types.SimpleNamespace(
        CliSignals=CliSignals, Server=object,
        export_source_identity=flare_heliopause.export_source_identity,
        export_reuse_weights=flare_heliopause.export_reuse_weights,
        export_development_publisher=flare_heliopause.export_development_publisher))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(set_device=lambda _: None), OutOfMemoryError=OutOfMemoryError))
    monkeypatch.setattr(module_images, "ModuleImages", Scope)
    monkeypatch.setattr(package_storage, "PackageStorage", Scope)

    def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False, package_storage):
        assert isinstance(package_storage, Scope)
        calls.update(snapshot=snapshot, plan=plan, compile_forwards=compile_forwards,
                     kernel_cache=kernel_cache, storage=package_storage, retain_graphs=retain_graphs)
        behavior = dict(calls["behavior"])
        if "raises" in behavior:
            raise behavior["raises"]
        calls["prepared"] = Prepared(plan, **behavior)
        return calls["prepared"]

    model = resolve("qwen3.8-flash")
    snapshot = model.snapshot(Configuration.load().directory("hub_cache"))
    snapshot.mkdir(parents=True)
    header = json.dumps({"weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
    header += b" " * (-len(header) % 8)
    (snapshot / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0")
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 1}, "weight_map": {"weight": "model.safetensors"}}))
    recipe = replace(model.recipe, prepare=prepare, usage=lambda prepared: prepared.usage())
    monkeypatch.setattr(serve, "resolve_model", lambda configuration, name: ResolvedModel(recipe, model.checkpoint))

    def export_model(prepared, images, storage, path, *, options, publisher, progress, console, staging):
        assert prepared is calls["prepared"] and not prepared.closed
        assert isinstance(images, Scope) and isinstance(storage, Scope)
        assert "api_key" not in options
        calls["publisher"] = publisher
        calls["options"] = options
        calls["exports"] += 1
        from ignition.preparation.package_snapshot import _OPTIONS
        assert path is None
        calls["cache"] = _OPTIONS.get().cache
        calls["entry_path"] = calls["cache"] / "models" / "qwen38" / f"gb202-tp1-test-{calls['exports']}"
        return types.SimpleNamespace(entry_path=calls["entry_path"])
    monkeypatch.setattr(model_package, "export_package", export_model)

    return calls


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def export(*arguments):
    return main(["export", *arguments])


def test_export_records_rebuild_settings_and_closes_preparation(host):
    assert export("qwen3.8-flash", "--no-compile") == 0
    assert host["compile_forwards"] is False and host["exports"] == 1
    assert host["retain_graphs"]
    assert [gpu.index for gpu in host["gpus"]] == [0]
    assert host["plan"].speculation == host["plan"].recipe.speculation
    assert not host["plan"].mixed_prefill and host["plan"].mixed_shapes == ()
    assert host["prepared"].closed and host["prepared"].started is None
    rebuild = types.SimpleNamespace(**host['options']['rebuild_manifest'])
    assert rebuild.version == 1 and rebuild.recipe == host['plan'].recipe.name
    assert rebuild.checkpoint_revision == host['plan'].checkpoint.revision
    assert rebuild.tensor_parallel == 1 and len(rebuild.source_identity) == 64
    assert rebuild.prefill_chunk == host['plan'].prefill_chunk
    assert rebuild.speculation['draft_tokens'] == host['plan'].speculation.draft_tokens
    assert 'history_drafts' not in host['options']
    assert 'history_drafts' not in rebuild.speculation
    from google.protobuf.json_format import ParseDict
    from ignition.preparation.runtime_package_pb2 import RebuildManifest
    encoded = ParseDict(host['options']['rebuild_manifest'], RebuildManifest()).SerializeToString()
    decoded = RebuildManifest.FromString(encoded)
    assert 'history_drafts' not in {field.name for field, _ in decoded.speculation.ListFields()}
    assert not rebuild.compiled_forwards and not rebuild.local_checkpoint
    from ignition.preparation.rebuild_manifest import from_plan
    local = ResolvedModel(host['plan'].recipe, host['plan'].checkpoint, Path('/local/checkpoint'))
    local_manifest = types.SimpleNamespace(**from_plan(local, host['plan'], compiled_forwards=True))
    assert local_manifest.local_checkpoint and not local_manifest.checkpoint_revision


@pytest.mark.parametrize(("alias", "served_name"), [
    ("qwen3.8-flash", "Qwen3.8-Flash-Next"),
    ("qwen3.8-27b", "Qwen3.8-27B"),
    ("glm-5.3-flash", "GLM-5.3-Flash"),
    ("deepseek-v4.1-flash", "DeepSeek-V4.1-Flash"),
    ("deepseek-v4-flash", "DeepSeek-V4-Flash-Vision-Exp"),
    ("mimo-v2.6-pro-mopd", "MiMo-V2.6-Pro-MOPD"),
    ("kimi-k3-mxfp4-csf", "Kimi-K3"),
    ("kimi-k3", "Kimi-K3"),
    ("kimi-k3-dspark", "Kimi-K3"),
    ("step5-preview-nvfp4", "Step-5-Preview"),
    ("nemotron-bolt-24gb", "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"),
])
@pytest.mark.parametrize("local", [False, True], ids=["checkpoint", "local-directory"])
def test_server_options_advertise_canonical_names_independently_of_checkpoint_source(
        tmp_path, alias, served_name, local):
    recipe, checkpoint = next((recipe, checkpoint) for recipe in recipes()
                              for checkpoint in recipe.checkpoints if checkpoint.alias == alias)
    directory = tmp_path / "custom-checkpoint" if local else None
    model = ResolvedModel(recipe, checkpoint, directory)
    plan = types.SimpleNamespace(recipe=recipe, checkpoint=checkpoint, reserve_bytes=0,
                                 concurrency=recipe.concurrency, max_images=recipe.max_images)
    config = tmp_path / "config.toml"
    config.write_text("")
    options = serve._server_options(Configuration.load(config), model, plan)
    assert options["served_names"] == [served_name]
    assert options["model_id"] == ("custom-checkpoint" if local else checkpoint.repository)


@pytest.mark.parametrize("extension", [None, types.SimpleNamespace()])
def test_diagnostic_serving_requires_native_http_before_preparation(host, monkeypatch, capsys, extension):
    monkeypatch.setitem(sys.modules, "flare_heliopause", extension)
    assert main(["debug-serve", "qwen3.8-flash", "--gpus", "0", "--port", str(free_port())]) == 2
    assert "qualification Cargo feature" in capsys.readouterr().err
    assert "prepared" not in host and "gpus" not in host


def test_direct_serving_does_not_require_native_http(host, monkeypatch):
    monkeypatch.setitem(sys.modules, "flare_heliopause", types.SimpleNamespace())
    discovered = []

    def discover():
        discovered.append(True)
        raise RuntimeError("GPU discovery reached")

    monkeypatch.setattr(serve.Host, "discover", discover)
    with pytest.raises(RuntimeError, match="GPU discovery reached"):
        main(["debug-serve", "qwen3.8-flash", "--direct", "--gpus", "0",
              "--port", str(free_port())])
    assert discovered == [True] and "prepared" not in host


def test_diagnostic_serving_retains_storage_until_native_shutdown(host, monkeypatch):
    closed = []

    def serve_prepared(prepared, bind, options):
        assert prepared is host["prepared"] and not prepared.closed
        assert host["exports"] == 0

        def close_region():
            assert prepared.closed
            closed.append(True)

        host["storage"].host_regions.append(types.SimpleNamespace(close=close_region))
        with prepared:
            return 0

    monkeypatch.setattr(serve, "_serve_prepared", serve_prepared)
    assert main(["debug-serve", "qwen3.8-flash", "--gpus", "0",
                 "--port", str(free_port())]) == 0
    assert host["exports"] == 0 and closed == [True]
    assert not host["retain_graphs"]


def test_diagnostic_startup_failure_retains_undrained_host_storage(host, monkeypatch):
    from ignition.preparation.model import PreparedModel

    original = RuntimeError("native startup retained resources")
    original.resources_retained = True
    released = []

    def start_server(self, **options):
        host["storage"].host_regions.append(types.SimpleNamespace(close=lambda: released.append(True)))
        raise original

    def close(self):
        raise RuntimeError("native drain was not confirmed")

    monkeypatch.setattr(Prepared, "memory_report", {}, raising=False)
    monkeypatch.setattr(Prepared, "start_server", start_server)
    monkeypatch.setattr(Prepared, "close", close, raising=False)
    monkeypatch.setattr(Prepared, "__exit__", PreparedModel.__exit__)
    with pytest.raises(RuntimeError, match="native startup retained resources") as caught:
        main(["debug-serve", "qwen3.8-flash", "--gpus", "0", "--port", str(free_port())])
    assert caught.value is original and "native drain was not confirmed" in original.__notes__[0]
    assert released == [] and not host["prepared"].closed


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("requests,width,snapshots,backed,expected", [
    (64, 6, 512, 449, 64), (1, 6, 512, 8, 1), (4, 1, 16, 21, 16), (3, 6, 512, 33, 8),
    (4, 1, 32, 9, 4),
])
def test_diagnostic_admission_uses_backed_rows_without_changing_export(
        grouped, requests, width, snapshots, backed, expected):
    config = {"snapshots": snapshots, "pages": 1000,
              "model_contract": {"graphs": [{"state_rows_per_request": 1},
                                            {"state_rows_per_request": width}]}}
    model = types.SimpleNamespace(max_active=requests, server_config=config,
        buffers={"memory_capacity": types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(
            tolist=lambda: [backed, 160]))})
    prepared = types.SimpleNamespace(model=model) if grouped else model
    options = {"bind": "0.0.0.0:8001"}
    limited = serve._diagnostic_options(prepared, options)
    assert limited == {**options, "snapshots": expected, "pages": 160}
    assert requests * width + limited["snapshots"] + 1 <= backed
    assert config["snapshots"] == snapshots and config["pages"] == 1000
    assert options == {"bind": "0.0.0.0:8001"}


def test_diagnostic_admission_rejects_insufficient_execution_backing():
    model = types.SimpleNamespace(max_active=4,
        server_config={"model_contract": {"graphs": [{"state_rows_per_request": 4}]}},
        buffers={"memory_capacity": types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(
            tolist=lambda: [16, 160]))})
    with pytest.raises(serve.PreparationError, match="cannot hold"):
        serve._diagnostic_options(model, {})


def test_preparation_flags_reach_the_exported_plan(host):
    assert export("qwen3.8-flash", "--max-concurrency", "2", "--prefill-chunk-size", "1024",
               "--mtp-draft-tokens", "2", "--no-mtp-adaptive-verification") == 0
    plan = host["plan"]
    assert (plan.concurrency, plan.prefill_chunk) == (2, 1024)
    assert (plan.speculation.draft_tokens, plan.speculation.adaptive) == (2, False)
    assert export("qwen3.8-flash", "--no-mtp") == 0
    assert host["exports"] == 2 and host["plan"].speculation is None


@pytest.mark.parametrize('flags,enabled', [
    ([], None), (['--history-drafts'], True), (['--no-history-drafts'], False),
    (['--history-drafts', '--no-history-drafts'], False),
])
def test_history_drafts_reach_serving_without_changing_preparation(host, monkeypatch, flags, enabled):
    received = []
    monkeypatch.setattr(serve, '_serve_prepared', lambda prepared, bind, options: received.append(options) or 0)
    assert main(['debug-serve', 'qwen3.8-flash', '--port', str(free_port()), *flags]) == 0
    assert host['plan'].speculation == resolve('qwen3.8-flash').recipe.speculation
    assert received[0].get('history_drafts') is enabled
    assert host['exports'] == 0


@pytest.mark.parametrize('flags', [['--no-mtp'], ['--mtp-draft-tokens', '0'], ['--direct']])
def test_history_drafts_reject_disabled_or_direct_execution(host, capsys, flags):
    assert main(['debug-serve', 'qwen3.8-flash', '--port', str(free_port()), '--history-drafts', *flags]) == 2
    assert '--history-drafts requires' in capsys.readouterr().err
    assert 'prepared' not in host


@pytest.mark.parametrize("overlap", [False, True])
def test_mixed_prefill_opt_in_reaches_export(host, capsys, overlap):
    options = () if overlap else ("--no-stream-overlap",)
    assert export("qwen3.8-flash", *options) == 0
    assert export("qwen3.8-flash", "--mixed-prefill", *options) == 0
    assert host["exports"] == 2
    assert host["plan"].concurrency == 32 and host["plan"].speculative_token_capacity == 128
    assert host["plan"].mixed_prefill and len(host["plan"].mixed_shapes) == 123
    assert host["plan"].stream_overlap == overlap
    assert "mixed prefill on" in capsys.readouterr().err
    assert export("qwen3.8-flash", "--mixed-prefill", "--no-mtp", *options) == 0
    assert host["exports"] == 3 and host["plan"].speculation is None
    assert host["plan"].stream_overlap == overlap
    assert len(host["plan"].mixed_shapes) == 50
    assert all(not enabled for enabled, _, _, _ in host["plan"].mixed_shapes)


@pytest.mark.parametrize("compression", [["--expert-scale-compression"], ["--expert-scale-compression", "csf"]])
def test_expert_scale_compression_reaches_export(host, monkeypatch, capsys, compression):
    original = serve.resolve_model(None, None)
    checkpoint = replace(original.checkpoint, scale_compressed_device_bytes={"csf": original.checkpoint.device_bytes})
    model = ResolvedModel(original.recipe, checkpoint)
    monkeypatch.setattr(serve, "resolve_model", lambda *_: model)
    assert export("qwen3.8-flash") == 0
    assert export("qwen3.8-flash", *compression) == 0
    assert host["exports"] == 2
    assert host["plan"].expert_scale_compression == "csf"
    assert "expert scales csf" in capsys.readouterr().err


def test_serialized_branches_reach_the_plan_and_the_startup_line(host, capsys):
    assert export("qwen3.8-flash") == 0
    assert host["plan"].stream_overlap and "branches serialized" not in capsys.readouterr().err
    assert export("qwen3.8-flash", "--no-stream-overlap") == 0
    assert host["plan"].stream_overlap is False
    assert "branches serialized" in capsys.readouterr().err


@pytest.mark.parametrize("reserve", [0, 2 * GIB])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_export_does_not_launch_a_server_and_preserves_the_serving_plan(host, tmp_path, reserve, mixed, overlap):
    config = tmp_path / "config.toml"
    config.write_text(f'[memory]\nreserve = "{reserve}"\n')
    assert main(["export", "qwen3.8-flash", "--config", str(config),
                 *(["--mixed-prefill"] if mixed else []), *([] if overlap else ["--no-stream-overlap"])]) == 0
    assert host["exports"] == 1 and host["prepared"].started is None
    assert host["plan"].prefill_chunk == 4096 and host["compile_forwards"]
    assert host["plan"].mixed_prefill == mixed
    assert host["plan"].stream_overlap == overlap
    assert host["options"]["queue_capacity"] == 128
    assert host["options"]["served_names"] == ["Qwen3.8-Flash-Next"]
    assert host["options"]["reserve_bytes"] == reserve
    assert "headroom_bytes" not in host["options"]
    assert host["prepared"].closed
    assert host["cache"] == Configuration.load(config).directory("package_cache")


def test_development_listener_and_authentication_use_configuration(host, tmp_path, monkeypatch):
    port = free_port()
    config = tmp_path / "config.toml"
    config.write_text(f'[server]\nport = {port}\napi_key = "secret"\n')
    calls = []
    monkeypatch.setattr(serve, "_serve_prepared", lambda prepared, bind, options: calls.append((bind, options)))
    assert main(["debug-serve", "qwen3.8-flash", "--gpus", "0", "--config", str(config)]) == 0
    assert calls[0][0] == f"127.0.0.1:{port}"
    assert calls[0][1]["api_key"] == "secret"
    assert host["exports"] == 0


def test_listener_flags_override_configuration(host, tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('[server]\nhost = "0.0.0.0"\nport = 8000\n')
    port = free_port()
    calls = []
    monkeypatch.setattr(serve, "_serve_prepared", lambda prepared, bind, options: calls.append(bind))
    assert main(["debug-serve", "qwen3.8-flash", "--gpus", "0", "--config", str(config),
                 "--host", "127.0.0.1", "--port", str(port)]) == 0
    assert calls == [f"127.0.0.1:{port}"]


def test_an_address_in_use_fails_before_preparation(host, capsys):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        assert main(["debug-serve", "qwen3.8-flash", "--port", str(occupied.getsockname()[1])]) == 2
    assert "cannot listen on" in capsys.readouterr().err and "prepared" not in host
    assert main(["debug-serve", "qwen3.8-flash", "--host", "localhost", "--port", str(free_port())]) == 2


@pytest.mark.parametrize("exception,code,message", [
    (RuntimeError("kernel tuning failed"), 5, "preparation failed: RuntimeError: kernel tuning failed"),
    (MemoryError("insufficient memory"), 4, "insufficient memory during model preparation"),
])
def test_preparation_failures_prevent_export(host, capsys, exception, code, message):
    host["behavior"] = {"raises": exception}
    assert export("qwen3.8-flash") == code
    assert message in capsys.readouterr().err and host["exports"] == 0
    assert host["signals"].exited >= 1


def test_exhausted_gpu_memory_is_a_capacity_error(host, capsys):
    host["behavior"] = {"raises": sys.modules["torch"].OutOfMemoryError("CUDA out of memory.")}
    assert export("qwen3.8-flash") == 4
    assert "Fewer concurrent requests, --no-mtp, or more GPUs" in capsys.readouterr().err


def test_preparation_preserves_the_configured_reserve(host, tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text('[memory]\nreserve = "2G"\n')
    host["behavior"] = {"extra": 2 * GIB}
    assert export("qwen3.8-flash", "--config", str(config)) == 4
    assert "preparation left 1.0 GiB free" in capsys.readouterr().err
    assert host["prepared"].closed and host["exports"] == 0


def test_uncached_checkpoint_downloads_before_export(host, monkeypatch):
    fetched = []
    monkeypatch.setattr(ResolvedModel, "cached", lambda self, hub_cache: False)
    monkeypatch.setattr(serve, "fetch", lambda model, hub_cache, progress_bars: fetched.append(model.checkpoint.alias))
    assert export("qwen3.8-flash") == 0
    assert fetched == ["qwen3.8-flash"]


def test_environment_selects_gpus_by_uuid_and_places_caches(monkeypatch, tmp_path):
    from ignition.cli.config import Configuration

    monkeypatch.setattr(environment.patches, "apply", lambda: ())
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "B12X_COMPILE_CACHE_DIR", "TRITON_CACHE_DIR",
                 "TORCHINDUCTOR_CACHE_DIR", "HF_HUB_CACHE", "FLASHINFER_WORKSPACE_BASE"):
        monkeypatch.setenv(name, "previous")
    path = tmp_path / "config.toml"
    path.write_text(f'[paths]\ncuda = "{tmp_path}/cuda"\nkernel_cache = "{tmp_path}/kernels"\n')
    gpus = (types.SimpleNamespace(uuid="GPU-b"), types.SimpleNamespace(uuid="GPU-a"))
    environment.apply(Configuration.load(path), gpus)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-b,GPU-a"
    assert os.environ["B12X_COMPILE_CACHE_DIR"] == f"{tmp_path}/kernels/b12x"
    assert Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).is_relative_to(tmp_path / "kernels" / "flashinfer")
    assert os.environ["TRITON_CACHE_DIR"] == f"{tmp_path}/kernels/triton"
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == f"{tmp_path}/kernels/inductor"
    assert os.environ["HF_HUB_CACHE"] == f"{tmp_path}/.cache/huggingface/hub"
    assert os.environ["CUDA_HOME"] == f"{tmp_path}/cuda"


@pytest.mark.parametrize("configured", (False, True))
@pytest.mark.parametrize("inherited_cuda", (None, "/system/cuda"))
def test_environment_leaves_pip_cuda_toolchain_selection_to_cutile(monkeypatch, tmp_path, configured, inherited_cuda):
    monkeypatch.setattr(environment.patches, "apply", lambda: ())
    package = tmp_path / "site-packages" / "nvidia"
    compiler = package / "cu13" / "bin" / "nvcc"
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    (package / "__init__.py").write_text("")
    monkeypatch.delitem(sys.modules, "nvidia", raising=False)
    monkeypatch.syspath_prepend(str(package.parent))
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_PATH", "TRITON_PTXAS.EXE_PATH",
                 "B12X_COMPILE_CACHE_DIR", "TRITON_CACHE_DIR",
                 "TORCHINDUCTOR_CACHE_DIR", "HF_HUB_CACHE", "FLASHINFER_WORKSPACE_BASE"):
        monkeypatch.setenv(name, "previous")
    if inherited_cuda is None:
        monkeypatch.delenv("CUDA_HOME", raising=False)
    else:
        monkeypatch.setenv("CUDA_HOME", inherited_cuda)
    cuda = tmp_path / "cuda"
    path = tmp_path / "config.toml"
    path.write_text(f'[paths]\ncuda = "{cuda}"\n' if configured else "")

    environment.apply(Configuration.load(path), ())

    assert os.environ.get("CUDA_HOME") == (str(cuda) if configured else inherited_cuda)
    assert "nvidia" not in sys.modules


def test_environment_sets_the_recipe_b12x_settings_over_inherited_values(monkeypatch, tmp_path):
    from ignition.cli.config import Configuration
    from ignition.recipes.kimi3.recipe import RECIPE

    monkeypatch.setattr(environment.patches, "apply", lambda: ())
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "B12X_COMPILE_CACHE_DIR", "TRITON_CACHE_DIR",
                 "TORCHINDUCTOR_CACHE_DIR", "HF_HUB_CACHE", "FLASHINFER_WORKSPACE_BASE", *RECIPE.b12x_environment):
        monkeypatch.setenv(name, "inherited")
    monkeypatch.setenv("B12X_PCIE_ALLREDUCE_ALGORITHM", "hierarchical")
    path = tmp_path / "config.toml"
    path.write_text(f'[paths]\nkernel_cache = "{tmp_path}/kernels"\n')
    environment.apply(Configuration.load(path), (types.SimpleNamespace(uuid="GPU-a"),), RECIPE.b12x_environment)
    assert {name: os.environ[name] for name in RECIPE.b12x_environment} == dict(RECIPE.b12x_environment)
    assert os.environ["B12X_PCIE_ALLREDUCE_ALGORITHM"] == "island_rs"


def test_flashinfer_workspaces_separate_checkouts_without_importing_them(monkeypatch, tmp_path):
    import importlib

    monkeypatch.delitem(sys.modules, "flashinfer", raising=False)
    roots = [tmp_path / name for name in ("source_a", "source_b")]
    for root in roots:
        package = root / "flashinfer"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('raise AssertionError("preparation imported too early")\n')
    monkeypatch.syspath_prepend(str(roots[0]))
    first = environment.flashinfer_workspace(tmp_path / "kernels")
    assert first == environment.flashinfer_workspace(tmp_path / "kernels")
    assert "flashinfer" not in sys.modules

    monkeypatch.syspath_prepend(str(roots[1]))
    importlib.invalidate_caches()
    second = environment.flashinfer_workspace(tmp_path / "kernels")
    assert second != first
    assert second.parent == first.parent == tmp_path / "kernels" / "flashinfer"
    assert "flashinfer" not in sys.modules

    alias = tmp_path / "source_alias"
    alias.symlink_to(roots[1], target_is_directory=True)
    monkeypatch.syspath_prepend(str(alias))
    assert environment.flashinfer_workspace(tmp_path / "kernels") == second


def test_tensor_parallel_export_preserves_rank_selection(host):
    assert export("qwen3.8-flash", "--gpus", "0,1") == 0
    assert host["plan"].world_size == 2
    assert [gpu.index for gpu in host["gpus"]] == [0, 1]


def test_multihost_export_dispatches_with_configured_cache(host, tmp_path, monkeypatch, capsys):
    from ignition.preparation.package_snapshot import _OPTIONS

    config = tmp_path / 'export.toml'
    cache = tmp_path / 'prepared-models'
    config.write_text(f'[paths]\npackage_cache = "{cache.as_posix()}"\n')
    entry = cache / 'models' / 'qwen38' / 'gb202-tp2-test'
    calls = []

    def export_hosts(args):
        calls.append((args.ranks, args.rendezvous, _OPTIONS.get().cache))
        return 0, types.SimpleNamespace(entry_path=entry)

    monkeypatch.setattr(serve, '_export_hosts', export_hosts)
    assert main(['export', 'qwen3.8-flash', '--ranks', '2', '--rendezvous', '127.0.0.1:29630',
                 '--config', str(config)]) == 0
    assert calls == [(2, '127.0.0.1:29630', cache)]
    assert f'Exported cache entry: {entry}' in capsys.readouterr().out
    assert _OPTIONS.get() is None


@pytest.mark.parametrize('command', ['export', 'debug-serve'])
def test_local_tuning_gpus_cannot_change_multihost_sharding(tmp_path, capsys, command):
    arguments = [command, 'qwen3.8-flash', '--ranks', '2', '--tuning-gpus', '0,1']
    assert main(arguments) != 0
    assert 'multi-host ranks retain their sharding' in capsys.readouterr().err


def test_export_rejects_size_sharding(host):
    assert export("qwen3.8-flash") == 0
    with pytest.raises(SystemExit) as error:
        export("qwen3.8-flash", "--shard-size", "32G")
    assert error.value.code == 2
    assert host["exports"] == 1


def test_export_reports_cache_entry_and_repository(host, tmp_path, capsys):
    config = tmp_path / 'export.toml'
    cache = tmp_path / 'prepared-models'
    config.write_text(f'[paths]\npackage_cache = "{cache.as_posix()}"\n')
    assert main(['export', 'qwen3.8-flash', '--config', str(config)]) == 0
    assert host['cache'] == cache
    output = capsys.readouterr().out
    assert f"Exported cache entry: {host['entry_path']}" in output
    assert 'Hugging Face target:' not in output
    assert main(['export', 'qwen3.8-flash', '--repo-id', 'example/custom']) == 0
    assert 'Hugging Face target: example/custom' in capsys.readouterr().out


def test_no_weights_requires_cache_before_preparation(host, tmp_path, capsys):
    assert main(['export', 'qwen3.8-flash', '--no-weights']) == 2
    assert host['exports'] == 0 and 'plan' not in host
    assert 'existing blob cache' in capsys.readouterr().err
    assert main(['export', 'qwen3.8-flash', '--repo-id', '../escape']) == 2
    assert host['exports'] == 0


def test_reuse_weights_requires_cached_version_before_preparation(host, capsys):
    assert main(['export', 'qwen3.8-flash', '--reuse-weights']) == 2
    assert '--reuse-weights requires a cached default model version' in capsys.readouterr().err
    assert 'prepared' not in host


def test_reuse_weights_passes_cached_metadata_through_export(host, monkeypatch, tmp_path):
    import flare_heliopause
    from ignition.preparation.package_snapshot import _OPTIONS

    source = {'envelope': [1, 2, 3], 'publishers': []}
    requests = []

    def reuse(cache, model, profiles):
        requests.append((cache, model, json.loads(profiles)))
        return types.SimpleNamespace(metadata=lambda: json.dumps(source))

    def prepare(args):
        settings = _OPTIONS.get()
        assert settings.no_weights and settings.reuse_weights == source
        return 0, types.SimpleNamespace(entry_path=tmp_path / 'entry')

    monkeypatch.setattr(flare_heliopause, 'export_reuse_weights', reuse)
    monkeypatch.setattr(serve, '_prepare', prepare)
    assert main(['export', 'qwen3.8-flash', '--reuse-weights']) == 0
    assert requests[0][1] == resolve('qwen3.8-flash').model_id
    assert _OPTIONS.get() is None


def test_export_weight_reuse_and_cache_only_flags_are_exclusive(host):
    with pytest.raises(SystemExit, match='2'):
        main(['export', 'qwen3.8-flash', '--reuse-weights', '--no-weights'])


@pytest.mark.parametrize("effort", ["off", "default", "audit"])
def test_autotune_effort_reaches_plan_and_startup_line(host, capsys, monkeypatch, effort):
    monkeypatch.setenv("COLUMNS", "400")
    assert export("qwen3.8-flash", "--autotune-effort", effort) == 0
    assert host["plan"].autotune_effort == effort
    assert host["plan"].to_dict()["autotune_effort"] == effort
    assert f"autotune effort {effort}" in capsys.readouterr().err


@pytest.mark.parametrize("native", [False, True])
def test_native_dense_activations_reach_plan(host, monkeypatch, native):
    monkeypatch.setattr(serve, "_serve_prepared", lambda *args: 0)
    flags = ["--native-dense-activations"] if native else []
    assert main(["debug-serve", "qwen3.8-flash", "--port", str(free_port()), *flags]) == 0
    assert host["plan"].native_dense_activations is native
    assert host["plan"].to_dict()["native_dense_activations"] is native


def test_lock_gpu_clocks_reaches_preparation_plan(host):
    assert export("qwen3.8-flash", "--lock-gpu-clocks") == 0
    assert host["plan"].lock_gpu_clocks is True
    assert host["plan"].to_dict()["lock_gpu_clocks"] is True


@pytest.mark.parametrize("command", ["export", "debug-serve"])
def test_runtime_collective_policy_requires_ladder_before_preparation(host, capsys, command):
    assert main([command, "qwen3.8-flash", "--collective-tuning", "runtime"]) == 2
    assert "requires --collective-ladder PATH" in capsys.readouterr().err
    assert "prepared" not in host and host["exports"] == 0


@pytest.mark.parametrize("arguments,configured,expected", [
    ((), None, 8 << 30),
    (("--compile-headroom", "6G"), None, 6 << 30),
    ((), "12G", 12 << 30),
    (("--compile-headroom", "6.5GiB"), "12G", 13 << 29),
])
def test_compile_headroom_reaches_plan_and_startup_line(host, capsys, monkeypatch, tmp_path,
                                                         arguments, configured, expected):
    monkeypatch.setenv("COLUMNS", "400")
    if configured is not None:
        config = tmp_path / "headroom.toml"
        config.write_text(f'[preparation]\ncompile_headroom = "{configured}"\n')
        arguments = (*arguments, "--config", str(config))
    assert export("qwen3.8-flash", *arguments) == 0
    assert host["plan"].compile_headroom_bytes == expected
    assert host["plan"].to_dict()["compile_headroom_bytes"] == expected
    assert f"compile headroom {expected / (1 << 30):g} GiB" in capsys.readouterr().err


def test_debug_serve_carries_compile_headroom_into_plan(host, monkeypatch):
    monkeypatch.setattr(serve, "_serve_prepared", lambda *args: 0)
    assert main(["debug-serve", "qwen3.8-flash", "--port", str(free_port()), "--compile-headroom", "6G"]) == 0
    assert host["plan"].compile_headroom_bytes == 6 << 30


@pytest.mark.parametrize("value", ["-6G", "lots"])
def test_invalid_compile_headroom_fails_before_preparation(host, capsys, value):
    assert export("qwen3.8-flash", f"--compile-headroom={value}") == 2
    assert "--compile-headroom" in capsys.readouterr().err
    assert "plan" not in host
