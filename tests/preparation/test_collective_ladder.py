"""Offline ladders select final geometry at inclusive payload boundaries."""

from copy import deepcopy
from pathlib import Path

import pytest

from ignition.errors import UsageError
from ignition.preparation.collective_ladder import load, resolve, select, validate

LADDER = Path(__file__).resolve().parents[2] / "ignition/preparation/ladders/gb202_pcie_tp4.json"


def choose(data, size, **changes):
    return select(data, size, **(dict(world_size=4, capability=(12, 0), inplace=False) | changes))


def test_supplied_ladder_covers_every_interval():
    data = load(LADDER)
    assert data["name"] == "gb202_pcie_tp4"
    assert len(data["buckets"]) == 210
    assert data["min_bytes"] == 16
    assert data["buckets"][-1]["max_bytes"] == 8 << 20
    lower = data["min_bytes"]
    for row in data["buckets"]:
        assert choose(data, lower) is row
        assert choose(data, row["max_bytes"]) is row
        assert sum(map(len, row["profiles"].values())) <= 4
        lower = row["max_bytes"] + 16


@pytest.mark.parametrize("size", [0, 8, 1025, (8 << 20) + 16, True])
def test_no_extrapolation_or_unaligned_payload(size):
    with pytest.raises(UsageError, match="no bucket"):
        choose(load(LADDER), size)


@pytest.mark.parametrize("changes", [dict(world_size=2), dict(capability=(12, 1)), dict(inplace=True)])
def test_rejects_incompatible_reduction(changes):
    with pytest.raises(UsageError, match="does not match"):
        choose(load(LADDER), 4096, **changes)


@pytest.mark.parametrize("changes", [dict(version=2), dict(extra=1), dict(world_size=1),
    dict(dtype="float16"), dict(inplace=0), dict(min_bytes=1025), dict(buckets=[]), dict(cache_policy="unknown")])
def test_invalid_contract(changes):
    with pytest.raises(UsageError, match="collective ladder"):
        validate(load(LADDER) | changes)


@pytest.mark.parametrize("changes", [dict(max_bytes=0), dict(implementation="oneshot"),
    dict(threads=96), dict(blocks=0), dict(profiles={}), dict(profiles={"full": [0]}),
    dict(profiles={"full": [65]}), dict(profiles={"full": [1, 1]}), dict(profiles={"bulk_push": [1]})])
def test_invalid_bucket(changes):
    data = deepcopy(load(LADDER))
    data["buckets"][0].update(changes)
    with pytest.raises(UsageError, match="collective ladder"):
        validate(data)


@pytest.mark.parametrize("family", ["joint_latency", "joint_bandwidth"])
@pytest.mark.parametrize("mode,counts", [("serial_push", [1]), ("queued_push", [1, 16, 64])])
def test_destination_serialization_profiles_require_multiple_peers(family, mode, counts):
    from b12x.comm.pcie._joint_profile import selected_profile

    data = load(LADDER)
    data["buckets"][0].update(implementation=family, profiles={mode: counts})
    assert validate(data) is data
    for count in counts:
        assert selected_profile(family, 64, mode, count, world_size=4) & 0xffff == count
    data["world_size"] = 2
    with pytest.raises(UsageError, match="more than two ranks"):
        validate(data)


@pytest.mark.parametrize("counts", [[2], [1, 2]])
def test_serial_destination_profile_rejects_multiple_active_ctas(counts):
    data = load(LADDER)
    data["buckets"][0]["profiles"] = {"serial_push": counts}
    with pytest.raises(UsageError, match="one active CTA"):
        validate(data)


def test_runtime_requires_file_and_static_rejects_it(tmp_path):
    with pytest.raises(UsageError, match="requires --collective-ladder"):
        resolve("runtime", None)
    with pytest.raises(UsageError, match="requires --collective-tuning runtime"):
        resolve("static", LADDER)
    with pytest.raises(UsageError, match="cannot read"):
        resolve("runtime", tmp_path / "missing.json")
    assert resolve("static", None) is None
    assert resolve("runtime", LADDER) == load(LADDER)
    bad = tmp_path / "bad.json"
    bad.write_text("{broken")
    with pytest.raises(UsageError, match="cannot read"):
        load(bad)


@pytest.mark.parametrize("command", ["export", "debug-serve", "bench"])
def test_preparation_cli_accepts_ladder(command):
    from ignition.cli import _parser

    args = _parser().parse_args([command, "muse_spark", "--collective-tuning", "runtime", "--collective-ladder", str(LADDER)])
    assert args.collective_tuning == "runtime"
    assert args.collective_ladder == LADDER


@pytest.mark.parametrize("effort", ["off", "default", "audit"])
def test_b12x_accepts_pinned_geometry_at_both_ends_of_every_bucket(effort):
    from b12x.comm.pcie._collective_tuning import CollectiveConfig, CollectiveQuery, TUNING
    from b12x.comm.pcie._joint_profile import selected_profile
    from b12x.preparation import FrozenMapping, autotune_effort

    data = load(LADDER)
    lower = data["min_bytes"]
    with autotune_effort(effort):
        for row in data["buckets"]:
            family = row["implementation"]
            config = CollectiveConfig(implementation=family, threads=row["threads"], blocks=row["blocks"],
                                      submode=next(iter(row["profiles"])), active_blocks=1)
            for size in (lower, row["max_bytes"]):
                query = CollectiveQuery(operation="all_reduce", world_size=4, rank=0,
                    alternatives=FrozenMapping({family: dict(surface="OneshotAllReduce.all_reduce",
                        world_size=4, rank=0, topology="pcie_ipc", setup={},
                        call=dict(threads=256, blocks=64, submode=config.submode, active_blocks=1,
                                  size_packs=size // 16, stage_input=True, dtype="bfloat16"))}),
                    default=config)
                configured = TUNING.configure(query, device=None, override=config, search=effort != "off")
                assert configured.pinned == config
            for mode, counts in row["profiles"].items():
                for count in counts:
                    assert selected_profile(family, row["blocks"], mode, count, world_size=4) & 0xffff == count
            lower = row["max_bytes"] + 16


def test_native_export_preserves_ladder_contents():
    import json
    from ignition.preparation.package_proto import serving
    from ignition.preparation.runtime_package_pb2 import ModelIndex
    from tests.preparation.test_frontend_metadata import server

    data = load(LADDER)
    source = server()
    source["rebuild_manifest"] = dict(version=1, collective_tuning="runtime",
                                      collective_ladder_json=json.dumps(data))
    index = ModelIndex()
    serving(index, source)
    restored = ModelIndex.FromString(index.SerializeToString())
    assert restored.rebuild.collective_tuning == "runtime"
    assert json.loads(restored.rebuild.collective_ladder_json) == data
