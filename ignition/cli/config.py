"""The configuration file: its schema, built-in values, and precedence under flags.

A flag overrides the configuration file, which overrides the recipe. Every
resolved setting carries the source that supplied it.
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ignition.errors import UsageError

DEFAULT_PATH = Path("~/.config/flare/config.toml")

_SECTIONS = {
    "publishing": {"signing_key": str, "namespace": str, "official_source": str},
    "server": {"host": str, "port": int, "api_key": str},
    "defaults": {"model": str, "max_concurrency": int, "image_concurrency": int,
                 "speculative_token_capacity": int, "speculative_graph_budget": int},
    "memory": {"reserve": str},
    "preparation": {"compile_workers": int, "autotune_effort": str, "collective_tuning": str, "checkpoint_token_file": str},
    "prefix_cache": {"memory": str, "disk_limit": str, "drain_on_shutdown": str},
    "paths": {"hub_cache": str, "kernel_cache": str, "package_cache": str, "export_staging": str,
              "prefix_cache": str, "cuda": str},
}
_MODEL_KEYS = {"max_concurrency": int, "image_concurrency": int,
               "speculative_token_capacity": int, "speculative_graph_budget": int}
# Empty prefix-cache values leave the decision to Flare, which derives them from the host.
_BUILT_IN = {
    "server.host": "127.0.0.1", "server.port": 8000, "server.api_key": "",
    "memory.reserve": "0",
    "preparation.compile_workers": 0,
    "preparation.autotune_effort": "default",
    "preparation.collective_tuning": None,
    "preparation.checkpoint_token_file": "~/.config/flare/agent.token",
    "prefix_cache.memory": "", "prefix_cache.disk_limit": "", "prefix_cache.drain_on_shutdown": "",
    "paths.hub_cache": "~/.cache/huggingface/hub", "paths.kernel_cache": "~/.cache/flare/kernels",
    "paths.package_cache": "~/.cache/flare/packages",
    "paths.export_staging": "",
    "paths.prefix_cache": "",
    "paths.cuda": "",
}
AUTOTUNE_EFFORTS = ("off", "default", "audit")
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:([KMGT])(?:iB|B)?)?", re.IGNORECASE)


def parse_size(text: str, what: str = "memory size") -> int:
    """Parse a byte count with an optional binary K, M, G, or T suffix."""
    match = _SIZE.fullmatch(text.strip())
    if match is None:
        raise UsageError(f'{what} {text!r} must be a number with an optional K, M, G, or T suffix, such as "16G"')
    return int(float(match[1]) * (1 << 10 * " KMGT".index((match[2] or " ").upper())))


@dataclass(frozen=True)
class Setting:
    value: object
    source: str


class Configuration:
    """Validated configuration file contents; an absent default file is empty."""

    def __init__(self, path: Path, values: dict, exists: bool):
        self.path, self.values, self.exists = path, values, exists

    @classmethod
    def load(cls, path: Path | None = None) -> "Configuration":
        resolved = (DEFAULT_PATH if path is None else Path(path)).expanduser()
        if not resolved.is_file():
            if path is not None:
                raise UsageError(f"configuration file {resolved} does not exist")
            return cls(resolved, {}, False)
        try:
            values = tomllib.loads(resolved.read_text())
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            raise UsageError(f"{resolved}: {error}") from error
        for section, table in values.items():
            if section == "models" and isinstance(table, dict):
                for alias, overrides in table.items():
                    cls._validate(resolved, f'models."{alias}"', overrides, _MODEL_KEYS)
            elif section in _SECTIONS:
                cls._validate(resolved, section, table, _SECTIONS[section])
            else:
                raise UsageError(f"{resolved}: unknown section [{section}]")
        configuration = cls(resolved, values, True)
        parse_size(configuration.get("memory.reserve").value)
        for key in ("prefix_cache.memory", "prefix_cache.disk_limit"):
            if configuration.get(key).value:
                parse_size(configuration.get(key).value, key.replace(".", " "))
        return configuration

    @staticmethod
    def _validate(path, section, table, schema):
        if not isinstance(table, dict):
            raise UsageError(f"{path}: [{section}] must be a table")
        for key, value in table.items():
            if key not in schema:
                raise UsageError(f"{path}: unknown key {key} in [{section}]")
            if type(value) is not schema[key]:
                raise UsageError(f"{path}: {key} in [{section}] must be {schema[key].__name__}")
            if key == "drain_on_shutdown":
                text = value.strip()
                if text in ("", "on", "off", "0"):
                    continue
                match = re.fullmatch(r"([0-9.eE+\-]+)(ms|s|m|h)?", text)
                try:
                    seconds = float(match[1]) * {None: 1, "ms": .001, "s": 1, "m": 60, "h": 3600}[match[2]] if match else 0
                except ValueError:
                    seconds = 0
                if not 0.5e-9 < seconds < 2**63:
                    raise UsageError(f'prefix_cache drain_on_shutdown {text!r} must be on, off, or a positive duration such as 90s or 2m')
            if key == "autotune_effort" and value not in AUTOTUNE_EFFORTS:
                raise UsageError(f"{path}: autotune_effort must be one of {', '.join(AUTOTUNE_EFFORTS)}")
            if key == "collective_tuning" and value not in ("static", "runtime"):
                raise UsageError(f"{path}: collective_tuning must be static or runtime")
            if key == "port" and not 1 <= value <= 65535:
                raise UsageError(f"{path}: port in [{section}] must be 1 through 65535")
            if key in ("max_concurrency", "image_concurrency", "speculative_token_capacity",
                       "speculative_graph_budget", "compile_workers") and value < 1:
                raise UsageError(f"{path}: {key} in [{section}] must be at least 1")

    def get(self, key: str, flag=None) -> Setting:
        """Resolve `section.key` from a flag, the file, or the built-in value."""
        if flag is not None:
            return Setting(flag, "flag")
        section, name = key.split(".")
        if name in self.values.get(section, {}):
            return Setting(self.values[section][name], f"[{section}]")
        return Setting(_BUILT_IN[key], "default")

    @property
    def model(self) -> str | None:
        return self.values.get("defaults", {}).get("model")

    def max_concurrency(self, alias: str, recipe_default: int, flag: int | None = None) -> Setting:
        return self.model_setting(alias, "max_concurrency", recipe_default, flag)

    def model_setting(self, alias: str, key: str, recipe_default, flag=None) -> Setting:
        if flag is not None:
            return Setting(flag, "flag")
        for section, table in ((f'[models."{alias}"]', self.values.get("models", {}).get(alias, {})),
                               ("[defaults]", self.values.get("defaults", {}))):
            if key in table:
                return Setting(table[key], section)
        return Setting(recipe_default, "recipe")

    @property
    def reserve_bytes(self) -> int:
        return parse_size(self.get("memory.reserve").value)

    def directory(self, name: str) -> Path | None:
        """Return a `[paths]` entry with the home directory expanded, or None when it is empty."""
        value = self.get(f"paths.{name}").value
        return Path(value).expanduser() if value else None

    def write_initial(self, model: str, max_concurrency: int) -> None:
        """Create the file with every resolved value present as a comment."""
        lines = [
            "# Flare configuration. Every key is optional; remove a leading # to set it.",
            "", "[publishing]",
            "", "[server]",
            f'# host = "{self.get("server.host").value}"',
            f'# port = {self.get("server.port").value}',
            '# api_key = ""                    # empty: no authentication',
            "", "[defaults]",
            f'# model = "{model}"',
            f"# max_concurrency = {max_concurrency}",
            '# image_concurrency = 4          # active image requests; bounded by max_concurrency',
            '# speculative_token_capacity = 8  # wider target graphs require batch * width within this limit',
            '# speculative_graph_budget = 64  # packed round graphs; excludes width-one and ordinary graphs',
            "", "[preparation]",
            '# compile_workers = 24           # CPU processes per host; default: automatic, bounded by memory',
            '# autotune_effort = "default"    # off, default, audit',
            '# collective_tuning = "static"  # static, runtime; omitted uses recipe default',
            "", "[memory]",
            f'# reserve = "{self.get("memory.reserve").value}"'
            "                   # GPU memory left untouched for processes that start later",
            "", "[paths]",
            f'# hub_cache = "{self.get("paths.hub_cache").value}"',
            f'# kernel_cache = "{self.get("paths.kernel_cache").value}"',
            f'# package_cache = "{self.get("paths.package_cache").value}"',
            '# export_staging = ""            # rank files an export assembles from; empty: beside the package',
            '# cuda = "/opt/cuda"              # explicit CUDA toolkit directory for native builds',
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(lines) + "\n")
