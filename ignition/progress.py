"""Terminal progress for model preparation; callbacks retain plain log output."""

from contextlib import contextmanager
import sys
from time import monotonic


class StartupProgress:
    def __init__(self, *, stream=None):
        from rich.console import Console

        self.console = Console(file=sys.stderr if stream is None else stream, highlight=False)
        self.active = None

    def __call__(self, message):
        if self.active is not None:
            self.active.update(message)


def graph_catalog(progress, signatures):
    """Persist the registered graph shapes without changing the active phase."""
    if progress is None:
        return
    from collections import Counter, defaultdict
    from math import prod
    from rich.table import Table

    counts = Counter((s.kind, s.speculation_enabled, s.draft_tokens or 0,
                      s.batch_capacity, s.token_capacity, s.packed_token_capacity, s.prefill_capacity)
                     for s in signatures)
    groups = defaultdict(list)
    for (kind, mode, drafts, batch, width, packed, prefill), copies in counts.items():
        groups[kind, mode, drafts, copies].append(((batch,), (width,), (packed,), (prefill,)))
    table = Table(title=f"Graph catalog: {sum(counts.values())} graphs (before compilation/capture)", title_justify="left",
                  box=None, padding=(0, 1), caption="B=batch  W=logical width/tokens  P=packed rows  D=draft depth",
                  caption_justify="left")
    for name in ("Kind", "Spec", "B", "W", "P", "D", "Prefill", "Count"):
        table.add_column(name, overflow="fold", no_wrap=name == "Kind")
    def values(items):
        return ",".join(str(value) if value else "-" for value in items)

    for (kind, mode, drafts, copies), shapes in groups.items():
        # Merge one axis only when every other axis has identical coverage.
        for axis in (2, 0, 1, 3):
            merged = defaultdict(set)
            for shape in shapes:
                merged[shape[:axis] + shape[axis + 1:]].update(shape[axis])
            shapes = [key[:axis] + (tuple(sorted(values)),) + key[axis:]
                      for key, values in merged.items()]
        for batch, width, packed, prefill in shapes:
            table.add_row(kind, "-" if mode is None else "on" if mode else "off",
                          values(batch), values(width), values(packed), str(drafts) if drafts else "-",
                          values(prefill), str(copies * prod(map(len, (batch, width, packed, prefill)))))
    if isinstance(progress, StartupProgress):
        progress.console.print(table)
    else:
        from rich.console import Console

        console = Console(width=120, color_system=None)
        with console.capture() as output:
            console.print(table)
        for line in output.get().splitlines():
            progress(line)


class _Phase:
    def __init__(self, progress, label, total):
        self.progress, self.label, self.total = progress, label, total
        self.console = progress.console if isinstance(progress, StartupProgress) else None
        self.completed = 0
        self.finished = False
        self.started = monotonic()
        self.detail = ""
        self.live = None
        self.failed = False

    def update(self, detail=None, *, advance=0):
        if detail is not None:
            self.detail = str(detail)
        self.completed += advance
        if self.progress is not None and not isinstance(self.progress, StartupProgress) and detail is not None:
            self.progress(f"{self.label}: {self.detail}")

    def __rich__(self):
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        from rich.text import Text

        now = monotonic()
        if self.failed:
            status = Text("  FAILED  ", style="bold red")
        elif self.finished:
            status = Text("    OK    ", style="bold green")
        else:
            status = ProgressBar(total=self.total, completed=self.completed, width=10,
                                 pulse=self.total is None, complete_style="cyan",
                                 finished_style="green", pulse_style="cyan")
        label = Text(f"] {self.label}.", overflow="ellipsis", no_wrap=True)
        if self.detail:
            label.append(f"  {self.detail}", style="dim")
        count = f"{self.completed}/{self.total}  " if self.total is not None else ""
        timing = Text(f"  {count}{now - self.started:.1f}s", style="dim")
        row = Table.grid(padding=0)
        row.add_column(width=1, no_wrap=True)
        row.add_column(width=10, no_wrap=True)
        # Only the label gives way in a narrow console; its text ends in an ellipsis on one line.
        row.add_column()
        row.add_column(width=len(timing.plain), no_wrap=True)
        row.add_row(Text("["), status, label, timing)
        return row


@contextmanager
def phase(progress, label, *, total=None, live=True):
    """Display completed work and elapsed time without estimating unknown progress."""
    stage = _Phase(progress, label, total)
    display = progress if isinstance(progress, StartupProgress) else None
    previous = display.active if display is not None else None
    if display is not None:
        from rich.live import Live

        display.active = stage
        if live and display.console.is_terminal:
            stage.live = Live(stage, console=display.console, refresh_per_second=8,
                              redirect_stdout=False, redirect_stderr=False)
            stage.live.start()
        else:
            display.console.print(stage)
    elif progress is not None:
        progress(f"{label} started")
    try:
        yield stage
    except BaseException:
        stage.failed = True
        raise
    else:
        stage.finished = True
        if stage.total is not None:
            stage.completed = stage.total
    finally:
        if stage.live is not None:
            stage.live.stop()
        elif progress is not None:
            message = f"{label} {'failed' if stage.failed else 'complete'} in {monotonic() - stage.started:.2f}s"
            if display is None:
                progress(message)
            else:
                display.console.print(stage)
        if display is not None:
            display.active = previous
