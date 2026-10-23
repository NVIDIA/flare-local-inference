"""Preparation error and speculation formatting."""

import sys


def _console(stream, **options):
    from rich.console import Console

    return Console(file=stream, highlight=False, markup=False, emoji=False, **options)



def error(message: str) -> None:
    """Print a failure on stderr with its continuation lines aligned under the first."""
    from rich.text import Text

    _console(sys.stderr, soft_wrap=True).print(
        Text.assemble(("error:", "bold red"), " " + message.replace("\n", "\n       ")))



def speculation(settings, *, brief: bool = False) -> str:
    """Describe speculation settings: the drafts verified per round and where they come from.

    The brief form fits a startup line.
    """
    if settings is None:
        return "speculation off" if brief else "off"
    drafts = settings.draft_tokens
    depth = f"{'up to' if settings.adaptive else 'exactly'} {drafts} draft token{'' if drafts == 1 else 's'}"
    selection = " · GPU prefix selection" if settings.draft_prefix_selection == "gpu" else ""
    heads = [f"{head} head {getattr(settings, head + '_head_quantization').upper()}"
             for head in ("draft", "markov") if getattr(settings, head + '_head_quantization') is not None]
    precision = " · " + ", ".join(heads) if heads else ""
    if brief:
        return depth + selection + precision
    return f"{depth} per round · MTP module" + selection + precision
