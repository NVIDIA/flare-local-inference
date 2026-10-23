"""Offline command help drawn from recipe declarations."""

import argparse

from ignition.recipes import recipes


def recipe_help(model=None):
    catalog = [(recipe, checkpoint) for recipe in recipes() for checkpoint in recipe.checkpoints]
    selected = next(((recipe, checkpoint) for recipe, checkpoint in catalog
                     if model in (checkpoint.alias, checkpoint.repository)), None)
    if selected is None:
        lines = ["Available model recipes:",
                 f"  {'MODEL':<26}{'GPUs':<14}Drafts (default/max)"]
        for recipe, checkpoint in catalog:
            gpus = ",".join(map(str, recipe.tensor_parallel))
            drafts = ("none" if recipe.speculation is None else
                      f"{recipe.speculation.draft_tokens}/{recipe.max_draft_tokens}")
            lines.append(f"  {checkpoint.alias:<26}{gpus:<14}{drafts}")
        lines.extend(("", "Show a recipe's defaults, limits, and optional features:",
                      "  ignition export MODEL --help",
                      "MODEL also accepts org/repository[@revision] or a checkpoint directory.",
                      "GPU counts are supported tensor-parallel sizes; memory must still fit."))
        return "\n".join(lines)

    recipe, checkpoint = selected
    lines = [f"Model recipe: {checkpoint.alias}", f"  Checkpoint: {checkpoint.repository}",
             f"  --gpus: select {', '.join(map(str, recipe.tensor_parallel))} GPUs",
             f"  --max-concurrency: 1..{recipe.max_concurrency}; recipe default {recipe.concurrency}",
             f"  --prefill-chunk-size: 64..{recipe.context}, multiple of 64; default {recipe.prefill_chunk}"]
    speculation = recipe.speculation
    if speculation is not None:
        lines.extend((
            f"  --mtp-draft-tokens: 0..{recipe.max_draft_tokens}; default {speculation.draft_tokens}",
            f"  Verification: {'adaptive' if speculation.adaptive else 'fixed width'} by default;"
            " --no-mtp excludes draft weights and state.",
            "  --mtp-draft-tokens 0 retains draft state and captures width-one rounds."))
        if speculation.token_capacity is None:
            lines.extend(("  --speculative-token-capacity: at least drafts + 1; default is the larger of",
                          "    power-of-two padded concurrency and drafts + 1."))
        else:
            lines.append(f"  --speculative-token-capacity: at least drafts + 1; default {speculation.token_capacity}.")
        if speculation.packed_verification:
            lines.extend(("    Target rows are shared across retained request prefixes.",
                          "    Packed rounds need more target rows than active requests."))
            if speculation.graph_budget is not None:
                lines.extend((f"  --speculative-graph-budget: positive integer; default {speculation.graph_budget}.",
                              "    Bounds packed round graphs; excludes mandatory width-one and ordinary graphs."))
        else:
            lines.append("    Wider graphs require batch * width <= capacity (width = draft count + 1).")
        layout = "packed" if speculation.packed_verification else "rectangular"
        lines.extend(("    Width one covers every batch.",
                      f"  Prepared verification: {layout} rows with {speculation.draft_prefix_selection.upper()}"
                      " prefix selection."))
    else:
        lines.append("  Multi-token speculation: unavailable.")
    optional = []
    for encoding, compressed in checkpoint.scale_compressed_device_bytes.items():
        saving = 100 * (1 - compressed / checkpoint.device_bytes)
        optional.append(f"--expert-scale-compression {encoding}"
                        f" (estimated {saving:.0f}% less resident weight storage; actual savings vary)")
    if recipe.mixed_prefill:
        optional.append("--mixed-prefill (requires concurrency >= 2)")
    if recipe.history_copy:
        optional.extend(("--mtp-proposal-source copy", "--history-drafts"))
    if any(cost.page_tokens for cost in recipe.ranks.values()):
        optional.append("--no-context-parallel")
    if recipe.stream_overlap:
        optional.append("--no-stream-overlap")
    lines.append("  Optional features for this recipe:")
    lines.extend(f"    {feature}" for feature in optional)
    if not optional:
        lines.append("    none")
    lines.extend(("  Images per request: bounded by the prepared image-token budget" if recipe.max_images else
                  "  Images: unavailable.",
                  "  Defaults above come from the recipe; configuration and explicit flags override them.",
                  "  Supported GPU counts do not guarantee that the selected capacities fit in memory."))
    return "\n".join(lines)


class RecipeHelp(argparse.Action):
    """Show settings for a MODEL preceding --help without resolving or loading it."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings=option_strings, dest=dest, nargs=0, default=default, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.epilog = recipe_help(getattr(namespace, "model", None))
        parser.print_help()
        parser.exit()
