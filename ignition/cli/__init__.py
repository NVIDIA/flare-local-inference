"""The `ignition` preparation and development serving command line."""

import argparse
from pathlib import Path

from ignition.errors import FlareError


def _parser() -> argparse.ArgumentParser:
    from . import serve
    from . import publishing
    from .help import RecipeHelp, recipe_help

    parser = argparse.ArgumentParser(
        prog="ignition", allow_abbrev=False,
        description="Prepare native runtime packages for NVIDIA SM103, SM120, and SM121 GPUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=recipe_help())
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name, handler, summary):
        subparser = subparsers.add_parser(
            name, help=summary, description=summary, allow_abbrev=False, add_help=False,
            formatter_class=argparse.RawDescriptionHelpFormatter)
        subparser.add_argument("-h", "--help",
                               action=RecipeHelp if name in ("export", "serve") else "help",
                               help="show this help message and exit")
        subparser.set_defaults(handler=handler)
        subparser.add_argument("--config", type=Path, metavar="PATH",
                               help="configuration file (default: ~/.config/flare/config.toml)")
        return subparser

    exporting = command("export", serve.export, "prepare a native model package")
    exporting.add_argument("--output", type=Path, metavar="FILE",
                           help="destination .safetensors file (default: a snapshot in the package cache)")
    exporting.add_argument("--no-weights", action="store_true",
                           help="require cached stored-data blobs; export execution resources and metadata")
    exporting.add_argument("--repo-id", metavar="NAMESPACE/REPOSITORY",
                           help="Hugging Face target used to organize local exports (default: recipe target)")
    _preparation_arguments(exporting)
    serving = command("serve", serve.serve, "prepare in this process and serve over HTTP, with peer ranks on other hosts")
    serving.add_argument("--host", metavar="ADDR", help="listening address (default: server.host, else 127.0.0.1)")
    serving.add_argument("--port", type=int, metavar="N", help="listening port (default: server.port, else 8000)")
    _preparation_arguments(serving)
    for spanning in (serving, exporting):
        hosts = spanning.add_argument_group("multi-host tensor parallelism")
        hosts.add_argument("--roce-ring", action="store_true",
                           help="route collectives and fabric TCP through neighbors in rank order")
        hosts.add_argument("--roce-chain", action="store_true",
                           help="like --roce-ring for hosts cabled as a chain, with no link between the first and"
                                " last rank")
        hosts.add_argument("--transfer-streams", type=int, choices=range(1, 65), metavar="N",
                           help="checkpoint and export fabric TCP streams per peer, 1..64 (default: 8)")
        hosts.add_argument("--ranks", type=int, default=1, choices=range(1, 17), metavar="N",
                           help="tensor-parallel ranks, one per host; this process is rank zero (default: 1);"
                                " an export assembles all ranks at --output on rank zero")
        hosts.add_argument("--rendezvous", metavar="ADDR:PORT",
                           help="IPv4 address and port on which rank zero admits its peers")
    serving.add_argument("--prefix-cache-states", type=int, metavar="N",
                         help="advanced: cached request states kept resident for prefix reuse; fewer leave more"
                              " memory for KV pages (default: sized from free memory)")
    serving.add_argument("--history-drafts", action=argparse.BooleanOptionalAction, default=None,
                         help="augment learned drafts with continuations from shared decoded history;"
                              " requires prepared history inputs (default: off)")
    diagnostics = serving.add_argument_group(
        "diagnostics", "Ignition's direct execution reproduces model-level faults without Helios. It is a"
                       " diagnostic and qualification path, not production serving.")
    diagnostics.add_argument("--direct", action="store_true",
                             help="schedule and pack each step in Python and replay its captured graph on every"
                                  " rank, with no Helios scheduler or native server; non-streaming chat only")
    diagnostics.add_argument("--eager", action="store_true",
                             help="with --direct: call the forward each graph captured instead of replaying it")
    diagnostics.add_argument("--graph-times", type=Path, metavar="PATH",
                             help="with --direct: graph replay saves its estimate of each graph's time here; with"
                                  " --eager, decode rounds are chosen from the times saved here")
    joining = command("join", serve.join, "run a peer rank for rank zero's rendezvous on another host")
    joining.add_argument("rendezvous", metavar="ADDR:PORT", help="rank zero's rendezvous address")
    joining.add_argument("--rank", type=int, required=True, metavar="R", help="this host's rank, 1 through N-1")
    joining.add_argument("--ranks", type=int, required=True, choices=range(2, 17), metavar="N", help="tensor-parallel ranks")
    joining.add_argument("--gpu", type=int, default=0, metavar="INDEX", help="this host's GPU (default: 0)")
    keys = command("keygen", publishing.keygen, "create an encrypted production signing key and public profile")
    keys.add_argument("--directory", required=True, type=Path, help="new private directory; must not exist")
    keys.add_argument("--name", required=True, help="publisher display name")
    keys.add_argument("--namespace", required=True, help="publisher Hugging Face namespace")
    password = command("key-password", publishing.key_password, "change a signing key's passphrase")
    password.add_argument("--signing-key", required=True, type=Path, help="Ed25519 PEM key to re-encrypt in place")
    signing = command("sign", publishing.sign, "verify and sign a development package for production")
    signing.add_argument("package", type=Path, help="signed development .safetensors package")
    signing.add_argument("--output", required=True, type=Path, help="destination package; must not exist")
    signing.add_argument("--signing-key", type=Path, help="Ed25519 PEM key (default: publishing.signing_key)")
    signing.add_argument("--namespace", help="publisher namespace (default: publishing.namespace)")
    signing.add_argument("--official-source", help="published repository URL (default: publishing.official_source)")
    signing.add_argument("--development-publisher", type=Path, help="source public profile (default: beside config)")
    converting = command("convert", publishing.convert, "convert a signed package to an immutable blob snapshot")
    converting.add_argument("package", type=Path, help="existing signed .safetensors package")
    converting.add_argument("--output", type=Path, help="destination primary (default: package cache snapshot)")
    converting.add_argument("--publisher-keys", type=Path, action="append", default=[],
                            help="trusted public publisher profiles; repeatable")
    def shard_size(value):
        from .config import parse_size
        try:
            parse_size(value, "shard size")
        except FlareError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        return value

    for publishing in (exporting, signing, converting):
        publishing.add_argument("--shard-size", type=shard_size, metavar="SIZE",
                                help="maximum payload file size including its header; at most 64G (default: 64G)")
    return parser


def _preparation_arguments(running):
    from .help import recipe_help

    running.usage = "%(prog)s [MODEL] [options]"
    running.epilog = recipe_help()
    running.set_defaults(roce_ring=False, roce_chain=False)
    running.add_argument("--official-source", metavar="URL", help="published checkpoint repository URL")
    running.add_argument("model", nargs="?", metavar="MODEL",
                         help="recipe alias, repository id, or checkpoint directory (default: defaults.model)")
    running.add_argument("--gpus", metavar="LIST",
                         help="GPU indices as NVML enumerates them, such as 0, 0,1, or 0-3;"
                              " their count is the tensor-parallel size (default: first GPUs that fit)")
    running.add_argument("--max-concurrency", type=int, metavar="N",
                         help="maximum simultaneously active requests (default: defaults.max_concurrency,"
                              " else the recipe default)")
    running.add_argument("--verbose", action="store_true", help="per-phase detail on stderr")
    running.add_argument("--image-concurrency", type=int, metavar="N",
                         help="maximum active image requests; excess requests queue independently of text"
                              " (default: smaller of 4 and max concurrency)")
    capacity = running.add_argument_group("capacity and compression")
    capacity.add_argument("--expert-scale-compression", choices=("csf", "native"), nargs="?", const="csf", metavar="MODE",
                          help="losslessly compress routed-expert quantization scales to reduce GPU weight memory."
                               " MODE: csf, a compressed scale encoding expanded during execution."
                               " With no MODE, selects csf; native disables compression."
                               " Omitted flag uses the recipe default."
                               " See MODEL --help for recipe support and estimated savings")
    capacity.add_argument("--prefill-chunk-size", type=int, metavar="N",
                          help="most prompt tokens per prefill step; larger chunks reach a long prompt's first"
                               " token sooner and pause other requests longer; a multiple of 64"
                               " (default: see MODEL --help)")
    capacity.add_argument("--prompt-logprobs", action="store_true",
                          help="capture dedicated prefill graphs for raw prompt token logprobs")
    capacity.add_argument("--mixed-prefill", action="store_true",
                          help="capture graphs combining prompt prefill with decode or verification"
                               " (Qwen only; default: disabled)")
    speculative = running.add_argument_group("speculative decoding",
        "MTP flags configure speculative decoding, including DSpark and DFlash.\n"
        "Recipe defaults and limits: ignition export MODEL --help.")
    speculative.add_argument("--no-mtp", action="store_true",
                          help="disable multi-token speculation: load no draft model and decode one token"
                               " per step; cannot accompany other speculation flags")
    speculative.add_argument("--mtp-draft-tokens", type=int, metavar="N",
                          help="most draft tokens verified per round; limits and defaults depend on MODEL."
                               " 0 selects width-one rounds while retaining draft state; --no-mtp excludes it")
    speculative.add_argument("--no-mtp-adaptive-verification", action="store_true",
                          help="default to the maximum verification width where captured, otherwise width one")
    speculative.add_argument("--speculative-token-capacity", type=int, metavar="N",
                          help="maximum target-token rows in wider speculative graphs; width one covers every"
                               " request batch. Rectangular graphs require batch * width <= N, where width is"
                               " drafts + 1; packed rounds share N rows across request prefixes"
                               " (default: see MODEL --help)")
    speculative.add_argument("--speculative-graph-budget", type=int, metavar="N",
                          help="maximum captured packed round graphs, independently of target-token rows;"
                               " excludes mandatory width-one and ordinary graphs. Requires packed adaptive"
                               " learned speculation (default: see MODEL --help)")
    advanced = running.add_argument_group("advanced options")
    advanced.add_argument("--draft-head-quantization", choices=("nvfp4",),
                          help="prepare a separate NVFP4 vocabulary head for learned draft proposals;"
                               " target verification retains its precision (DeepSeek V4/V4.1)")
    advanced.add_argument("--markov-head-quantization", choices=("nvfp4",),
                          help="prepare the draft Markov transition projection as NVFP4"
                               " (DeepSeek V4/V4.1; default: checkpoint precision)")
    advanced.add_argument("--mla-compute-mode", choices=("auto", "bf16"),
                          help="constrain MLA arithmetic while retaining packed KV storage and compatible"
                               " kernel tuning (default: see MODEL --help)")
    from .config import AUTOTUNE_EFFORTS

    advanced.add_argument("--autotune-effort", choices=AUTOTUNE_EFFORTS,
                          help="kernel search effort: off uses heuristics; low applies broad exclusions;"
                               " medium balances search cost; high keeps conservative exclusions;"
                               " max searches all legal candidates (default: medium)")
    advanced.add_argument("--compile-workers", type=int, metavar="N",
                          help="CPU compiler process budget per host, shared by local preparation ranks;"
                               " bounded by available RAM and device memory (default: automatic)")
    advanced.add_argument("--tuning-gpus", metavar="LIST",
                          help="local GPUs for kernel tuning, starting with --gpus and followed by complete"
                               " TP replicas; serving TP size is unchanged (default: serving GPUs)")
    advanced.add_argument("--no-context-parallel", action="store_true",
                          help="keep each rank's attention context whole instead of sharding it across"
                               " tensor-parallel ranks; fewer cached tokens (default: the recipe's)")
    advanced.add_argument("--no-stream-overlap", action="store_true",
                          help="run every concurrent branch (shared experts, projection chains, query chains)"
                               " serially on the main stream, to diagnose scheduling-dependent faults; slower"
                               " (default: overlap)")
    advanced.add_argument("--no-compile", action="store_true",
                          help="development: capture graphs without compiling forwards")
    advanced.add_argument("--online-mxfp8", action="store_true",
                          help="store the recipe's eligible BF16 projections as MXFP8 while loading; faster"
                               " decode at reduced precision (Kimi K3 KDA, MLA except batched key/value, shared"
                               " experts, MoE latent projections, and DSpark draft context, outputs and MLPs;"
                               " default: checkpoint precision)")
    advanced.add_argument("--language-model-only", action="store_true",
                          help="prepare no vision encoder and reject images; its weights and scratch become KV"
                               " pages (Kimi K3; default: the recipe's image support)")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args) or 0
    except FlareError as error:
        from . import display

        display.error(str(error))
        return error.exit_code
