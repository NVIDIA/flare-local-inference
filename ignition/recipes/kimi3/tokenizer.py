"""Convert K3's byte-ranked vocabulary to the native Hugging Face tokenizer."""

import base64
import json
from pathlib import Path

from tokenizers import AddedToken
from transformers.convert_slow_tokenizer import TikTokenConverter


PATTERN = "|".join((
    r"[\p{Han}]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n]*", r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
))


class _Converter(TikTokenConverter):
    @staticmethod
    def load_tiktoken_bpe(path):
        ranks = {}
        for line in Path(path).read_bytes().splitlines():
            token, rank = line.split()
            token, rank = base64.b64decode(token, validate=True), int(rank)
            if token in ranks or rank < 0:
                raise ValueError("duplicate or invalid K3 vocabulary entry")
            ranks[token] = rank
        if set(ranks.values()) != set(range(len(ranks))):
            raise ValueError("K3 vocabulary ranks must be contiguous")
        return ranks


def convert(snapshot):
    snapshot = Path(snapshot)
    config = json.loads((snapshot / "tokenizer_config.json").read_text())
    converter = _Converter(vocab_file=str(snapshot / "tiktoken.model"), pattern=PATTERN)
    count = len(converter.load_tiktoken_bpe(converter.vocab_file))
    if count != 163584:
        raise ValueError("K3 requires 163584 byte-ranked tokens and 256 reserved tokens")
    decoder = config["added_tokens_decoder"]
    special = [decoder.get(str(i), {"content": f"<|reserved_token_{i}|>"})["content"]
               for i in range(count, count + 256)]
    if len(set(special)) != len(special):
        raise ValueError("K3 reserved tokens must be distinct")
    tokenizer = converter.converted()
    tokenizer.add_tokens([AddedToken(token, normalized=False,
        special=decoder.get(str(index), {}).get("special", False))
        for index, token in enumerate(special, count)])
    for index, token in enumerate(special, count):
        if tokenizer.token_to_id(token) != index:
            raise ValueError(f"K3 tokenizer does not preserve token ID {index}")
    return tokenizer


def runtime_assets(tokenizer):
    """Describe the converted vocabulary and bundled template without creating files."""
    return {"tokenizer": {"text": tokenizer.to_str(pretty=True)},
            "chat_template": {"path": str(Path(__file__).with_name("assets") / "chat_template.jinja")}}
