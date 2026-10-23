"""Gemma 4 text inputs, continuation state, and paged attention storage."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p

from .config import PAGE_TOKENS


FRONTEND = c.frontend(
    tools=p.GEMMA_CALL,
    assistant_marker="<|turn>model\n",
    message_end_markers=["<turn|>", "<|tool_response>", "<eos>"],
    prefix_boundary=c.prefix_boundary(
        ["system", "developer"],
        append_role="user",
        minimum_messages=0,
        required_start="<bos><|turn>system\n",
        required_marker="<turn|>",
    ),
)
FRONTEND["reasoning"] = dict(
    kind=p.CHANNEL_THOUGHT,
    open="<|channel>thought",
    close="<channel|>",
    wrapping_newlines=1,
    terminal_markers=["<|tool_call>"],
)


def prepare(signatures, buffers, pool, *, vocabulary, world_size: int = 1, mtp: bool = False):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()}}
    if world_size == 1 and "decode" in inputs:
        inputs["decode"].pop("token_ids", None)
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = (
        dict(
            state=c.state_view(
                "state_pool", pool.layout.row_bytes, offset=consumed.offset, nbytes=consumed.nbytes
            ),
            lag_tokens=0,
        )
        if mtp
        else None
    )
    pending = c.state_view("request_pending_token", 8)
    regions = [
        c.history_region(buffers, f"cache.{name}", tensor, p.PAGED_KEY_VALUE)
        for name, tensor in pool.pages.items()
    ]
    return c.model_contract(
        vocabulary=vocabulary,
        page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[
            dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0),
            *regions,
        ],
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending, cursor=cursor),
        frontend=FRONTEND,
    )
