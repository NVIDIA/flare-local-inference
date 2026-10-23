"""Prepared graph inputs, continuation storage, and frontend syntax."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p


VOCABULARY = 154880
INPUT_VOCABULARY = 154880
VISION = dict(features_per_request=8192, max_images=4, patches_per_feature=4,
              patch_capacities=[64, 256, 1024, 4096, 16384, 32768])
FRONTEND = c.frontend(tools=p.GLM_ARGUMENT_TAGS, assistant_marker="<|assistant|>",
    rejected_tokens=[*range(154830, 154836), 154854, 154855],
    message_end_markers=["<|user|>", "<|system|>", "<|observation|>"],
    prefix_boundary=c.prefix_boundary(["system"], append_role="user", minimum_messages=0,
        required_start="[gMASK]<sop><|system|>", required_end="<|user|>"))


def prepare(signatures, buffers, pool, *, page_tokens, draft_lag, rows=1, embedding_tables=(), regions=(),
            image_slots=4):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != "vision"}
    inputs["vision"] = c.vision_inputs({field for s in signatures.values() if s.kind == "vision" for field in s.inputs})
    vision = None
    frontend = FRONTEND
    if any(s.kind == "vision" for s in signatures.values()):
        vision = {**VISION, "feature_slots": image_slots}
        frontend = {**FRONTEND, "rejected_tokens": [*range(154832, 154836), 154855],
            "image_preparation": p.GLM_IMAGE_PREPARATION, "image_layout": p.GLM_IMAGE_LAYOUT,
            "image_tokens": dict(feature_token=154854, span=None)}
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = None if draft_lag is None else dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=draft_lag)
    pending = c.state_view("request_pending_token", 8)
    graphs = c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending,
        cursor=cursor, rows=rows, embedding_tables=embedding_tables)
    return c.model_contract(vocabulary=VOCABULARY, input_vocabulary=INPUT_VOCABULARY,
        page_tokens=page_tokens, page_tables=[dict(buffer="page_tables", page_offset=0),
            *([dict(buffer="draft_page_tables", page_offset=1)] if "draft_page_tables" in buffers else [])],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=graphs, vision=vision, frontend=frontend)
