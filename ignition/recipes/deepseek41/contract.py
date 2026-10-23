"""Prepared graph inputs, continuation storage, and frontend syntax."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p


VOCABULARY = 129280
INPUT_VOCABULARY = 129284
PATCH_CAPACITIES = (576, 1152, 2304, 4608, 9216)
VISION = dict(features_per_request=4096, max_images=4, patches_per_feature=9, patch_capacities=PATCH_CAPACITIES)
TEXT_VISION = dict(features_per_request=0, max_images=0, patches_per_feature=0, patch_capacities=[])
FRONTEND = c.frontend(tools=p.DSML, tag_separator=" ", assistant_marker="<｜Assistant｜>",
    message_end_markers=["<｜User｜>", "<｜end▁of▁sentence｜>"], rejected_tokens=[129264],
    prefix_boundary=c.prefix_boundary(["system"]))
# Internal input IDs distinguish image types without changing the model vocabulary.
# Every image position receives a learned delimiter or a feature before the backbone.
VISION_FRONTEND = c.frontend(tools=p.DSML, tag_separator=" ", assistant_marker="<｜Assistant｜>",
    message_end_markers=["<｜User｜>", "<｜end▁of▁sentence｜>"],
    prefix_boundary=c.prefix_boundary(["system"]),
    image_tokens=dict(feature_token=129281, span=dict(start_token=129280, end_token=129283,
        max_tokens=1024, interior_tokens=[129282], unscoped_tokens=[])),
    image_preparation=p.DEEPSEEK_ROW_MAJOR_IMAGE_PREPARATION, image_layout=p.DEEPSEEK_ROW_MAJOR_IMAGE_LAYOUT)


def prepare(signatures, buffers, pool, *, image_slots=4, page_tokens, draft_lag, rows=1, embedding_tables=(), regions=()):
    regions = list(regions)
    for name, tensor in pool.pages.items():
        layer = int(name.split(".")[0].removeprefix("layer"))
        regions.append(c.history_region(buffers, "cache." + name, tensor, p.PAGED_LATENT,
            tokens_per_record=pool.config.text["compress_ratios"][layer]))
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != "vision"}
    inputs["vision"] = c.vision_inputs({field for s in signatures.values() if s.kind == "vision" for field in s.inputs})
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = None if draft_lag is None else dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=draft_lag)
    pending = c.state_view("request_pending_token", 8)
    graphs = c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending,
        cursor=cursor, rows=rows, embedding_tables=embedding_tables)
    for graph in graphs:
        if graph["kind"] in (p.PROPOSAL, p.VERIFICATION, p.PACKED_SPECULATIVE):
            graph["state"].extend(dict(view=c.state_view(name, tensor.stride(0) * tensor.element_size()),
                access=p.READ_WRITE) for name, tensor in buffers.items() if name.startswith("proposal_"))
    vision = any(s.kind == "vision" for s in signatures.values())
    return c.model_contract(vocabulary=VOCABULARY, input_vocabulary=INPUT_VOCABULARY if vision else VOCABULARY,
        page_tokens=page_tokens, page_tables=[dict(buffer="page_tables", page_offset=0),
            *([dict(buffer="draft_page_tables", page_offset=1)] if "draft_page_tables" in buffers else [])],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=graphs, vision={**VISION, "feature_slots": image_slots} if vision else TEXT_VISION,
        frontend=VISION_FRONTEND if vision else FRONTEND)
