"""Typed M3 graph inputs, complete continuations, and GQA/index page bindings."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p

from .config import PAGE_TOKENS


def prepare(signatures, buffers, pool, *, vocabulary, frontend, image_slots=4, dspark=False):
    kinds = {signature.kind for signature in signatures.values()}
    inputs = {kind: c.text_inputs(kind) for kind in kinds if kind != "vision"}
    if "vision" in kinds:
        inputs["vision"] = c.vision_inputs({field for signature in signatures.values()
            if signature.kind == "vision" for field in signature.inputs})
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=0) if dspark else None
    regions = [c.history_region(buffers, "cache." + name, tensor,
        p.PAGED_LATENT if name.endswith(".index") else p.PAGED_KEY_VALUE)
        for name, tensor in pool.pages.items()]
    graphs = c.graph_contracts(signatures, inputs, continuation=continuation,
        pending=c.state_view("request_pending_token", 8), cursor=cursor)
    for graph in graphs:
        if graph["kind"] in (p.PROPOSAL, p.VERIFICATION, p.PACKED_SPECULATIVE):
            graph["state"].extend(dict(view=c.state_view(name, tensor.stride(0) * tensor.element_size()),
                access=p.READ_WRITE) for name, tensor in buffers.items() if name.startswith("proposal_"))
    vision = None
    if "vision" in kinds:
        from .vision import FEATURES, PATCH_CAPACITIES

        vision = dict(feature_slots=image_slots, features_per_request=FEATURES,
            max_images=4, patches_per_feature=4, patch_capacities=list(PATCH_CAPACITIES))
    else:
        frontend = {name: value for name, value in frontend.items()
            if name not in ("image_tokens", "image_preparation", "image_layout")}
    return c.model_contract(vocabulary=vocabulary, page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=graphs, vision=vision, frontend=frontend)
