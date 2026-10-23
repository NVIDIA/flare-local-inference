"""Diagnostic host uploads from the same semantic bindings consumed by Helios."""

import numpy as np

from ignition.preparation import runtime_package_pb2 as p


def pack(layout, step):
    uploads = []
    for binding in layout.bindings[step.graph]:
        name, source, indexing = binding["buffer"], binding["source"], binding["indexing"]
        dtype, shape = layout.buffer_specs[name]
        width = int(np.prod(shape if indexing == p.SCALAR else shape[1:]))
        def put(offset, values):
            array = np.asarray(values, dtype=dtype)
            if array.size:
                uploads.append((name, offset * array.itemsize, array.tobytes()))
        if indexing == p.SCALAR:
            value = len(step.lanes) if source == p.LIVE_ROWS else 0 if source == p.CONSTANT else -1 if source == p.IMAGE_FEATURE_SLOT else None
            if value is None:
                raise ValueError(f"direct text execution cannot produce scalar source {source}")
            put(0, [value + binding["value_offset"]])
            continue
        padded = [binding["padding"]] * (layout.capacity * width) if indexing == p.SUBMISSION_ROW else None
        offset = 0
        for row, lane in enumerate(step.lanes):
            request = lane.request
            destination = (request.slot if indexing == p.REQUEST_ROW else offset if indexing == p.PACKED_TOKEN else row) * width
            offset += lane.tokens
            live_row = layout.state_row(request.slot)
            if source == p.CONSTANT: values = [0] * width
            elif source == p.REQUEST_SLOT: values = [request.slot]
            elif source == p.REQUEST_GENERATION: values = [request.generation]
            elif source in (p.QUERY_LENGTH, p.SPECULATIVE_PREFIX_LIMIT): values = [lane.tokens]
            elif source == p.SAMPLE_ENABLED: values = [int(step.kind != "prefill" or lane.start + lane.tokens == len(request.prompt))]
            elif source == p.PROMPT_TOKEN: values = request.prompt[lane.start:lane.start + lane.tokens]
            elif source == p.PENDING_TOKEN: values = [request.pending_token]
            elif source == p.PROMPT_POSITION: values = np.repeat(np.arange(lane.start, lane.start + lane.tokens), width).tolist()
            elif source == p.IMAGE_FEATURE_INDEX: values = [-1] * lane.tokens
            elif source in (p.IMAGE_LEFT_CONTEXT, p.IMAGE_RIGHT_CONTEXT): values = [0] * lane.tokens
            elif source == p.PAGE_TABLE:
                shift = layout.page_views[name]
                values = request.pages[shift:shift + width]
            elif source == p.LIVE_STATE_ROWS: values = [live_row + column if column < layout.execution_rows else 0 for column in range(width)]
            elif source == p.LIVE_STATE_ROW: values = [live_row]
            elif source == p.PREFIX_STATE_ROW: values = [layout.snapshot_row(request.prefix.snapshot)]
            elif source == p.SNAPSHOT_STATE_ROW: values = [layout.snapshot_row(request.capture)]
            elif source in (p.CONSUMED_TOKENS, p.SNAPSHOT_CONSUMED_TOKENS): values = [request.consumed]
            elif source == p.SAMPLING_SEED: values = [request.sampling.seed if request.sampling.seed < 1 << 63 else request.sampling.seed - (1 << 64)]
            elif source == p.SAMPLING_PARAMETERS: values = request.sampling.values()
            elif source == p.SAMPLING_TOP_K: values = [request.sampling.top_k]
            elif source == p.MAX_NEW_TOKENS: values = [request.max_new_tokens]
            elif source == p.EOS_TOKEN_IDS: values = list(layout.eos) + [-1] * (width - len(layout.eos))
            elif source == p.EOS_COUNT: values = [0 if request.ignore_eos else len(layout.eos)]
            elif source == p.PROMPT_HISTOGRAM: values = np.bincount(np.asarray(request.prompt), minlength=width)[:width].tolist()
            elif source == p.GRAMMAR_MASK: values = [-1] * width
            elif source in (p.GRAMMAR_ENABLED, p.COPY_TOKEN_COUNT, p.COPY_GENERATION, p.COPY_GRAMMAR_COUNT): values = [0]
            elif source in (p.COPY_TOKEN_IDS, p.COPY_CONSUMED_TOKENS): values = [-1] * width
            elif source == p.COPY_GRAMMAR_MASK: values = []
            else: raise ValueError(f"direct text execution cannot produce source {source}")
            values = [v + binding["value_offset"] for v in values]
            if destination + len(values) > np.prod(shape):
                raise ValueError(f"semantic values exceed prepared buffer {name}")
            if padded is None: put(destination, values)
            else: padded[destination:destination + len(values)] = values
        if padded is not None: put(0, padded)
    return uploads
