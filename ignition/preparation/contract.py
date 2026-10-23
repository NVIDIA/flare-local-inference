"""Semantic host uploads and persistent-state dependencies of prepared programs."""

from . import runtime_package_pb2 as p


def binding(buffer, source, indexing=p.SUBMISSION_ROW, *, offset=0, padding=0, tensor=""):
    return dict(buffer=buffer, source=source, indexing=indexing, value_offset=offset,
                padding=padding, tensor=tensor)


def state_view(buffer, stride, *, offset=0, nbytes=None):
    return dict(buffer=buffer, offset=offset, stride=stride, nbytes=stride if nbytes is None else nbytes)


def text_inputs(kind):
    """Uploads for the shared admission, paged-text, and state-transfer components."""
    b = binding
    values = [b("request_state_rows", p.LIVE_STATE_ROWS, p.REQUEST_ROW)]
    if kind not in ("init", "restore"):
        values += [b("row_slot_ids", p.REQUEST_SLOT, padding=-1),
                   b("row_generations", p.REQUEST_GENERATION), b("live_rows", p.LIVE_ROWS, p.SCALAR)]
    if kind != "resume":
        values += [b("page_tables", p.PAGE_TABLE, p.REQUEST_ROW),
                   b("draft_page_tables", p.PAGE_TABLE, p.REQUEST_ROW)]
    if kind in ("prefill", "scored_prefill", "decode", "speculative", "proposal", "verification", "packed_speculative", "mixed", "resume"):
        values += [b("sampling_grammar_mask", p.GRAMMAR_MASK, p.REQUEST_ROW)]
        if kind != "resume":
            values += [b("sample_enabled", p.SAMPLE_ENABLED)]
            values += ([b("speculative_prefix_limits", p.SPECULATIVE_PREFIX_LIMIT)]
                       if kind == "packed_speculative" else [b("query_lengths", p.QUERY_LENGTH)])
    if kind in ("prefill", "scored_prefill", "mixed"):
        values += [b("token_ids", p.PROMPT_AND_PENDING_TOKEN if kind == "mixed" else p.PROMPT_TOKEN, p.PACKED_TOKEN),
                   b("prefill_positions", p.PROMPT_POSITION, p.PACKED_TOKEN),
                   b("vision_token_indices", p.IMAGE_FEATURE_INDEX, p.PACKED_TOKEN),
                   b("vision_feature_slot", p.IMAGE_FEATURE_SLOT, p.SCALAR, padding=-1),
                   b("vision_left", p.IMAGE_LEFT_CONTEXT, p.PACKED_TOKEN),
                   b("vision_right", p.IMAGE_RIGHT_CONTEXT, p.PACKED_TOKEN)]
    if kind == "scored_prefill":
        values += [b("prompt_target_tokens", p.PROMPT_TARGET_TOKEN, p.PACKED_TOKEN, padding=-1)]
    if kind in ("decode", "speculative", "proposal", "verification", "packed_speculative"):
        values += [b("token_ids", p.PENDING_TOKEN)]
    if kind == "mixed":
        values += [b("prefill_row", p.PREFILL_ROW, p.SCALAR)]
    if kind in ("speculative", "mixed"):
        for name, source in (("copy_token_ids", p.COPY_TOKEN_IDS), ("copy_token_count", p.COPY_TOKEN_COUNT),
                             ("copy_expected_consumed", p.COPY_CONSUMED_TOKENS), ("copy_generation", p.COPY_GENERATION),
                             ("copy_grammar_mask", p.COPY_GRAMMAR_MASK), ("copy_grammar_mask_count", p.COPY_GRAMMAR_COUNT)):
            values.append(b(name, source, p.REQUEST_ROW))
    if kind in ("init", "restore", "snapshot"):
        source = b("state_source_slots", p.CONSTANT, offset=-1, padding=-1) if kind == "init" else b(
            "state_source_slots", p.PREFIX_STATE_ROW if kind == "restore" else p.LIVE_STATE_ROW, padding=-1)
        values += [source, b("state_destination_slots", p.SNAPSHOT_STATE_ROW if kind == "snapshot" else p.LIVE_STATE_ROW, padding=-1),
                   b("state_request_slots", p.REQUEST_SLOT, padding=-1),
                   b("state_operation", p.CONSTANT, offset=int(kind == "init")),
                   b("state_expected_consumed", p.SNAPSHOT_CONSUMED_TOKENS if kind == "snapshot" else p.CONSUMED_TOKENS, padding=-1),
                   b("state_num_ops", p.LIVE_ROWS, p.SCALAR)]
    if kind in ("init", "restore"):
        values += [b("admission_count", p.LIVE_ROWS, p.SCALAR), b("admission_slots", p.REQUEST_SLOT, padding=-1),
                   b("admission_generations", p.REQUEST_GENERATION)]
        for name, source in (("seed_bits", p.SAMPLING_SEED), ("parameters", p.SAMPLING_PARAMETERS),
                             ("top_k", p.SAMPLING_TOP_K), ("max_new_tokens", p.MAX_NEW_TOKENS),
                             ("eos_token_ids", p.EOS_TOKEN_IDS), ("eos_count", p.EOS_COUNT),
                             ("histogram", p.PROMPT_HISTOGRAM), ("grammar_mask", p.GRAMMAR_MASK),
                             ("grammar_enabled", p.GRAMMAR_ENABLED)):
            values.append(b("admission_" + name, source, p.ACTIVE_ROW))
    return {v["buffer"]: v for v in values}


def graph_contracts(signatures, inputs, *, continuation, pending, cursor=None, rows=1, embedding_tables=()):
    """Bind semantic uploads to the captured catalog; omitted transfers retain device bytes."""
    result = []
    for name, signature in signatures.items():
        kind = signature.kind
        selected = inputs[kind]
        values = [selected[field] for field in signature.inputs if field in selected]
        states = [] if kind == "vision" else [dict(view=continuation, access=p.READ if kind == "proposal" else p.READ_WRITE)]
        reads_pending = kind in ("decode", "speculative", "proposal", "verification", "packed_speculative", "mixed") and not any(
            v["source"] in (p.PENDING_TOKEN, p.PROMPT_AND_PENDING_TOKEN) for v in values)
        if pending is not None and kind not in ("snapshot", "vision"):
            states.append(dict(view=pending, access=p.WRITE if kind in ("init", "restore") else p.READ if kind == "proposal" else p.READ_WRITE))
        result.append(dict(graph=name, kind=p.GraphOperation.Value(kind.upper()), inputs=values, state=states,
            state_rows_per_request=signature.token_capacity if rows is None and kind in ("speculative", "verification", "packed_speculative") else (rows or 1),
            draft_cursor=cursor if signature.speculation_enabled and kind in ("speculative", "verification", "packed_speculative", "mixed") else None,
            embedding_tables=list(embedding_tables) if kind in (
                "prefill", "scored_prefill", "decode", "speculative", "verification", "packed_speculative", "mixed") else [],
            pending_token=pending if reads_pending else None))
    return result


def frontend(*, tools, assistant_marker, message_end_markers, prefix_boundary, wrapping_newlines=0,
             tag_separator="", terminal_markers=(), image_preparation=None, image_layout=None,
             position_dimensions=0, rejected_tokens=(), image_tokens=None):
    return dict(tools=dict(kind=tools, wrapping_newlines=wrapping_newlines, tag_separator=tag_separator),
        reasoning=dict(kind=p.THINK_TAGS, open="<think>", close="</think>",
                       wrapping_newlines=wrapping_newlines, terminal_markers=list(terminal_markers)),
        image_preparation=image_preparation, image_layout=image_layout, position_dimensions=position_dimensions,
        rejected_tokens=list(rejected_tokens), assistant_marker=assistant_marker,
        message_end_markers=list(message_end_markers), prefix_boundary=prefix_boundary, image_tokens=image_tokens)


def prefix_boundary(roles, *, append_role=None, minimum_messages=1, required_start="", required_end="", required_marker=""):
    return dict(roles=list(roles), append_role=append_role, continue_final_message=append_role is not None,
        minimum_messages=minimum_messages, required_start=required_start, required_end=required_end,
        required_marker=required_marker)


def model_contract(*, vocabulary, input_vocabulary=None, vision=None, page_tokens, page_tables, graphs, frontend,
                   regions):
    return dict(version=1, vocabulary=vocabulary, input_vocabulary=input_vocabulary or vocabulary,
        vision={"feature_slots": 0, **(vision or dict(features_per_request=0, max_images=0, patches_per_feature=0, patch_capacities=[]))},
        cache=dict(page_tokens=page_tokens, page_tables=page_tables, regions=regions), graphs=graphs, frontend=frontend)


def vision_inputs(fields):
    return {name: binding(name, p.REQUEST_SLOT if name == "vision_slot" else
                         p.REQUEST_GENERATION if name == "vision_generation" else
                         p.IMAGE_FEATURE_SLOT if name == "vision_feature_slot" else p.IMAGE_TENSOR,
                         p.SCALAR, tensor=name) for name in fields}


def history_region(buffers, name, tensor, kind, *, tokens_per_record=1, window_tokens=0):
    """Register a physical history plane with one leading row per cache unit."""
    buffers[name] = tensor
    stride = tensor.stride(0) * tensor.element_size()
    return dict(kind=kind, storage=state_view(name, stride),
                tokens_per_record=tokens_per_record, window_tokens=window_tokens)
