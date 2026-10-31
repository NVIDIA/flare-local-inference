# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation policy for the Qwen 3.8 operator catalog."""

from ignition.preparation.host_memory import COMPILE_HEADROOM_BYTES
from ignition.preparation.tuning import RecordedPreparationSession


def preparation_session(*, device, checkpoint, namespace=None, coordinator=None, headroom_bytes=0,
                        benchmark_storage=None, compile_workers=None, tuning_replica=None, autotune_effort="default", collective_tuning="static", collective_ladder=None,
                        native_dense_activations=False, lock_gpu_clocks=False,
                        compile_headroom_bytes=COMPILE_HEADROOM_BYTES):
    return RecordedPreparationSession(
        device=device, checkpoint=checkpoint, compile_workers=compile_workers, tuning_replica=tuning_replica,
        collective_tuning=collective_tuning, collective_ladder=collective_ladder, autotune_effort=autotune_effort, lock_gpu_clocks=lock_gpu_clocks, native_dense_activations=native_dense_activations,
        namespace=namespace, coordinator=coordinator,
        headroom_bytes=headroom_bytes, compile_headroom_bytes=compile_headroom_bytes,
        benchmark_storage=benchmark_storage,
        workload_schema="qwen38-active-operators-v1",
        numerical_policy="qsa16-gdn-sequential-k1-dense-mxfp8-auto-vision-nvfp4-a16-moe-a4-route-fp32",
        search_limits={
            "bf16_projection": "all eligible b12x backends and rows_per_tile configurations",
            "mxfp8_projection": "BF16 or quantized activations with eligible b12x projection configurations",
            "qsa": "fixed attention_splits=16; eligible direct-KV warp configurations",
            "gdn_prefill": "sequential k_split=1; v_split, stages, and window_tiles remain tunable",
            "operator_scratch": "one arena sized to the complete candidate envelope",
        },
    )
