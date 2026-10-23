//! Shared validation and selection for every prepared model's graph catalog.

use crate::{
    Result,
    execution::{
        BATCH_CAPACITIES, ExecutionManifest, GraphKind, SPECULATIVE_ABI_VERSION, SamplingMode,
    },
    model::ModelConfig,
    scheduler::speculation,
};

pub fn validate(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
    validate_inner(config, manifest).map_err(|error| match error {
        crate::Error::Execution(crate::ExecutionError::Manifest(_)) => error,
        _ => crate::ExecutionError::Manifest(crate::Failure::caused_by(error.to_string(), error))
            .into(),
    })
}

fn validate_inner(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
    manifest.validate()?;
    if config.max_top_logprobs != manifest.max_top_logprobs()?
        || config.logprob_token_id_capacity != manifest.logprob_token_id_capacity()
    {
        return Err("model logprob capacity differs from its declared output buffers".into());
    }
    config.contract.validate(manifest)?;
    config.sampling_ranges.validate(&config.sampling_defaults)?;
    if config
        .sampling_ranges
        .top_k
        .is_some_and(|range| range.max > config.contract.vocabulary as i32)
    {
        return Err("sampling top_k range exceeds the prepared vocabulary".into());
    }
    config.limits.validate_prefill()?;
    let proposals: Vec<_> = manifest
        .graphs
        .iter()
        .filter(|g| g.kind == GraphKind::Proposal)
        .collect();
    let split = !proposals.is_empty();
    let packed_round = manifest.graphs.iter().any(|g| g.kind == GraphKind::PackedSpeculative);
    if packed_round && (config.history_drafts
        || config.speculative_proposal_source != speculation::ProposalSource::Mtp) {
        return Err("packed speculative rounds require learned fresh proposals".into());
    }
    if config.speculative_fixed_width.is_some() && manifest.graphs.iter().any(|g| g.packed_token_capacity != 0) {
        return Err("packed verification catalogs require adaptive prefix selection".into());
    }
    if split {
        let maximum = proposals.iter().map(|g| g.token_capacity).max().unwrap();
        if config.history_drafts
            || config.speculative_proposal_source != speculation::ProposalSource::Mtp
            || config.speculative_widths != (1..=*config.speculative_widths.last().unwrap_or(&0)).collect::<Vec<_>>()
            || maximum > *config.speculative_widths.last().unwrap_or(&0)
        {
            return Err("retained proposal catalogs require contiguous prefix widths and proposal-free width-one execution".into());
        }
    } else if manifest
        .graphs
        .iter()
        .any(|g| g.kind == GraphKind::Verification)
    {
        return Err("verification graphs require a retained proposal graph".into());
    }
    let page_tokens = crate::model::scheduler_capabilities(config).page_tokens;
    if config.model_id.is_empty()
        || config.limits.max_active == 0
        || config.limits.max_active > manifest.max_active
        || config.limits.request_slots != manifest.request_capacity()
        || config.limits.max_sequence_tokens == 0
        || config.limits.max_sequence_tokens > config.prepared_sequence_tokens
        || !config
            .prepared_sequence_tokens
            .is_multiple_of(page_tokens.max(1))
        || page_tokens == 0
        || !config
            .limits
            .max_sequence_tokens
            .is_multiple_of(page_tokens)
        || config.limits.pages == 0
        || config.limits.pages > i32::MAX as u32
        || config.limits.snapshots > i32::MAX as u32 - (config.limits.request_capacity() + 1) as u32
        || config.max_new_tokens == 0
        || config.max_new_tokens > i32::MAX as u32
        || config.sampling_defaults.top_k > config.contract.vocabulary as i32
        || config.eos_token_ids.len() > 8
        || config
            .eos_token_ids
            .iter()
            .any(|id| *id as usize >= config.contract.vocabulary)
    {
        return Err("invalid prepared model configuration".into());
    }
    speculation::validate_widths(&config.speculative_widths)?;
    if config.speculative_widths.last().is_some_and(|w| config.speculative_token_capacity < *w) {
        return Err("speculative token capacity must cover the maximum verification width".into());
    }
    if config
        .speculative_fixed_width
        .is_some_and(|w| !config.speculative_widths.contains(&w))
    {
        return Err(
            "speculative_fixed_width must belong to the prepared speculative_widths".into(),
        );
    }
    if config.speculative_widths.is_empty() {
        if config.speculative_proposal_source == speculation::ProposalSource::Copy {
            return Err("copy proposal source requires speculative graphs".into());
        }
    } else if manifest.abi_version != SPECULATIVE_ABI_VERSION {
        return Err("speculative catalog requires ABI3".into());
    }
    for graph in &manifest.graphs {
        if matches!(graph.kind, GraphKind::Speculative | GraphKind::Mixed)
            && graph.token_capacity > 1
            && (config.history_drafts
                || config.speculative_proposal_source == speculation::ProposalSource::Copy)
        {
            use super::contract::ValueSource::*;
            let inputs = &config.contract.graph(&graph.name)?.inputs;
            let mut required = vec![
                CopyTokenIds,
                CopyTokenCount,
                CopyConsumedTokens,
                CopyGeneration,
            ];
            if config.speculative_proposal_source == speculation::ProposalSource::Copy {
                required.extend([CopyGrammarMask, CopyGrammarCount]);
            }
            if required
                .iter()
                .any(|source| !inputs.iter().any(|input| input.source == *source))
            {
                return Err(
                    "copied proposals require prepared token, generation, cursor, and mask inputs"
                        .into(),
                );
            }
        }
        select_shape_with_drafts(
            config,
            manifest,
            graph.kind,
            graph.batch_capacity,
            graph.token_capacity,
            graph.prefill_capacity,
            graph.packed_token_capacity,
            graph.draft_tokens,
            graph.sampling_mode,
        )?;
    }
    for kind in [
        GraphKind::Init,
        GraphKind::Restore,
        GraphKind::Snapshot,
        GraphKind::Resume,
        GraphKind::Decode,
    ] {
        if kind == GraphKind::Decode && !config.speculative_widths.is_empty() {
            continue;
        }
        for batch in batches(config) {
            select(config, manifest, kind, batch, 1, 0, SamplingMode::General)?;
        }
    }
    select(
        config,
        manifest,
        GraphKind::Prefill,
        1,
        config.limits.prefill_chunk_size,
        0,
        SamplingMode::General,
    )?;
    for &tokens in &config.contract.vision.patch_capacities {
        select(
            config,
            manifest,
            GraphKind::Vision,
            1,
            tokens,
            0,
            SamplingMode::General,
        )?;
    }
    if !config.speculative_widths.is_empty() {
        for batch in batches(config) {
            if split {
                let proposal_width = proposals.iter().filter(|g| g.batch_capacity == batch)
                    .map(|g| g.token_capacity).max()
                    .ok_or_else(|| format!("retained proposal catalog lacks batch {batch}"))?;
                select(config, manifest, GraphKind::Proposal, batch,
                    proposal_width, 0, SamplingMode::General)?;
                select(config, manifest, GraphKind::Speculative, batch, 1, 0, SamplingMode::General)?;
            }
            if packed_round && !split {
                select(config, manifest, GraphKind::Speculative, batch, 1, 0, SamplingMode::General)?;
                if batch * 2 <= config.speculative_token_capacity
                    && !manifest.graphs.iter().any(|g| g.kind == GraphKind::PackedSpeculative && g.batch_capacity == batch) {
                    return Err(format!("packed speculative catalog lacks batch {batch}").into());
                }
                continue;
            }
            for &width in &config.speculative_widths {
                if speculative_shape(config, batch, width) {
                    select(config, manifest, if split { GraphKind::Verification } else { GraphKind::Speculative }, batch,
                        width, 0, SamplingMode::General)?;
                }
            }
        }
    }
    if !config.speculative_widths.is_empty() {
        for batch in batches(config) {
            if packed_round && !split {
                continue;
            }
            if !manifest.graphs.iter().any(|graph| {
                graph.kind
                    == if split {
                        GraphKind::Verification
                    } else {
                        GraphKind::Speculative
                    }
                    && graph.batch_capacity == batch
            }) {
                return Err(
                    format!("prepared model lacks speculative graphs for batch {batch}").into(),
                );
            }
        }
    }
    Ok(())
}

fn speculative_shape(config: &ModelConfig, batch: usize, tokens: usize) -> bool {
    batches(config).any(|capacity| capacity == batch)
        && config.speculative_widths.contains(&tokens)
        && (tokens == 1 || (batch * tokens <= config.speculative_token_capacity && config
            .speculative_fixed_width
            .is_none_or(|width| width == tokens)))
}

fn batches(config: &ModelConfig) -> impl Iterator<Item = usize> {
    BATCH_CAPACITIES
        .into_iter()
        .filter(|batch| *batch <= config.limits.max_active.next_power_of_two())
}

pub fn select(
    config: &ModelConfig,
    manifest: &ExecutionManifest,
    kind: GraphKind,
    batch: usize,
    tokens: usize,
    prefill: usize,
    mode: SamplingMode,
) -> Result<usize> {
    if kind == GraphKind::Verification {
        if let Some((index, _)) = manifest.graphs.iter().enumerate()
            .filter(|(_, graph)| graph.kind == kind && graph.batch_capacity == batch
                && graph.token_capacity >= tokens && graph.packed_token_capacity >= batch.saturating_mul(tokens)
                && graph.prefill_capacity == prefill && graph.sampling_mode == mode)
            .min_by_key(|(_, graph)| graph.packed_token_capacity) {
            return Ok(index);
        }
    }
    select_shape(config, manifest, kind, batch, tokens, prefill, 0, mode)
}

pub fn select_shape(
    config: &ModelConfig, manifest: &ExecutionManifest, kind: GraphKind,
    batch: usize, tokens: usize, prefill: usize, packed: usize, mode: SamplingMode,
) -> Result<usize> {
    select_shape_with_drafts(config, manifest, kind, batch, tokens, prefill, packed, 0, mode)
}

pub fn select_shape_with_drafts(
    config: &ModelConfig, manifest: &ExecutionManifest, kind: GraphKind,
    batch: usize, tokens: usize, prefill: usize, packed: usize, drafts: usize, mode: SamplingMode,
) -> Result<usize> {
    let declared = batches(config).any(|capacity| capacity == batch)
        && match kind {
            GraphKind::Mixed => {
                batch >= 2
                    && prefill > 0
                    && prefill.is_multiple_of(64)
                    && prefill <= config.limits.prefill_chunk_size
                    && if config.speculative_widths.is_empty() {
                        tokens == 1
                    } else {
                        speculative_shape(config, batch, tokens)
                    }
            }
            GraphKind::Prefill | GraphKind::ScoredPrefill => {
                batch == 1
                    && tokens > 0
                    && tokens <= config.limits.prefill_chunk_size
                    && tokens.is_multiple_of(64)
            }
            GraphKind::Vision => {
                batch == 1 && config.contract.vision.patch_capacities.contains(&tokens)
            }
            GraphKind::PackedSpeculative => {
                config.speculative_widths.contains(&tokens)
                    && drafts > 0 && drafts + 1 <= tokens
                    && packed >= batch && packed <= batch.saturating_mul(tokens)
                    && packed <= config.speculative_token_capacity
            }
            GraphKind::Speculative | GraphKind::Verification => {
                if packed != 0 && kind == GraphKind::Verification {
                    config.speculative_widths.contains(&tokens)
                        && packed <= config.speculative_token_capacity
                        && packed >= batch && packed <= batch.saturating_mul(tokens)
                } else {
                    speculative_shape(config, batch, tokens)
                }
            }
            GraphKind::Proposal => tokens > 1 && config.speculative_widths.contains(&tokens),
            _ => tokens == 1,
        };
    if !declared || (kind != GraphKind::Mixed && prefill != 0) {
        return Err(format!(
            "undeclared prepared graph kind or capacity: {kind:?}, batch {batch}, tokens {tokens}"
        )
        .into());
    }
    let mut matches = manifest.graphs.iter().enumerate().filter(|(_, graph)| {
        graph.kind == kind
            && graph.batch_capacity == batch
            && graph.token_capacity == tokens
            && graph.prefill_capacity == prefill
            && graph.packed_token_capacity == packed
            && graph.draft_tokens == drafts
            && graph.sampling_mode == mode
    });
    let (index, _) = matches.next().ok_or_else(|| {
        format!("prepared model lacks {kind:?}/{mode:?} graph with batch {batch}, tokens {tokens}")
    })?;
    if matches.next().is_some() {
        return Err("ambiguous prepared model graph capacity".into());
    }
    Ok(index)
}
