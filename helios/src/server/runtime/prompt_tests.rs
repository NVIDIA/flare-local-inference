//! Native graph timings for an explicitly supplied, trusted TP1 package.

use super::*;

#[test]
#[ignore = "requires CUDA and FLARE_PROMPT_TEST_PACKAGE plus FLARE_PROMPT_TEST_PUBLISHERS"]
fn prompt_prefill_gpu_and_native_costs() {
    use crate::runtime_package::{LoadOptions, NativeModel, Publisher, VerifiedPackage};
    use serde_json::json;
    use std::{collections::VecDeque, time::Instant};

    let path = std::env::var("FLARE_PROMPT_TEST_PACKAGE").unwrap();
    let profiles = std::env::var("FLARE_PROMPT_TEST_PUBLISHERS").unwrap();
    let publishers: Vec<Publisher> = serde_json::from_slice(&std::fs::read(profiles).unwrap()).unwrap();
    let verified = VerifiedPackage::open(std::path::Path::new(&path), &publishers, None).unwrap();
    let namespace = verified.publisher().namespace.clone();
    let authorized = verified.authorize(&namespace, false).unwrap();
    // The test's explicit publisher file authorizes this package's captured code.
    let owner = unsafe { NativeModel::load(authorized, &[0], &LoadOptions::default(),
        &crate::startup::StartupProgress::silent()).unwrap() };
    let manifest = owner.manifests().unwrap().remove(0);
    let bootstrap = crate::server::Bootstrap::load(&owner.server_config().to_string(), &manifest).unwrap();
    let frontend = Arc::new(bootstrap.frontend);
    let config = bootstrap.model;
    println!("{}", json!({"package":path,"pages":config.limits.pages,
        "page_tokens":config.contract.cache.page_tokens,"graphs":manifest.graphs.len()}));
    let scoring = manifest.prompt_logprob_capacity().unwrap() > 0;
    let runtime = unsafe { ModelRuntime::start(manifest, config, 16, 256, frontend,
        None, crate::scheduler::PolicyConfig::default()).unwrap() };
    runtime.metrics().scheduler_trace.start();
    let asynchronous = tokio::runtime::Runtime::new().unwrap();
    let mut samples = VecDeque::new();
    let mut serial = 0;
    asynchronous.block_on(async {
        for length in [128, 512, 4096, 4097] {
            for trial in 0..3 {
                for (scored, budget) in [(false, 1), (true, 1), (true, 0)] {
                    if scored && !scoring { continue; }
                    serial += 1;
                    let mut tokens = vec![42; length];
                    tokens[0] = 100 + serial;
                    let prompt = Prompt {
                        tokens, message_boundaries: vec![],
                        data: crate::model::PromptData { positions: (0..length).map(|i| [i as i64; 3]).collect(),
                            vision_token_indices: vec![-1; length], spans: vec![] },
                        conditioned_inputs: vec![], conditioning: vec![], sampling: crate::model::SamplingParameters {
                            top_k: 1, ..Default::default()
                        }, max_new_tokens: budget, stop_sequences: vec![], stop_strings: vec![],
                        grammar: None, ignore_eos: false, logprobs: false,
                        logprob_token_ids: vec![], prompt_logprobs: scored,
                    };
                    runtime.metrics().scheduler_trace.drain(&mut samples);
                    let started = Instant::now();
                    let mut generation = runtime.generate(prompt, None).unwrap();
                    let mut positions = 0;
                    loop {
                        match generation.events.recv().await.unwrap() {
                            GenerationEvent::PromptLogprobs { start, tokens, values } => {
                                assert!(scored);
                                assert_eq!(start, positions);
                                assert_eq!(tokens.len(), values.len());
                                if start == 0 { assert!(values[0].is_none()); }
                                positions += values.len();
                            }
                            GenerationEvent::Finished { reused_prompt_tokens, .. } => {
                                assert_eq!(reused_prompt_tokens, 0);
                                break;
                            }
                            GenerationEvent::Error(e) | GenerationEvent::AdmissionRejected(e) => panic!("{e}"),
                            _ => {}
                        }
                    }
                    let wall_ns = started.elapsed().as_nanos();
                    assert_eq!(positions, if scored { length } else { 0 });
                    runtime.metrics().scheduler_trace.drain(&mut samples);
                    let catalog = runtime.metrics().scheduler_trace.catalog.get().unwrap();
                    let prefill: Vec<_> = samples.iter().filter(|s| catalog[s.graph].kind.is_prefill()).collect();
                    assert!(!prefill.is_empty());
                    assert!(prefill.iter().all(|s| catalog[s.graph].kind == if scored {
                        GraphKind::ScoredPrefill
                    } else { GraphKind::Prefill }));
                    let gpu_ns: u64 = prefill.iter().map(|s| s.gpu_ns.unwrap()).sum();
                    println!("{}", json!({"length":length,"trial":trial,"scored":scored,"max_new_tokens":budget,
                        "gpu_prefill_ns":gpu_ns,"native_wall_ns":wall_ns,"prefill_graphs":prefill.len(),
                        "graphs":prefill.iter().map(|s| &catalog[s.graph].name).collect::<Vec<_>>()}));
                }
            }
        }
    });
    let shutdown = runtime.close();
    assert!(shutdown.drained);
    drop(runtime);
    drop(owner);
}
