//! Cache lifecycle tests for the rank distribution transaction helper.
//!
//! Every test drives `receive_rank` through a real `Source`/`Relay` socket
//! transfer; none calls `receive_files` directly.

use std::{
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    sync::Arc,
};

use super::*;

struct Transfer {
    source: Source,
    relays: Vec<Relay>,
    topology: Topology,
}

fn transfer(files: &[PathBuf], specs: &[FileSpec], world: usize) -> Transfer {
    let handles: Vec<Arc<File>> = files
        .iter()
        .map(|path| Arc::new(File::open(path).unwrap()))
        .collect();
    let all = specs.to_vec();
    let source = Source::start(vec![handles; world], vec![all; world], 1).unwrap();
    let network: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let ports: Vec<u16> = network
        .iter()
        .map(|listener| listener.local_addr().unwrap().port())
        .collect();
    let topology =
        Topology::plan(&vec![b"a 127.0.0.1/8\n".to_vec(); world], &ports, false).unwrap();
    let relays = network
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            Relay::start(
                rank,
                1,
                topology.clone(),
                listener,
                if rank == 0 { source.port } else { 0 },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    Transfer {
        source,
        relays,
        topology,
    }
}

pub(super) fn fixture(parent: &Path, export: &str) -> (Vec<PathBuf>, Vec<FileSpec>, Plan) {
    fixture_tps(parent, export, &[2, 4])
}

fn fixture_tps(
    parent: &Path,
    export: &str,
    tensor_parallel: &[u32],
) -> (Vec<PathBuf>, Vec<FileSpec>, Plan) {
    use ring::signature::Ed25519KeyPair;

    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
    let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
    fixture_tps_signed(parent, export, tensor_parallel, [31u8; 32], &key)
}

fn fixture_tps_signed(
    parent: &Path,
    export: &str,
    tensor_parallel: &[u32],
    package_id: [u8; 32],
    key: &ring::signature::Ed25519KeyPair,
) -> (Vec<PathBuf>, Vec<FileSpec>, Plan) {
    use flare_helios::package::{PayloadId, container, export as publish, schema::wire};
    use flare_helios::runtime_package::Publisher;
    use ring::signature::KeyPair;

    let package = parent.join("package");
    fs::create_dir_all(&package).unwrap();
    let base_weights = b"multi rank weights".repeat(8);
    let shared_weights = b"shared rank weights".repeat(7);
    let shared_raw = parent.join("weights-shared.bin");
    fs::write(&shared_raw, &shared_weights).unwrap();
    let shared_id = PayloadId::of(&shared_weights);
    let graph = parent.join("graph.bin");
    fs::write(&graph, b"graph-bytes").unwrap();
    let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
    let publisher = Publisher {
        name: "Multi".into(),
        namespace: "local-development".into(),
        public_key,
        development: true,
    };
    let contract = flare_helios::runtime_package::wire::ServingContract {
        model_id: "multi".into(),
        identity: "multi-sm120".into(),
        max_sequence_tokens: 8,
        prefill_chunk_size: 1,
        pages: 1,
        state_rows: 1,
        page_tokens: 1,
        model_contract: Some(flare_helios::runtime_package::wire::ModelContract {
            vocabulary: 8,
            graphs: vec![flare_helios::runtime_package::wire::GraphContract {
                graph: "decode".into(),
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let deployment = |tp: u32| {
        let ranks = (0..tp)
            .map(|rank| {
                let weights = [
                    base_weights.as_slice(),
                    format!("tp{tp}rank{rank}").as_bytes(),
                ]
                .concat();
                let id = PayloadId::of(&weights);
                let raw = parent.join(format!("weights-{tp}-{rank}.bin"));
                fs::write(&raw, &weights).unwrap();
                let mut initializers = vec![wire::Initializer {
                    allocation: "device-0".into(),
                    destination_offset: 0,
                    source: Some(wire::PayloadUse {
                        slice: Some(wire::PayloadSlice {
                            payload_id: id.as_bytes().to_vec(),
                            offset: 0,
                            nbytes: weights.len() as u64,
                        }),
                        role: wire::PayloadRole::Weights as i32,
                    }),
                }];
                if tp == 2 {
                    initializers.push(wire::Initializer {
                        allocation: "device-0".into(),
                        destination_offset: weights.len() as u64,
                        source: Some(wire::PayloadUse {
                            slice: Some(wire::PayloadSlice {
                                payload_id: shared_id.as_bytes().to_vec(),
                                offset: 0,
                                nbytes: shared_weights.len() as u64,
                            }),
                            role: wire::PayloadRole::Weights as i32,
                        }),
                    });
                }
                publish::PublishRank {
                    body: wire::RankBody {
                        rank,
                        deployment_id: format!("sm120-tp{tp}"),
                        context: Some(Default::default()),
                        allocations: vec![wire::Allocation {
                            name: "device-0".into(),
                            nbytes: weights.len() as u64
                                + if tp == 2 {
                                    shared_weights.len() as u64
                                } else {
                                    0
                                },
                            alignment: 4096,
                            ..Default::default()
                        }],
                        initializers,
                        graphs: vec![wire::Graph {
                            name: "decode".into(),
                            resource: 0,
                            node_count: 1,
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    payloads: {
                        let mut payloads = vec![publish::PublishPayload {
                            role: wire::PayloadRole::Weights,
                            relative: format!("weights/tp{tp}/rank{rank}.safetensors"),
                            source: container::Slice::new(
                                std::sync::Arc::new(std::fs::File::open(&raw).unwrap()),
                                0,
                                weights.len() as u64,
                            )
                            .unwrap(),
                            id,
                            tensor: None,
                        }];
                        if tp == 2 {
                            payloads.push(publish::PublishPayload {
                                role: wire::PayloadRole::Weights,
                                relative: format!("weights/tp2/shared-rank{rank}.safetensors"),
                                source: container::Slice::new(
                                    std::sync::Arc::new(std::fs::File::open(&shared_raw).unwrap()),
                                    0,
                                    shared_weights.len() as u64,
                                )
                                .unwrap(),
                                id: shared_id,
                                tensor: None,
                            });
                        }
                        payloads
                    },
                    resources: vec![publish::PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph.clone(),
                        nbytes: 11,
                    }],
                }
            })
            .collect();
        publish::PublishDeployment {
            body: wire::DeploymentBody {
                deployment_id: format!("sm120-tp{tp}"),
                architecture: vec![12, 0],
                tensor_parallel: tp,
                contract: Some(contract.clone()),
                cluster: Some(flare_helios::runtime_package::wire::Cluster {
                    world: tp,
                    export_id: export.into(),
                }),
                ..Default::default()
            },
            ranks,
            payloads: vec![],
        }
    };
    let request = publish::PublishRequest {
        package: wire::PackageBody {
            model_identity: "owner/multi".into(),
            hardware_identity: "sm120".into(),
            frontend: Some(flare_helios::runtime_package::wire::FrontendSettings::default()),
            sampling_defaults: Some(
                flare_helios::runtime_package::wire::DefaultSamplingParameters::default(),
            ),
            publisher: Some(wire::PublisherMetadata {
                namespace: "local-development".into(),
                official_source: "local-development".into(),
                development: true,
                ..Default::default()
            }),
            ..Default::default()
        },
        payloads: vec![],
        deployments: tensor_parallel.iter().map(|&tp| deployment(tp)).collect(),
        package_id,
        public_key,
        accepted: "local-development".into(),
        no_weights: false,
    };
    publish::publish(&request, &key, publish::Destination::Directory(&package)).unwrap();
    let metadata = flare_helios::package::reader::Metadata::open(&package).unwrap();
    let names: Vec<String> = std::iter::once("model.safetensors".to_owned())
        .chain(
            metadata
                .locations
                .locations
                .iter()
                .filter(|location| !location.file.is_empty())
                .map(|location| location.file.clone()),
        )
        .collect();
    let files: Vec<PathBuf> = names.iter().map(|name| package.join(name)).collect();
    let specs: Vec<FileSpec> = names
        .iter()
        .map(|name| {
            let size = fs::metadata(package.join(name)).unwrap().len();
            FileSpec {
                name: name.clone(),
                size,
                ranges: vec![(0, size)],
            }
        })
        .collect();
    let expected_envelope = flare_helios::runtime_package::VerifiedPackage::open_tensor_parallel(
        &package.join("model.safetensors"),
        std::slice::from_ref(&publisher),
        2,
        Some(1),
    )
    .unwrap()
    .envelope_digest();
    let plan = Plan {
        version: VERSION,
        rank: 1,
        export: export.into(),
        primary: "model.safetensors".into(),
        directory: "package".into(),
        files: specs.clone(),
        model: "owner/multi".into(),
        platform: "gb202".into(),
        tensor_parallel: 2,
        expected_envelope,
        publisher,
        topology: Topology::plan(&vec![b"a 127.0.0.1/8\n".to_vec(); 2], &[1, 2], false).unwrap(),
        streams: 1,
        stage_only: true,
    };
    (files, specs, plan)
}

fn with_topology(plan: &mut Plan, topology: &Topology) {
    plan.topology = topology.clone();
}

fn plan_like(plan: &Plan, export: &str) -> Plan {
    Plan {
        version: plan.version,
        rank: plan.rank,
        export: export.into(),
        primary: plan.primary.clone(),
        directory: plan.directory.clone(),
        files: plan.files.clone(),
        model: plan.model.clone(),
        platform: plan.platform.clone(),
        tensor_parallel: plan.tensor_parallel,
        expected_envelope: plan.expected_envelope,
        publisher: plan.publisher.clone(),
        topology: plan.topology.clone(),
        streams: plan.streams,
        stage_only: plan.stage_only,
    }
}

fn model_key() -> String {
    flare_helios::package::cache::model_key("owner/multi").unwrap()
}

fn run(
    files: &[PathBuf],
    specs: &[FileSpec],
    mut plan: Plan,
    model_root: &Path,
) -> Result<Entry, Failure> {
    let transfer = transfer(files, specs, 2);
    with_topology(&mut plan, &transfer.topology);
    let progress = flare_helios::startup::StartupProgress::stderr_rows();
    let rank = plan.rank;
    let result = receive_rank(model_root, &plan, &transfer.relays[1], &progress, rank);
    let mut finish = Ok(());
    for relay in transfer.relays {
        finish = finish.and_then(|()| relay.finish());
    }
    let source_finish = transfer.source.finish();
    let entry = result?;
    finish?;
    source_finish?;
    Ok(entry)
}

#[test]
fn empty_platform_uses_unknown_device_and_keeps_signed_closure() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (files, specs, plan) = fixture(temp.path(), "export-unknown-device");
    let mut plan = plan_like(&plan, &plan.export);
    plan.platform.clear();
    let entry = run(&files, &specs, plan_like(&plan, &plan.export), &model_root).unwrap();
    assert!(
        entry.id().starts_with("unknown-device-tp2-"),
        "empty platform must use the explicit compatibility label: {}",
        entry.id()
    );
    // The selected rank closure remains a valid signed deployment, not merely an entry name.
    let verified = flare_helios::runtime_package::VerifiedPackage::open_tensor_parallel(
        &entry.directory().join("model.safetensors"),
        std::slice::from_ref(&plan.publisher),
        plan.tensor_parallel,
        Some(plan.rank as u32),
    )
    .unwrap();
    verified.verify_stored_data().unwrap();
}

#[test]
fn warm_reference_rebuilds_a_different_signed_envelope() {
    use ring::signature::{Ed25519KeyPair, KeyPair};

    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
    let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (first_files, first_specs, first_plan) = fixture_tps_signed(
        &temp.path().join("first"),
        "export-envelope",
        &[2],
        [31u8; 32],
        &key,
    );
    let first = run(&first_files, &first_specs, first_plan.clone(), &model_root).unwrap();

    let (second_files, second_specs, second_plan) = fixture_tps_signed(
        &temp.path().join("second"),
        "export-envelope",
        &[2],
        [32u8; 32],
        &key,
    );
    assert_ne!(
        first_plan.expected_envelope, second_plan.expected_envelope,
        "fixtures must carry different signed envelopes"
    );
    let second = run(
        &second_files,
        &second_specs,
        second_plan.clone(),
        &model_root,
    )
    .unwrap();
    assert_ne!(
        second.id(),
        first.id(),
        "a different signed envelope must not satisfy the warm source ref"
    );
    let root = Root::open(&model_root).unwrap();
    assert_eq!(
        root.ref_entry(&model_key()).unwrap().as_deref(),
        Some(second.id())
    );
    let verified = flare_helios::runtime_package::VerifiedPackage::open_tensor_parallel(
        &second.directory().join("model.safetensors"),
        std::slice::from_ref(&second_plan.publisher),
        2,
        Some(1),
    )
    .unwrap();
    assert_eq!(verified.envelope_digest(), second_plan.expected_envelope);
}

#[test]
fn local_package_branch_rejects_a_different_signed_envelope() {
    use ring::signature::{Ed25519KeyPair, KeyPair};

    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
    let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let (_, _, first_plan) = fixture_tps_signed(
        &temp.path().join("first"),
        "export-local-envelope",
        &[2],
        [31u8; 32],
        &key,
    );
    let (second_files, _, second_plan) = fixture_tps_signed(
        &temp.path().join("second"),
        "export-local-envelope",
        &[2],
        [32u8; 32],
        &key,
    );
    let package_dir = second_files
        .iter()
        .find(|path| path.ends_with("model.safetensors"))
        .unwrap()
        .parent()
        .unwrap();
    let verified = flare_helios::runtime_package::VerifiedPackage::open_tensor_parallel(
        package_dir,
        std::slice::from_ref(&second_plan.publisher),
        2,
        None,
    )
    .unwrap();
    let authorized = verified.authorize("local-development", false).unwrap();
    assert_ne!(first_plan.expected_envelope, second_plan.expected_envelope);
    verify_local_package(&authorized, &second_plan).unwrap();
    let error = verify_local_package(&authorized, &first_plan).unwrap_err();
    assert!(
        error.to_string().contains("signed envelope"),
        "local package branch did not bind the expected envelope: {error}"
    );
    for (label, change) in [
        (
            "model",
            (|plan: &mut Plan| plan.model = "other/model".into()) as fn(&mut Plan),
        ),
        ("tensor_parallel", |plan: &mut Plan| {
            plan.tensor_parallel = 3
        }),
        ("export", |plan: &mut Plan| {
            plan.export = "other-export".into()
        }),
    ] {
        let mut wrong = plan_like(&second_plan, &second_plan.export);
        change(&mut wrong);
        assert!(
            verify_local_package(&authorized, &wrong).is_err(),
            "local package branch accepted wrong {label}"
        );
    }
}

#[test]
fn cold_wrong_expected_envelope_does_not_publish_a_default() {
    use ring::signature::{Ed25519KeyPair, KeyPair};

    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
    let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (first_files, first_specs, first_plan) = fixture_tps_signed(
        &temp.path().join("first"),
        "export-wrong-envelope",
        &[2],
        [31u8; 32],
        &key,
    );
    let (second_files, second_specs, mut second_plan) = fixture_tps_signed(
        &temp.path().join("second"),
        "export-wrong-envelope",
        &[2],
        [32u8; 32],
        &key,
    );
    let _ = first_files;
    let _ = first_specs;
    second_plan.expected_envelope = first_plan.expected_envelope;
    let error = match run(&second_files, &second_specs, second_plan, &model_root) {
        Ok(_) => panic!("wrong expected envelope unexpectedly published a rank"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("signed envelope"),
        "wrong expected envelope was not rejected: {error}"
    );
    let root = Root::open(&model_root).unwrap();
    assert_eq!(root.ref_entry(&model_key()).unwrap(), None);
    assert!(root.versions(&model_key()).unwrap_or_default().is_empty());
}

#[test]
fn cold_publish_keeps_canonical_nested_paths_and_warm_reuse_leases() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (files, specs, plan) = fixture(temp.path(), "export-cold");
    let entry = run(&files, &specs, plan_like(&plan, &plan.export), &model_root).unwrap();
    assert!(entry.directory().join("model.safetensors").is_file());
    assert!(
        entry
            .directory()
            .join("weights/tp2/rank0.safetensors")
            .is_file()
    );
    let root = Root::open(&model_root).unwrap();
    assert_eq!(
        root.ref_entry(&model_key()).unwrap().as_deref(),
        Some(entry.id())
    );

    let warm = run(&files, &specs, plan_like(&plan, &plan.export), &model_root).unwrap();
    assert_eq!(warm.id(), entry.id());
    assert_eq!(
        root.ref_entry(&model_key()).unwrap().as_deref(),
        Some(entry.id())
    );
    let error = root
        .evict(&model_key(), entry.id())
        .unwrap_err()
        .to_string();
    assert!(error.contains("leased"), "{error}");
    let explicit = root
        .entry_for_path(&entry.directory().join("model.safetensors"))
        .unwrap()
        .expect("managed cache path must lease its entry");
    assert!(root.evict(&model_key(), entry.id()).is_err());
    drop(explicit);
    drop(warm);
    drop(entry);
    let selected = root.ref_entry(&model_key()).unwrap().unwrap();
    root.evict(&model_key(), &selected).unwrap();
}

#[test]
fn warm_reuse_reselects_the_reused_entry_as_default() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let a = temp.path().join("a");
    let b = temp.path().join("b");
    let (files, specs, first) = fixture(&a, "export-a");
    let (files_b, specs_b, second_plan) = fixture(&b, "export-b");
    let first_entry = run(&files, &specs, plan_like(&first, "export-a"), &model_root).unwrap();
    let second_entry = run(&files_b, &specs_b, second_plan, &model_root).unwrap();
    let root = Root::open(&model_root).unwrap();
    assert_eq!(
        root.ref_entry(&model_key()).unwrap().as_deref(),
        Some(second_entry.id())
    );
    let warm = run(&files, &specs, plan_like(&first, "export-a"), &model_root).unwrap();
    assert_eq!(warm.id(), first_entry.id());
    assert_eq!(
        root.ref_entry(&model_key()).unwrap().as_deref(),
        Some(first_entry.id())
    );
    drop(warm);
    drop(second_entry);
    drop(first_entry);
}

#[test]
fn warm_ref_with_mismatched_signed_identity_is_not_trusted() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let a = temp.path().join("a");
    let b = temp.path().join("b");
    let (files, specs, plan) = fixture(&a, "export-a");
    let (_, _, other) = fixture(&b, "export-b");
    let other_dir = b.join("package");
    let metadata = flare_helios::package::reader::Metadata::open(&other_dir).unwrap();
    let names: Vec<String> = std::iter::once("model.safetensors".to_owned())
        .chain(
            metadata
                .locations
                .locations
                .iter()
                .filter(|l| !l.file.is_empty())
                .map(|l| l.file.clone()),
        )
        .collect();
    let other_files: Vec<PathBuf> = names.iter().map(|n| other_dir.join(n)).collect();
    let other_specs: Vec<FileSpec> = names
        .iter()
        .map(|n| {
            let size = fs::metadata(other_dir.join(n)).unwrap().len();
            FileSpec {
                name: n.clone(),
                size,
                ranges: vec![(0, size)],
            }
        })
        .collect();
    let other_entry = run(&other_files, &other_specs, other.clone(), &model_root).unwrap();
    let root = Root::open(&model_root).unwrap();
    // Point export-a's selector at export-b's verified entry without matching signed metadata.
    root.publish_source_ref(
        &format!("{}-rank{}", plan.export, plan.rank),
        "revision",
        &other_entry,
        other.expected_envelope,
    )
    .unwrap();
    let selected = run(&files, &specs, plan_like(&plan, "export-a"), &model_root).unwrap();
    assert_ne!(selected.id(), other_entry.id());
    assert_eq!(
        root.ref_entry(&model_key()).unwrap().as_deref(),
        Some(selected.id())
    );
    drop(selected);
    drop(other_entry);
}

#[test]
fn selected_tp2_rank1_closure_excludes_other_deployments_and_ranks() {
    use flare_helios::runtime_package::{VerifiedPackage, rank_ranges};

    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (files, _specs, plan) = fixture(temp.path(), "export-closure");
    // The fixture package contains TP2 and TP4 deployments; authorize TP2 rank 1
    // exactly as distribution's rank-zero closure planner does.
    let package_dir = files
        .iter()
        .find(|path| path.ends_with("model.safetensors"))
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    let verified = VerifiedPackage::open_tensor_parallel(
        &package_dir,
        std::slice::from_ref(&plan.publisher),
        2,
        None,
    )
    .unwrap();
    let authorized = verified.authorize("local-development", false).unwrap();
    let ranges = rank_ranges(&authorized, 1).unwrap();
    let mut selected_files = Vec::new();
    let mut selected_specs = Vec::new();
    for range in &ranges {
        let name = if range.filename.is_empty() {
            plan.primary.clone()
        } else {
            range.filename.clone()
        };
        selected_files.push(package_dir.join(&name));
        selected_specs.push(FileSpec {
            name,
            size: range.file.metadata().unwrap().len(),
            ranges: range.ranges.clone(),
        });
    }
    let metadata = flare_helios::package::reader::Metadata::open(&package_dir).unwrap();
    let package_verified = flare_helios::package::protection::verify(
        &metadata.envelope,
        std::slice::from_ref(&plan.publisher),
        flare_helios::package::protection::Selection::Deployment {
            deployment: 0,
            rank: Some(1),
        },
    )
    .unwrap();
    let rank = package_verified
        .ranks()
        .iter()
        .find(|rank| rank.rank == 1)
        .unwrap();
    let mut expected_ids = std::collections::BTreeSet::new();
    for initializer in &rank.initializers {
        if let Some(slice) = initializer
            .source
            .as_ref()
            .and_then(|use_| use_.slice.as_ref())
        {
            expected_ids.insert(slice.payload_id.clone());
        }
    }
    for resource in &rank.resources {
        if let Some(stored) = resource.stored.as_ref() {
            expected_ids.insert(stored.payload_id.clone());
        }
    }
    for module in &rank.modules {
        for global in &module.globals {
            if let Some(slice) = global
                .contents
                .as_ref()
                .and_then(|use_| use_.slice.as_ref())
            {
                expected_ids.insert(slice.payload_id.clone());
            }
        }
    }
    for asset in &package_verified.package().assets {
        if let Some(slice) = asset
            .payload
            .as_ref()
            .and_then(|use_| use_.slice.as_ref())
        {
            expected_ids.insert(slice.payload_id.clone());
        }
    }
    let mut selected_ids = std::collections::BTreeSet::new();
    for spec in &selected_specs {
        if spec.name == plan.primary {
            continue;
        }
        let location = metadata
            .locations
            .locations
            .iter()
            .find(|location| location.file == spec.name)
            .unwrap();
        selected_ids.insert(location.payload_id.clone());
    }
    assert_eq!(selected_ids, expected_ids);
    assert_eq!(selected_specs.len(), expected_ids.len() + 1);
    let mut closure_names: Vec<String> = selected_specs
        .iter()
        .map(|spec| spec.name.clone())
        .collect();
    closure_names.sort();
    assert_eq!(
        closure_names.iter().filter(|name| name.ends_with("/graphs.safetensors")).count(),
        1
    );
    assert!(closure_names.contains(&"weights/tp2/rank1.safetensors".to_owned()));
    assert!(closure_names.contains(&"weights/tp2/shared-rank0.safetensors".to_owned()));

    let handles: Vec<Arc<File>> = selected_files
        .iter()
        .map(|path| Arc::new(File::open(path).unwrap()))
        .collect();
    let source = Source::start(
        vec![Vec::new(), handles],
        vec![Vec::new(), selected_specs.clone()],
        1,
    )
    .unwrap();
    let network: Vec<TcpListener> = (0..2)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let ports: Vec<u16> = network
        .iter()
        .map(|l| l.local_addr().unwrap().port())
        .collect();
    let topology = Topology::plan(&vec![b"a 127.0.0.1/8\n".to_vec(); 2], &ports, false).unwrap();
    let mut listeners = network.into_iter();
    let relays = vec![
        Relay::start(
            0,
            1,
            topology.clone(),
            listeners.next().unwrap(),
            source.port,
        )
        .unwrap(),
        Relay::start(1, 1, topology.clone(), listeners.next().unwrap(), 0).unwrap(),
    ];
    let mut plan = plan;
    plan.rank = 1;
    plan.topology = topology;
    assert_eq!(
        selected_specs
            .iter()
            .filter(|spec| spec.name == "weights/tp2/shared-rank0.safetensors")
            .count(),
        1
    );
    plan.files = selected_specs;
    let progress = flare_helios::startup::StartupProgress::stderr_rows();
    let entry = receive_rank(&model_root, &plan, &relays[1], &progress, 1).unwrap();
    for file in &plan.files {
        assert!(
            entry.directory().join(&file.name).is_file(),
            "{}",
            file.name
        );
    }
    for private in [
        "weights/tp2/rank0.safetensors",
        "weights/tp4/rank0.safetensors",
        "weights/tp4/rank1.safetensors",
        "weights/tp4/rank2.safetensors",
        "weights/tp4/rank3.safetensors",
    ] {
        assert!(
            !entry.directory().join(private).exists(),
            "private other-rank payload {private} was installed"
        );
    }
    for relay in relays {
        let _ = relay.finish();
    }
    let _ = source.finish();
}

#[test]
fn signed_legacy_cluster_custom_primary_distributes_a_sparse_rank_and_warms() {
    use flare_helios::runtime_package::{Publisher, VerifiedPackage, rank_ranges};

    let temp = tempfile::tempdir().unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../helios/tests/fixtures/legacy-cluster-custom");
    let source_dir = temp.path().join("source");
    fs::create_dir_all(&source_dir).unwrap();
    for entry in fs::read_dir(&fixture).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let target = if name == "model.safetensors" {
            source_dir.join("custom-name.safetensors")
        } else {
            source_dir.join(name)
        };
        fs::copy(entry.path(), target).unwrap();
    }
    let custom = source_dir.join("custom-name.safetensors");
    let publisher = Publisher {
        name: "Flare test publisher".into(),
        namespace: "flare-tests".into(),
        public_key: [
            0x19, 0x7f, 0x6b, 0x23, 0xe1, 0x6c, 0x85, 0x32, 0xc6, 0xab, 0xc8, 0x38, 0xfa, 0xcd,
            0x5e, 0xa7, 0x89, 0xbe, 0x0c, 0x76, 0xb2, 0x92, 0x03, 0x34, 0x03, 0x9b, 0xfa, 0x8b,
            0x3d, 0x36, 0x8d, 0x61,
        ],
        development: false,
    };
    let verified = VerifiedPackage::open(&custom, std::slice::from_ref(&publisher), None).unwrap();
    let authorized = verified.authorize("flare-tests", false).unwrap();
    let ranges = rank_ranges(&authorized, 1).unwrap();
    let mut selected_files = Vec::new();
    let mut selected_specs = Vec::new();
    let mut rank_handles = Vec::new();
    for range in &ranges {
        let name = if range.filename.is_empty() {
            "custom-name.safetensors".to_owned()
        } else {
            range.filename.clone()
        };
        let path = source_dir.join(&name);
        let size = range.file.metadata().unwrap().len();
        selected_specs.push(FileSpec {
            name,
            size,
            ranges: range.ranges.clone(),
        });
        rank_handles.push(Arc::new(File::open(&path).unwrap()));
        selected_files.push(path);
    }
    assert!(
        selected_specs.len() > 1,
        "fixture must exercise sparse rank closure"
    );
    let model_root = temp.path().join("flare");
    let legacy_key = flare_helios::package::cache::model_key("owner/cluster-legacy").unwrap();
    let plan = Plan {
        version: VERSION,
        rank: 1,
        export: "export-legacy".into(),
        primary: "custom-name.safetensors".into(),
        directory: "package".into(),
        files: selected_specs.clone(),
        model: "owner/cluster-legacy".into(),
        platform: "gb202".into(),
        tensor_parallel: 2,
        expected_envelope: authorized.envelope_digest(),
        publisher,
        topology: Topology::plan(&vec![b"a 127.0.0.1/8\n".to_vec(); 2], &[1, 2], false).unwrap(),
        streams: 1,
        stage_only: true,
    };

    let receive = |files: Vec<Vec<Arc<File>>>, specs: Vec<Vec<FileSpec>>| {
        let source = Source::start(files, specs, 1).unwrap();
        let network: Vec<TcpListener> = (0..2)
            .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
            .collect();
        let ports: Vec<u16> = network
            .iter()
            .map(|listener| listener.local_addr().unwrap().port())
            .collect();
        let topology =
            Topology::plan(&vec![b"a 127.0.0.1/8\n".to_vec(); 2], &ports, false).unwrap();
        let relays = network
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                Relay::start(
                    rank,
                    1,
                    topology.clone(),
                    listener,
                    if rank == 0 { source.port } else { 0 },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut plan = plan_like(&plan, "export-legacy");
        plan.topology = topology;
        let progress = flare_helios::startup::StartupProgress::stderr_rows();
        let result = receive_rank(&model_root, &plan, &relays[1], &progress, 1);
        for relay in relays {
            let _ = relay.finish();
        }
        let _ = source.finish();
        result
    };

    let first = receive(
        vec![Vec::new(), rank_handles],
        vec![Vec::new(), selected_specs.clone()],
    )
    .unwrap();
    assert!(first.directory().join("model.safetensors").is_file());
    assert!(!first.directory().join("custom-name.safetensors").exists());
    for spec in &selected_specs {
        if spec.name != "custom-name.safetensors" {
            assert!(first.directory().join(&spec.name).is_file());
        }
    }
    assert!(
        !first
            .directory()
            .join("model-data-01-of-02.safetensors")
            .exists()
    );
    let root = Root::open(&model_root).unwrap();
    assert_eq!(
        root.ref_entry(&legacy_key).unwrap().as_deref(),
        Some(first.id())
    );
    let _lease = root
        .entry_for_path(&first.directory().join("model.safetensors"))
        .unwrap()
        .expect("custom primary must resolve to the managed entry");

    let warm_handles: Vec<Arc<File>> = selected_files
        .iter()
        .map(|path| Arc::new(File::open(path).unwrap()))
        .collect();
    let warm = receive(
        vec![Vec::new(), warm_handles],
        vec![Vec::new(), selected_specs],
    )
    .unwrap();
    assert_eq!(warm.id(), first.id());
    assert_eq!(
        root.ref_entry(&legacy_key).unwrap().as_deref(),
        Some(first.id())
    );
    drop(warm);
    drop(first);
}

#[test]
fn cold_rejects_wrong_cluster_world_before_publication() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (files, specs, mut plan) = fixture(temp.path(), "export-world");
    let transfer = transfer(&files, &specs, 3);
    plan.topology = transfer.topology.clone();
    let progress = flare_helios::startup::StartupProgress::stderr_rows();
    let rank = plan.rank;
    assert!(receive_rank(&model_root, &plan, &transfer.relays[1], &progress, rank).is_err());
    for relay in transfer.relays {
        let _ = relay.finish();
    }
    let _ = transfer.source.finish();
    let root = Root::open(&model_root).unwrap();
    assert!(root.ref_entry(&model_key()).unwrap().is_none());
    assert!(root.versions(&model_key()).unwrap_or_default().is_empty());
}

#[test]
fn cold_rejects_wrong_plan_identity_before_publication() {
    let cases: [(&str, fn(&mut Plan)); 3] = [
        ("model", |plan| plan.model = "other/model".into()),
        ("tensor_parallel", |plan| plan.tensor_parallel = 3),
        ("export", |plan| plan.export = "other-export".into()),
    ];
    for (name, change) in cases {
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("flare");
        let (files, specs, mut plan) = fixture(temp.path(), "export-identity");
        change(&mut plan);
        let error = run(&files, &specs, plan, &model_root)
            .err()
            .unwrap_or_else(|| panic!("{name}: wrong plan identity was published"));
        assert!(!error.to_string().is_empty(), "{name}");
        let root = Root::open(&model_root).unwrap();
        assert!(root.ref_entry(&model_key()).unwrap().is_none(), "{name}");
        assert!(
            root.versions(&model_key()).unwrap_or_default().is_empty(),
            "{name}"
        );
    }
}

#[test]
fn active_entry_leases_block_eviction_and_managed_paths_resolve_to_leases() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let root = Root::open(&model_root).unwrap();
    let mut staging = root.stage("lease-model", "lease-entry").unwrap();
    staging.write("model.safetensors", b"primary").unwrap();
    let entry = staging.commit().unwrap();
    let model = entry.model().to_owned();
    let id = entry.id().to_owned();
    let error = root.evict(&model, &id).unwrap_err().to_string();
    assert!(error.contains("leased"), "{error}");
    let leased = root
        .entry_for_path(&entry.directory().join("model.safetensors"))
        .unwrap()
        .expect("managed path must lease");
    assert_eq!(leased.id(), id);
    assert!(root.evict(&model, &id).is_err());
    assert!(
        root.entry_for_path(&temp.path().join("elsewhere.safetensors"))
            .unwrap()
            .is_none()
    );
    drop(leased);
    assert!(root.evict(&model, &id).is_err());
    drop(entry);
    root.evict(&model, &id).unwrap();
    assert!(root.versions(&model).unwrap().is_empty());
}

#[test]
fn entry_for_path_returns_none_after_the_managed_entry_is_gone() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let root = Root::open(&model_root).unwrap();
    let mut staging = root.stage("gone-model", "gone-entry").unwrap();
    staging.write("model.safetensors", b"primary").unwrap();
    let entry = staging.commit().unwrap();
    let path = entry.directory().join("model.safetensors");
    let (model, id) = (entry.model().to_owned(), entry.id().to_owned());
    drop(entry);
    root.evict(&model, &id).unwrap();
    assert!(
        root.entry_for_path(&path).unwrap().is_none(),
        "a removed managed entry must resolve to None"
    );
}

#[test]
fn failed_verification_leaves_the_previous_default_untouched() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (files, specs, plan) = fixture(temp.path(), "export-bad");
    let root = Root::open(&model_root).unwrap();
    let mut previous_staging = root.stage(&model_key(), "previous").unwrap();
    previous_staging
        .write("model.safetensors", b"previous default")
        .unwrap();
    let previous = previous_staging.commit().unwrap();
    root.publish_entry(&previous).unwrap();
    let companion = files
        .iter()
        .find(|path| {
            path.to_string_lossy()
                .ends_with("weights/tp2/rank1.safetensors")
        })
        .expect("rank 1 fixture companion");
    let mut bytes = fs::read(companion).unwrap();
    let end = bytes.len() - 1;
    bytes[end] ^= 0xff;
    fs::write(companion, bytes).unwrap();
    assert!(run(&files, &specs, plan, &model_root).is_err());
    assert_eq!(
        root.ref_entry(&model_key()).unwrap().as_deref(),
        Some(previous.id())
    );
    assert!(previous.directory().join("model.safetensors").is_file());
    assert_eq!(
        fs::read_dir(root.path().join("staging/trees"))
            .unwrap()
            .count(),
        0
    );
    drop(previous);
}

#[test]
fn corrupted_warm_rank_entry_is_rebuilt_and_republished() {
    use std::io::Write as _;

    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (files, specs, plan) = fixture(temp.path(), "export-warm-corrupt");
    let first = run(&files, &specs, plan_like(&plan, &plan.export), &model_root).unwrap();
    let root = Root::open(&model_root).unwrap();
    assert_eq!(
        root.ref_entry(&model_key()).unwrap().as_deref(),
        Some(first.id())
    );

    // Corrupt the selected warm entry's rank payload while its lease is live.
    let companion = first.directory().join("weights/tp2/rank1.safetensors");
    let mut permissions = fs::metadata(&companion).unwrap().permissions();
    permissions.set_readonly(false);
    fs::set_permissions(&companion, permissions).unwrap();
    let mut bytes = fs::read(&companion).unwrap();
    let end = bytes.len() - 1;
    bytes[end] ^= 0xff;
    fs::write(&companion, &bytes).unwrap();

    // Warm verification must reject the corrupted entry, receive a fresh rank
    // package over the real Source/Relay path, and select the rebuilt entry.
    let second = run(&files, &specs, plan_like(&plan, &plan.export), &model_root).unwrap();
    assert_ne!(second.id(), first.id());
    let rebuilt_companion = second.directory().join("weights/tp2/rank1.safetensors");
    let source_companion = files
        .iter()
        .find(|path| {
            path.to_string_lossy()
                .ends_with("weights/tp2/rank1.safetensors")
        })
        .expect("source rank 1 companion");
    assert_eq!(
        fs::read(&rebuilt_companion).unwrap(),
        fs::read(source_companion).unwrap(),
        "rebuilt rank payload must match the authenticated source bytes"
    );
    assert_eq!(
        root.ref_entry(&model_key()).unwrap().as_deref(),
        Some(second.id())
    );
    assert!(
        root.versions(&model_key())
            .unwrap()
            .contains(&first.id().to_owned())
    );
    assert!(
        root.versions(&model_key())
            .unwrap()
            .contains(&second.id().to_owned())
    );
    assert_eq!(
        fs::read_dir(root.path().join("staging/trees"))
            .unwrap()
            .count(),
        0
    );
    drop(second);
    drop(first);
}

#[test]
fn cold_receive_records_a_verified_marker_that_tracks_every_file() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (files, specs, plan) = fixture(temp.path(), "export-marker");
    let entry = run(&files, &specs, plan_like(&plan, &plan.export), &model_root).unwrap();
    let marker_path = entry.directory().join(VERIFIED_MARKER);
    let marker: VerifiedMarker =
        serde_json::from_slice(&fs::read(&marker_path).expect("cold receive writes a marker"))
            .unwrap();
    assert_eq!(marker.version, 2);
    assert_eq!(marker.rank, plan.rank);
    assert_eq!(marker.export, plan.export);
    assert_eq!(marker.envelope, hex_bytes(&plan.expected_envelope));
    assert_eq!(marker.files.len(), plan.files.len());
    assert!(verified_marker_matches(entry.directory(), &plan, plan.rank));

    // Rewriting any recorded file invalidates the marker, even at the same size.
    let companion = entry.directory().join("weights/tp2/rank1.safetensors");
    let bytes = fs::read(&companion).unwrap();
    let mut permissions = fs::metadata(&companion).unwrap().permissions();
    permissions.set_readonly(false);
    fs::set_permissions(&companion, permissions).unwrap();
    fs::write(&companion, &bytes).unwrap();
    assert!(!verified_marker_matches(entry.directory(), &plan, plan.rank));

    // A warm launch rehashes the entry once and records a fresh marker.
    let warm = run(&files, &specs, plan_like(&plan, &plan.export), &model_root).unwrap();
    assert_eq!(warm.id(), entry.id());
    assert!(verified_marker_matches(warm.directory(), &plan, plan.rank));
}

#[test]
fn warm_launch_reuses_a_verified_entry_without_rewriting_its_marker() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (files, specs, plan) = fixture(temp.path(), "export-warm-marker");
    let entry = run(&files, &specs, plan_like(&plan, &plan.export), &model_root).unwrap();
    let marker_path = entry.directory().join(VERIFIED_MARKER);
    let recorded = fs::metadata(&marker_path).unwrap();
    let recorded_ns = modified_ns(&recorded);
    std::thread::sleep(std::time::Duration::from_millis(10));
    let warm = run(&files, &specs, plan_like(&plan, &plan.export), &model_root).unwrap();
    assert_eq!(warm.id(), entry.id());
    assert_eq!(
        modified_ns(&fs::metadata(&marker_path).unwrap()),
        recorded_ns,
        "a marker hit must not rehash or rewrite the marker"
    );
}

#[test]
fn unreadable_marker_directory_still_serves_the_verified_rank() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (files, specs, plan) = fixture(temp.path(), "export-readonly-marker");
    let entry = run(&files, &specs, plan_like(&plan, &plan.export), &model_root).unwrap();
    let directory = entry.directory().to_owned();
    fs::remove_file(directory.join(VERIFIED_MARKER)).unwrap();
    let mut permissions = fs::metadata(&directory).unwrap().permissions();
    let original = permissions.clone();
    permissions.set_readonly(true);
    fs::set_permissions(&directory, permissions).unwrap();
    let warm = run(&files, &specs, plan_like(&plan, &plan.export), &model_root);
    fs::set_permissions(&directory, original).unwrap();
    let warm = warm.expect("a cache that cannot record a marker must still serve");
    assert_eq!(warm.id(), entry.id());
    assert!(
        !directory.join(VERIFIED_MARKER).exists(),
        "read-only entry unexpectedly recorded a marker"
    );
}

#[test]
fn warm_entry_is_not_trusted_without_its_signed_envelope() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (files, specs, plan) = fixture(temp.path(), "export-warm-envelope");
    let entry = run(&files, &specs, plan_like(&plan, &plan.export), &model_root).unwrap();
    let mut wrong = plan_like(&plan, &plan.export);
    wrong.expected_envelope = [0u8; 32];
    let error = run(&files, &specs, wrong, &model_root)
        .err()
        .expect("a stale signed envelope must not reuse the warm entry");
    assert!(
        error.to_string().contains("signed envelope"),
        "warm reuse did not re-authenticate the envelope: {error}"
    );
    let root = Root::open(&model_root).unwrap();
    assert_eq!(
        root.ref_entry(&model_key()).unwrap().as_deref(),
        Some(entry.id()),
        "rejected warm reuse must leave the previous default untouched"
    );
}

#[test]
fn interrupted_receive_removes_its_staging_tree() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let (files, mut specs, plan) = fixture(temp.path(), "export-interrupted");
    let index = specs.len() - 1;
    specs[index].ranges = vec![(specs[index].size + 4096, 64)];
    let transfer = transfer(&files, &specs, 2);
    let mut plan = plan;
    with_topology(&mut plan, &transfer.topology);
    let progress = flare_helios::startup::StartupProgress::stderr_rows();
    let rank = plan.rank;
    assert!(receive_rank(&model_root, &plan, &transfer.relays[1], &progress, rank).is_err());
    for relay in transfer.relays {
        let _ = relay.finish();
    }
    let _ = transfer.source.finish();
    let root = Root::open(&model_root).unwrap();
    assert_eq!(
        fs::read_dir(root.path().join("staging/trees"))
            .unwrap()
            .count(),
        0
    );
    assert!(root.ref_entry(&model_key()).unwrap().is_none());
}

#[test]
fn concurrent_receivers_publish_complete_entries_and_last_default_wins() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path().join("flare");
    let a = temp.path().join("a");
    let b = temp.path().join("b");
    let (files, specs, plan) = fixture(&a, "export-a");
    let (files_b, specs_b, second) = fixture(&b, "export-b");
    let root = Root::open(&model_root).unwrap();
    let (first, other) = std::thread::scope(|scope| {
        let first = scope.spawn(|| run(&files, &specs, plan, &model_root));
        let second = scope.spawn(|| run(&files_b, &specs_b, second, &model_root));
        (first.join().unwrap(), second.join().unwrap())
    });
    let first = first.unwrap();
    let other = other.unwrap();
    assert_ne!(first.id(), other.id());
    let mut versions = root.versions(&model_key()).unwrap();
    versions.sort();
    assert_eq!(versions.len(), 2);
    assert!(root.evict(&model_key(), first.id()).is_err());
    assert!(root.evict(&model_key(), other.id()).is_err());
    drop(first);
    drop(other);
    let selected = root.ref_entry(&model_key()).unwrap().unwrap();
    assert!(versions.contains(&selected));
}

fn copy_fixture_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_fixture_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
#[ignore = "explicit cross-host distribution fixture export"]
fn export_signed_payload_distribution_fixture() {
    let output = std::env::var_os("FLARE_DISTRIBUTION_FIXTURE_OUTPUT")
        .expect("set FLARE_DISTRIBUTION_FIXTURE_OUTPUT to an absent output directory");
    let output = PathBuf::from(output);
    assert!(
        !output.exists(),
        "fixture output already exists: {}",
        output.display()
    );
    let scratch = tempfile::tempdir().unwrap();
    let (_, _, plan) = fixture(scratch.path(), "payload-distribution-fixture");
    copy_fixture_tree(&scratch.path().join("package"), &output.join("package"));
    fs::write(
        output.join("publisher.json"),
        serde_json::to_vec(&[plan.publisher]).unwrap(),
    )
    .unwrap();
}
