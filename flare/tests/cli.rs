use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use flare_helios::package::{
    PayloadId, cache, container, export, protection, reader,
    schema::wire,
};
use flare_helios::runtime_package::Publisher;
use ring::signature::{Ed25519KeyPair, KeyPair};

fn flare(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_flare"))
        .args(arguments)
        .output()
        .unwrap()
}

#[test]
fn single_host_run_reaches_package_loading() {
    let temp = tempfile::tempdir().unwrap();
    let package = temp.path().join("missing.safetensors");
    for options in [
        vec![],
        vec![
            "--publisher-keys",
            "development-publisher.json",
            "--prefix-cache-disk",
            temp.path().to_str().unwrap(),
            "--kv-cache-limit",
            "concurrency",
            "--host",
            "0.0.0.0",
            "--served-model-name",
            "qwen3.8-flash",
        ],
        vec!["--gpus", "0,1"],
    ] {
        let mut arguments = vec!["run", package.to_str().unwrap()];
        arguments.extend(options);
        let output = flare(&arguments);
        assert!(!output.status.success(), "{arguments:?}: {output:?}");
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(
            error.contains("cannot access package"),
            "{arguments:?}: {error}"
        );
    }
}

#[cfg(windows)]
#[test]
fn windows_rejects_multi_host_options_before_loading() {
    for options in [
        vec!["--rank", "0"],
        vec!["--rank", "1"],
        vec!["--rendezvous", "127.0.0.1:29630"],
        vec!["--rank", "0", "--rendezvous", "127.0.0.1:29630"],
    ] {
        let mut arguments = vec!["run", "/nonexistent/model.safetensors"];
        arguments.extend(options);
        let output = flare(&arguments);
        assert!(!output.status.success(), "{arguments:?}: {output:?}");
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(
            error.contains("multi-host serving is not supported on Windows"),
            "{arguments:?}: {error}"
        );
    }
}

#[test]
fn help_exits_successfully_before_accessing_files() {
    for arguments in [
        vec!["--help"],
        vec!["-h"],
        vec!["run", "--help"],
        vec!["run", "-h"],
        vec!["help", "run"],
        vec!["inspect", "--help"],
        vec!["help", "inspect"],
        vec!["download", "--help"],
        vec!["download", "-h"],
        vec!["help", "download"],
        vec![
            "download",
            "owner/model",
            "--token-file",
            "/nonexistent/token",
            "--help",
        ],
        vec![
            "run",
            "/nonexistent/model.safetensors",
            "--embedding-table-storage",
            "disk",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--served-model-name",
            "local-model",
            "--api-key-file",
            "/nonexistent/key.txt",
            "--help",
        ],
    ] {
        let output = flare(&arguments);
        assert!(output.status.success(), "{arguments:?}: {output:?}");
        assert!(output.stderr.is_empty(), "{arguments:?}: {output:?}");
        let help = String::from_utf8(output.stdout).unwrap();
        assert!(help.contains("Usage: flare"), "{arguments:?}: {help}");
        if arguments.contains(&"run") {
            assert!(help.contains("--embedding-table-storage"), "{help}");
            assert!(help.contains("--no-mtp-adaptive-verification"), "{help}");
            assert!(help.contains("--mtp-adaptive-verification-alpha"), "{help}");
            assert!(help.contains("--prefix-cache-pool-ca"), "{help}");
            assert!(help.contains("--host"), "{help}");
            assert!(help.contains("--port"), "{help}");
            assert!(help.contains("--served-model-name"), "{help}");
            assert!(help.contains("--scheduler-terminal-value"), "{help}");
            assert!(help.contains("--scheduler-terminal-age-price"), "{help}");
            assert!(help.contains("127.0.0.1"), "{help}");
            assert!(!help.contains("--bind"), "{help}");
        }
    }
}

fn signed_package(scratch: &Path) -> (PathBuf, Publisher) {
    let package = scratch.join("package");
    std::fs::create_dir_all(&package).unwrap();
    let weights = b"offline cli package weights".repeat(16);
    let weight_path = scratch.join("weights.bin");
    std::fs::write(&weight_path, &weights).unwrap();
    let weight_id = PayloadId::of(&weights);
    let graph = scratch.join("graph.bin");
    std::fs::write(&graph, b"graph-bytes").unwrap();
    let key = Ed25519KeyPair::from_pkcs8(
        Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
            .unwrap()
            .as_ref(),
    )
    .unwrap();
    let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
    let publisher = Publisher {
        name: "CLI fixture".into(),
        namespace: "local-development".into(),
        public_key,
        development: true,
    };
    let contract = flare_helios::runtime_package::wire::ServingContract {
        model_id: "cli-model".into(),
        identity: "cli-model".into(),
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
    let rank = wire::RankBody {
        rank: 0,
        deployment_id: "gb202-tp1".into(),
        context: Some(Default::default()),
        allocations: vec![wire::Allocation {
            name: "device-0".into(),
            nbytes: weights.len() as u64,
            alignment: 4096,
            ..Default::default()
        }],
        initializers: vec![wire::Initializer {
            allocation: "device-0".into(),
            destination_offset: 0,
            source: Some(wire::PayloadUse {
                slice: Some(wire::PayloadSlice {
                    payload_id: weight_id.as_bytes().to_vec(),
                    offset: 0,
                    nbytes: weights.len() as u64,
                }),
                role: wire::PayloadRole::Weights as i32,
            }),
        }],
        graphs: vec![wire::Graph {
            name: "decode".into(),
            resource: 0,
            node_count: 1,
            ..Default::default()
        }],
        ..Default::default()
    };
    let request = export::PublishRequest {
        package: wire::PackageBody {
            model_identity: "owner/model".into(),
            hardware_identity: "gb202".into(),
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
        deployments: vec![export::PublishDeployment {
            body: wire::DeploymentBody {
                deployment_id: "gb202-tp1".into(),
                architecture: vec![12, 0],
                tensor_parallel: 1,
                contract: Some(contract),
                ..Default::default()
            },
            ranks: vec![export::PublishRank {
                body: rank,
                payloads: vec![export::PublishPayload {
                    role: wire::PayloadRole::Weights,
                    relative: "weights/model.safetensors".into(),
                    source: container::Slice::new(
                        std::sync::Arc::new(std::fs::File::open(&weight_path).unwrap()),
                        0,
                        weights.len() as u64,
                    )
                    .unwrap(),
                    id: weight_id,
                    tensor: None,
                }],
                resources: vec![export::PublishResource {
                    kind: wire::ExecutableKind::Graph,
                    path: graph,
                    nbytes: 11,
                }],
            }],
            payloads: vec![],
        }],
        package_id: [73u8; 32],
        public_key,
        accepted: "local-development".into(),
        no_weights: false,
    };
    export::publish(&request, &key, export::Destination::Directory(&package)).unwrap();
    (package, publisher)
}

fn copy_tree(source: &Path, destination: &Path) {
    std::fs::create_dir_all(destination).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn offline_download_uses_the_selected_flare_cache_entry_and_rejects_hf_only_cache() {
    let temp = tempfile::tempdir().unwrap();
    let commit = "0123456789012345678901234567890123456789";
    let (package, publisher) = signed_package(temp.path());
    let profile = temp.path().join("publisher.json");
    std::fs::write(
        &profile,
        serde_json::to_vec(&vec![publisher.clone()]).unwrap(),
    )
    .unwrap();

    let model_cache = temp.path().join("flare-cache");
    let trust_store = temp.path().join("trust.toml");
    let store = cache::Root::open(&model_cache).unwrap();
    let metadata = reader::Metadata::open(&package).unwrap();
    let source = cache::PackageSource::open(
        &package,
        std::slice::from_ref(&publisher),
        protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        },
    )
    .unwrap();
    let model_key = cache::model_key("owner/model").unwrap();
    let entry = store.install_package(source, &model_key, "entry").unwrap();
    let selector = serde_json::json!([
        "huggingface",
        "owner/model",
        commit,
        serde_json::Value::Null,
        []
    ])
    .to_string();
    store
        .publish_source_ref(
            &selector,
            commit,
            &entry,
            *PayloadId::of(&metadata.envelope).as_bytes(),
        )
        .unwrap();

    store.publish_entry_source(
        &serde_json::json!(["huggingface", "owner/model", commit, "model.safetensors", [1]]).to_string(),
        commit, &entry, *PayloadId::of(&metadata.envelope).as_bytes(),
    ).unwrap();

    for command in [vec!["download"], vec!["models", "download"]] {
        let repository = format!("owner/model@{commit}");
        let mut arguments = command;
        arguments.extend([
            &repository,
            "--offline",
            "--model-cache",
            model_cache.to_str().unwrap(),
            "--publisher-keys",
            profile.to_str().unwrap(),
            "--trust-store",
            trust_store.to_str().unwrap(),
        ]);
        let output = flare(&arguments);
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            String::from_utf8(output.stdout).unwrap().trim(),
            entry.directory().display().to_string()
        );
    }

    let hf_cache = temp.path().join("hf-only");
    let snapshot = hf_cache
        .join("models--owner--model")
        .join("snapshots")
        .join(commit);
    copy_tree(&package, &snapshot);
    let empty_model_cache = temp.path().join("empty-flare-cache");
    let output = flare(&[
        "download",
        &format!("owner/model@{commit}"),
        "--offline",
        "--cache-dir",
        hf_cache.to_str().unwrap(),
        "--model-cache",
        empty_model_cache.to_str().unwrap(),
        "--publisher-keys",
        profile.to_str().unwrap(),
        "--trust-store",
        trust_store.to_str().unwrap(),
    ]);
    assert!(!output.status.success(), "{output:?}");
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("not available in the Flare cache"), "{error}");

    let output = flare(&[
        "run",
        "owner/model",
        "--revision",
        commit,
        "--offline",
        "--model-cache",
        empty_model_cache.to_str().unwrap(),
        "--publisher-keys",
        profile.to_str().unwrap(),
        "--trust-store",
        trust_store.to_str().unwrap(),
    ]);
    assert!(!output.status.success(), "{output:?}");
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("not available in the Flare cache"), "{error}");
    assert!(!error.contains("cannot load NVIDIA"), "{error}");

    let output = flare(&[
        "run",
        "owner/model",
        "--revision",
        commit,
        "--offline",
        "--model-cache",
        model_cache.to_str().unwrap(),
        "--publisher-keys",
        profile.to_str().unwrap(),
        "--trust-store",
        trust_store.to_str().unwrap(),
    ]);
    assert!(!output.status.success(), "{output:?}");
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains(
            "unattended first use requires --dangerously-approve-key FINGERPRINT --dangerously-accept-publisher NAMESPACE"
        ),
        "{error}"
    );
    assert!(!error.contains("cannot load NVIDIA"), "{error}");
    assert!(!error.contains("cannot access package"), "{error}");
}

#[test]
fn inspect_help_lists_the_tensor_parallel_selector() {
    let output = flare(&["inspect", "--help"]);
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("--tensor-parallel"), "{help}");
}

#[test]
fn usage_errors_identify_the_problem_before_accessing_files() {
    for (arguments, expected) in [
        (vec![], "Usage: flare"),
        (vec!["run"], "<PACKAGE>"),
        (vec!["inspect"], "<PACKAGE|MODEL[@VERSION]>"),
        (vec!["rnu"], "unrecognized subcommand"),
        (
            vec!["run", "/nonexistent/model", "--bogus"],
            "unexpected argument '--bogus'",
        ),
        (
            vec!["run", "/nonexistent/model", "--host"],
            "a value is required for '--host",
        ),
        (
            vec!["run", "/nonexistent/model", "--host", "localhost"],
            "invalid value 'localhost' for '--host",
        ),
        (
            vec!["run", "/nonexistent/model", "--port", "65536"],
            "invalid value '65536' for '--port",
        ),
        (
            vec!["run", "/nonexistent/model", "--served-model-name", " "],
            "served model name must not be empty",
        ),
        (
            vec!["run", "/nonexistent/model", "--bind", "127.0.0.1:8000"],
            "unexpected argument '--bind'",
        ),
        (
            vec![
                "run",
                "/nonexistent/model",
                "--embedding-table-storage",
                "bad",
            ],
            "possible values: disk, ram",
        ),
        (
            vec!["run", "/nonexistent/model", "--max-concurrency", "0"],
            "invalid value '0' for '--max-concurrency",
        ),
        (
            vec!["run", "/nonexistent/model", "--mtp-draft-tokens", "bad"],
            "invalid value 'bad' for '--mtp-draft-tokens",
        ),
        (
            vec!["run", "/nonexistent/model", "--prefix-cache-memory", "bad"],
            "invalid value 'bad' for '--prefix-cache-memory",
        ),
    ] {
        let output = flare(&arguments);
        assert_eq!(output.status.code(), Some(2), "{arguments:?}: {output:?}");
        assert!(output.stdout.is_empty(), "{arguments:?}: {output:?}");
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains(expected), "{arguments:?}: {error}");
        assert!(error.contains("--help"), "{arguments:?}: {error}");
    }
}

#[test]
fn configuration_is_loaded_by_the_native_entrypoint() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config.toml");
    let missing = temp.path().join("missing.safetensors");
    let config_arg = config.to_str().unwrap();
    let package_arg = missing.to_str().unwrap();
    let output = flare(&["run", package_arg, "--config", config_arg]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot read"));

    std::fs::write(&config, "[memory]\nreserve = 'invalid'\n").unwrap();
    let output = flare(&["run", package_arg, "--config", config_arg]);
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("memory.reserve"), "{error}");
    assert!(!error.contains("cannot access package"), "{error}");
    assert!(
        flare(&["run", "--config", config_arg, "--help"])
            .status
            .success()
    );

    let cache = temp.path().join("models");
    std::fs::write(
        &config,
        format!(
            "[server]\nhost = '::1'\n[paths]\npackage_cache = {:?}\n",
            cache.to_str().unwrap()
        ),
    )
    .unwrap();
    let output = flare(&["models", "list", "--json", "--config", config_arg]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(cache.is_dir(), "configured cache must be opened");
    let output = flare(&["run", package_arg, "--config", config_arg]);
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot access package"));
}
