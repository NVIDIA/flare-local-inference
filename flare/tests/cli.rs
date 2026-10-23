use std::process::{Command, Output};

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
            assert!(help.contains("--prefix-cache-disk-transport"), "{help}");
            assert!(help.contains("--host"), "{help}");
            assert!(help.contains("--port"), "{help}");
            assert!(help.contains("--served-model-name"), "{help}");
            assert!(help.contains("127.0.0.1"), "{help}");
            assert!(!help.contains("--bind"), "{help}");
        }
    }
}

#[test]
fn offline_download_and_run_share_the_cached_package_path() {
    let temp = tempfile::tempdir().unwrap();
    let commit = "0123456789012345678901234567890123456789";
    let directory = temp
        .path()
        .join("models--owner--model")
        .join("snapshots")
        .join(commit);
    std::fs::create_dir_all(&directory).unwrap();
    let package = directory.join("model.safetensors");
    // Discoverable format-4 metadata without publisher authentication.
    let envelope = b"\x08\x04\x22\x08\x08\x01\x12\x04\x0a\x02\x08\x04\x22\x02\x08\x02\x22\x02\x08\x03\x2a\x06\x08\x04\x12\x02\x12\x00";
    let header = serde_json::json!({
        "__metadata__": {"format": "flare"},
        "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
        "flare.metadata": {"dtype": "U8", "shape": [envelope.len()], "data_offsets": [0, envelope.len()]},
    });
    let mut bytes = 4088u64.to_le_bytes().to_vec();
    bytes.extend(serde_json::to_vec(&header).unwrap());
    bytes.resize(4096, b' ');
    bytes.extend(envelope);
    std::fs::write(&package, bytes).unwrap();
    let cache = temp.path().to_str().unwrap();
    let output = flare(&[
        "download",
        &format!("owner/model@{commit}"),
        "--offline",
        "--cache-dir",
        cache,
    ]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        package.to_str().unwrap()
    );

    let output = flare(&[
        "run",
        "owner/model",
        "--revision",
        commit,
        "--offline",
        "--cache-dir",
        cache,
    ]);
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("Verify package"), "{error}");
    assert!(!error.contains("cannot load NVIDIA"), "{error}");
    assert!(!error.contains("cannot access package"), "{error}");
}

#[test]
fn usage_errors_identify_the_problem_before_accessing_files() {
    for (arguments, expected) in [
        (vec![], "Usage: flare"),
        (vec!["run"], "<PACKAGE>"),
        (vec!["inspect"], "<PACKAGE>"),
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
