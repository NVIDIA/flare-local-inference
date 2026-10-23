//! Download prepared packages from hardware-specific Hugging Face revisions.

use std::{
    io::IsTerminal,
    path::{Path, PathBuf},
};

use hf_hub::{
    Cache, Repo, RepoType,
    api::sync::{Api, ApiBuilder},
};

use crate::cli::HubOptions;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn is_repository(path: &Path) -> bool {
    !path.exists()
        && !path.is_absolute()
        && !path.starts_with(".")
        && !path.starts_with("..")
        && path.to_str().is_some_and(|text| {
            !text.contains('\\') && !text.ends_with(".safetensors") && repository(text).is_ok()
        })
}

pub(crate) fn repository(text: &str) -> Result<(&str, Option<&str>)> {
    let (id, revision) = match text.split_once('@') {
        Some((id, revision)) if !revision.is_empty() => (id, Some(revision)),
        Some(_) => return Err("Hugging Face revision cannot be empty".into()),
        None => (text, None),
    };
    let parts: Vec<_> = id.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || part.starts_with(['.', '-'])
                || part.ends_with(['.', '-'])
                || part.contains("..")
                || part.contains("--")
                || !part
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
        })
    {
        return Err("expected a Hugging Face repository as ORG/NAME[@REVISION]".into());
    }
    if let Some(revision) = revision {
        validate_relative(revision)?;
    }
    Ok((id, revision))
}

fn validate_relative(value: &str) -> Result<()> {
    if value.is_empty()
        || value.contains(['\\', '\0'])
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == ".." || part.contains(':'))
    {
        return Err(format!("invalid repository-relative path or revision: {value:?}").into());
    }
    Ok(())
}

fn is_companion(filename: &str) -> bool {
    let Some(stem) = filename.strip_suffix(".safetensors") else {
        return false;
    };
    let stem = stem.rsplit('/').next().unwrap_or(stem);
    if stem.len() == 64
        && stem
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return true;
    }
    let Some((_, numbers)) = stem.rsplit_once("-data-") else {
        return false;
    };
    let bytes = numbers.as_bytes();
    bytes.len() == 8
        && &bytes[2..6] == b"-of-"
        && bytes[..2].iter().chain(&bytes[6..]).all(u8::is_ascii_digit)
}

fn choose_file(files: impl IntoIterator<Item = String>, requested: Option<&str>) -> Result<String> {
    if let Some(filename) = requested {
        validate_relative(filename)?;
        if !filename.ends_with(".safetensors") || is_companion(filename) {
            return Err("--filename must name a prepared .safetensors package".into());
        }
        return Ok(filename.to_owned());
    }
    let mut candidates: Vec<_> = files
        .into_iter()
        .filter(|file| file.ends_with(".safetensors") && !is_companion(file))
        .collect();
    candidates.sort();
    match candidates.as_slice() {
        [filename] => {
            validate_relative(filename)?;
            Ok(filename.clone())
        }
        [] => Err("revision contains no .safetensors package; choose a revision containing prepared Flare packages".into()),
        _ => Err(format!("revision contains several .safetensors files; select one with --filename:\n  {}", candidates.join("\n  ")).into()),
    }
}

pub fn download(source: &str, options: &HubOptions, gpu_count: usize) -> Result<PathBuf> {
    let access_token = if options.offline {
        None
    } else {
        token(options)?
    };
    download_with_token(source, options, gpu_count, access_token)
}

pub(crate) fn download_with_token(
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    access_token: Option<String>,
) -> Result<PathBuf> {
    let (id, given_revision) = repository(source)?;
    let revision = match options.revision.as_deref().or(given_revision) {
        Some(revision) => revision.to_owned(),
        None => crate::hardware::detect_isolated(gpu_count)?,
    };
    validate_relative(&revision)?;
    if let Some(filename) = &options.filename {
        choose_file([], Some(filename))?;
    }
    let cache_dir = match &options.cache_dir {
        Some(path) => std::path::absolute(path)?,
        None => crate::environment::home_directory()?.join(".cache/huggingface/hub"),
    };
    let cache = Cache::new(cache_dir);
    let repo = Repo::with_revision(id.to_owned(), RepoType::Model, revision.clone());
    eprintln!("Package source: {id}@{revision}");
    if options.offline {
        return cached(&cache, &repo, options.filename.as_deref());
    }
    let api = ApiBuilder::from_cache(cache.clone())
        .with_token(access_token)
        .with_endpoint("https://huggingface.co".into())
        .with_progress(std::io::stderr().is_terminal())
        .with_retries(3)
        .build()?;
    fetch(&api, &cache, &repo, options.filename.as_deref())
}

/// The access token named by `--token-file`, for private repositories.
pub(crate) fn token(options: &HubOptions) -> Result<Option<String>> {
    options
        .token_file
        .as_ref()
        .map(|path| {
            let value = std::fs::read_to_string(path)
                .map_err(|error| format!("cannot read token file {}: {error}", path.display()))?;
            let token = value.trim();
            if token.is_empty() {
                return Err("Hugging Face token file is empty".into());
            }
            Ok(token.to_owned())
        })
        .transpose()
}

fn fetch(api: &Api, cache: &Cache, repo: &Repo, requested: Option<&str>) -> Result<PathBuf> {
    let info = api.repo(repo.clone()).info()?;
    if info.sha.len() != 40 || !info.sha.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("Hugging Face returned an invalid commit ID".into());
    }
    let filename = choose_file(
        info.siblings.into_iter().map(|file| file.rfilename),
        requested,
    )?;
    let pinned = Repo::with_revision(repo.url(), RepoType::Model, info.sha.clone());
    eprintln!("Resolved package: {}@{} / {filename}", repo.url(), info.sha);
    let path = fetch_file(api, cache, &pinned, &filename)?;
    let companions = flare_helios::runtime_package::package_companions(&path)?;
    let parent = Path::new(&filename).parent().unwrap();
    for name in &companions {
        let relative = parent.join(name).to_string_lossy().replace('\\', "/");
        fetch_file(api, cache, &pinned, &relative)?;
    }
    let path = package_view(cache, repo, &info.sha, &filename, &companions)?;
    cache.repo(repo.clone()).create_ref(&info.sha)?;
    Ok(path)
}

/// Reuse the repository's content-addressed blob even when this revision has no pointer yet.
fn fetch_file(api: &Api, cache: &Cache, pinned: &Repo, filename: &str) -> Result<PathBuf> {
    let cached_repo = cache.repo(pinned.clone());
    let path = cached_repo.pointer_path(pinned.revision()).join(filename);
    if path.is_file() {
        return Ok(path);
    }
    let remote = api.repo(pinned.clone());
    let metadata = api.metadata(&remote.url(filename))?;
    let etag = metadata.etag();
    if metadata.commit_hash() != pinned.revision()
        || etag.is_empty()
        || !etag
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
    {
        return Err("Hugging Face returned an invalid file identity".into());
    }
    let blob = cached_repo.blob_path(etag);
    std::fs::create_dir_all(blob.parent().unwrap())?;
    // hf-hub locks its transfer but replaces an existing blob. Serialize our lookup
    // and transfer together so concurrent Flare downloads preserve shared inodes.
    let _lock = blob_lock(&blob)?;
    if path.is_file() {
        return Ok(path);
    }
    match std::fs::symlink_metadata(&blob) {
        Ok(info) => {
            if !info.is_file() || info.len() != metadata.size() as u64 {
                return Err(format!("invalid cached Hugging Face blob: {}", blob.display()).into());
            }
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::hard_link(&blob, &path).map_err(|error| {
                format!(
                    "cannot hardlink cached package file {}: {error}",
                    path.display()
                )
            })?;
            eprintln!("Reusing cached model file: {filename}");
            Ok(path)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Downloading model file: {filename}...");
            let downloaded = remote.download(filename)?;
            // Without symlink privileges, hf-hub moves the blob into the snapshot.
            // Retain a cache link so other revisions can still reuse those bytes.
            #[cfg(windows)]
            if !blob.exists() {
                std::fs::hard_link(&downloaded, &blob)?;
            }
            Ok(downloaded)
        }
        Err(error) => Err(error.into()),
    }
}

fn blob_lock(blob: &Path) -> Result<std::fs::File> {
    let path = blob.with_extension("flare.lock");
    #[cfg(unix)]
    let lock = {
        use std::os::unix::{fs::OpenOptionsExt, io::AsRawFd};
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)?;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        file
    };
    #[cfg(windows)]
    let lock = {
        let file = crate::windows::private_file(&path, false)?;
        crate::windows::lock(&file)?;
        file
    };
    Ok(lock)
}

fn snapshot(cache: &Cache, repo: &Repo, commit: &str) -> PathBuf {
    cache
        .path()
        .join(repo.folder_name())
        .join("snapshots")
        .join(commit)
}

fn cached(cache: &Cache, repo: &Repo, requested: Option<&str>) -> Result<PathBuf> {
    let revision = repo.revision();
    let commit =
        if revision.len() == 40 && revision.bytes().all(|c| c.is_ascii_hexdigit()) {
            revision.to_owned()
        } else {
            let path = cache
                .path()
                .join(repo.folder_name())
                .join("refs")
                .join(revision);
            std::fs::read_to_string(&path).map_err(|error| {
            format!("cannot resolve cached revision {revision}: {error}; download it online first")
        })?.trim().to_owned()
        };
    if commit.len() != 40 || !commit.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("cached revision contains an invalid commit ID".into());
    }
    let directory = snapshot(cache, repo, &commit);
    let filename = if let Some(filename) = requested {
        choose_file([], Some(filename))?
    } else {
        let mut files = Vec::new();
        collect_files(&directory, &directory, &mut files)?;
        choose_file(files, None)?
    };
    let path = directory.join(&filename);
    if !path.is_file() {
        return Err(format!("package is not cached: {}", path.display()).into());
    }
    let companions = flare_helios::runtime_package::package_companions(&path)?;
    package_view(cache, repo, &commit, &filename, &companions)
}

/// Load regular snapshot entries directly. A hardlink view of symlink snapshots
/// preserves companion paths when the primary is canonicalized, without copying bytes.
fn package_view(
    cache: &Cache,
    repo: &Repo,
    commit: &str,
    filename: &str,
    companions: &[String],
) -> Result<PathBuf> {
    let primary = snapshot(cache, repo, commit).join(filename);
    if companions.is_empty() {
        return Ok(primary);
    }
    if std::iter::once(primary.clone())
        .chain(
            companions
                .iter()
                .map(|name| primary.parent().unwrap().join(name)),
        )
        .all(|path| std::fs::symlink_metadata(path).is_ok_and(|info| info.is_file()))
    {
        return Ok(primary);
    }
    let view = cache
        .path()
        .join(repo.folder_name())
        .join("flare-packages")
        .join(commit)
        .join(filename);
    std::fs::create_dir_all(&view)?;
    let basename = primary
        .file_name()
        .ok_or("missing package filename")?
        .to_string_lossy()
        .into_owned();
    for name in companions.iter().chain(std::iter::once(&basename)) {
        let source = primary
            .parent()
            .unwrap()
            .join(name)
            .canonicalize()
            .map_err(|e| format!("package file is not cached: {name}: {e}"))?;
        if !source.is_file() {
            return Err(format!("package file is not regular: {name}").into());
        }
        let target = view.join(name);
        match std::fs::hard_link(&source, &target) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let info = std::fs::symlink_metadata(&target)?;
                if !info.is_file() || info.len() != std::fs::metadata(&source)?.len() {
                    return Err(format!("invalid cached package view: {}", target.display()).into());
                }
            }
            Err(e) => {
                return Err(format!(
                    "cannot hardlink package cache view {}: {e}",
                    target.display()
                )
                .into());
            }
        }
    }
    Ok(view.join(basename))
}

fn collect_files(root: &Path, directory: &Path, files: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            collect_files(root, &entry.path(), files)?;
        } else if entry.path().is_file() {
            files.push(
                entry
                    .path()
                    .strip_prefix(root)?
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        thread,
        time::Duration,
    };

    const COMMIT: &str = "0123456789012345678901234567890123456789";

    pub(crate) fn fixture(companions: &[&str]) -> Vec<u8> {
        fn bytes(field: u8, body: &[u8]) -> Vec<u8> {
            let mut out = vec![field * 8 + 2];
            let mut n = body.len();
            while n >= 128 {
                out.push((n as u8 & 127) | 128);
                n >>= 7;
            }
            out.push(n as u8);
            out.extend_from_slice(body);
            out
        }
        // ProtectedIndex -> PublisherMetadata -> ModelIndex, with unsigned discovery metadata.
        let index = [8, if companions.is_empty() { 4 } else { 5 }];
        let metadata = bytes(1, &index);
        let mut envelope = vec![8, 4];
        for (kind, body) in [(1, metadata), (2, vec![]), (3, vec![])] {
            envelope.extend(bytes(4, &[vec![8, kind], bytes(2, &body)].concat()));
        }
        let mut data = Vec::new();
        for name in companions {
            data.extend(bytes(4, &bytes(3, name.as_bytes())));
        }
        let rank = [bytes(2, &[]), bytes(5, &data)].concat();
        envelope.extend(bytes(5, &[vec![8, 4], bytes(2, &rank)].concat()));
        let header = serde_json::json!({
            "__metadata__": {"format": "flare"},
            "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
            "flare.metadata": {"dtype": "U8", "shape": [envelope.len()], "data_offsets": [0, envelope.len()]},
        });
        let mut result = 4088u64.to_le_bytes().to_vec();
        result.extend(serde_json::to_vec(&header).unwrap());
        result.resize(4096, b' ');
        result.extend(envelope);
        result
    }

    #[test]
    fn local_paths_and_repository_revisions_are_distinct() {
        assert!(is_repository(Path::new("owner/model")));
        assert!(is_repository(Path::new("owner/model@release/gb202")));
        for path in [
            "model.safetensors",
            "dir/model.safetensors",
            "./owner/model",
            "/owner/model",
            "../owner/model",
            "C:\\models\\model.safetensors",
        ] {
            assert!(!is_repository(Path::new(path)), "{path}");
        }
        assert_eq!(
            repository("owner/model@gb10").unwrap(),
            ("owner/model", Some("gb10"))
        );
        for source in [
            "owner",
            "owner/model/extra",
            "../model",
            "owner/model@",
            "owner/model@../../other",
        ] {
            assert!(repository(source).is_err(), "{source}");
        }
    }

    #[test]
    fn package_selection_rejects_ambiguous_and_escaping_filenames() {
        assert_eq!(
            choose_file(["README.md".into(), "model.safetensors".into()], None).unwrap(),
            "model.safetensors"
        );
        let error = choose_file(["tp4.safetensors".into(), "tp2.safetensors".into()], None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--filename"));
        assert!(error.contains("tp2.safetensors\n  tp4.safetensors"));
        for file in [
            "../model.safetensors",
            "/model.safetensors",
            "C:\\model.safetensors",
            "weights.json",
        ] {
            assert!(choose_file([], Some(file)).is_err(), "{file}");
        }
    }

    #[test]
    fn offline_cache_reads_python_style_snapshots_without_commit_refs() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), &fixture(&[])).unwrap();
        cache.repo(repo.clone()).create_ref(COMMIT).unwrap();
        assert_eq!(
            cached(&cache, &repo, None).unwrap(),
            directory.join("model.safetensors")
        );
        let pinned = Repo::with_revision(repo.url(), RepoType::Model, COMMIT.into());
        assert_eq!(
            cached(&cache, &pinned, None).unwrap(),
            directory.join("model.safetensors")
        );
        assert!(cached(&cache, &repo, Some("missing.safetensors")).is_err());
        std::fs::write(directory.join("another.safetensors"), b"other").unwrap();
        assert!(
            cached(&cache, &repo, None)
                .unwrap_err()
                .to_string()
                .contains("--filename")
        );
        cache
            .repo(repo.clone())
            .create_ref("../../outside")
            .unwrap();
        assert!(
            cached(&cache, &repo, None)
                .unwrap_err()
                .to_string()
                .contains("invalid commit")
        );
    }

    fn mock_hub(responses: Vec<(String, Vec<u8>)>) -> (String, thread::JoinHandle<()>) {
        mock_hub_steps(responses, |_| {})
    }

    fn mock_hub_steps(
        responses: Vec<(String, Vec<u8>)>,
        mut after_response: impl FnMut(usize) + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let worker = thread::spawn(move || {
            for (step, (expected, response)) in responses.into_iter().enumerate() {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(&mut socket);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert!(line.starts_with(&format!("GET {expected} ")), "{line}");
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                }
                socket.write_all(&response).unwrap();
                after_response(step);
            }
        });
        (endpoint, worker)
    }

    fn metadata_response() -> Vec<u8> {
        revision_response(COMMIT)
    }

    fn revision_response(commit: &str) -> Vec<u8> {
        let body =
            format!(r#"{{"sha":"{commit}","siblings":[{{"rfilename":"model.safetensors"}}]}}"#);
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ).into_bytes()
    }

    fn file_metadata(commit: &str, etag: &str, size: usize) -> Vec<u8> {
        format!("HTTP/1.1 206 Partial Content\r\nx-repo-commit: {commit}\r\netag: {etag}\r\nContent-Range: bytes 0-0/{size}\r\nContent-Length: 1\r\nConnection: close\r\n\r\np").into_bytes()
    }

    fn file_response(body: &[u8]) -> Vec<u8> {
        [
            format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .into_bytes(),
            body.to_vec(),
        ]
        .concat()
    }

    #[cfg(unix)]
    fn assert_same_file(left: &Path, right: &Path) {
        use std::os::unix::fs::MetadataExt;
        let left = std::fs::metadata(left).unwrap();
        let right = std::fs::metadata(right).unwrap();
        assert_eq!((left.dev(), left.ino()), (right.dev(), right.ino()));
    }

    fn test_api(cache: &Cache, endpoint: String) -> Api {
        ApiBuilder::from_cache(cache.clone())
            .with_token(None)
            .with_endpoint(endpoint)
            .with_progress(false)
            .build()
            .unwrap()
    }

    #[test]
    fn download_resolves_hardware_branch_then_fetches_the_exact_commit() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let resolve = format!("/owner/model/resolve/{COMMIT}/model.safetensors");
        let package = fixture(&[]);
        let (endpoint, worker) = mock_hub(vec![
            (
                "/api/models/owner/model/revision/gb202".into(),
                metadata_response(),
            ),
            (
                resolve.clone(),
                file_metadata(COMMIT, "fixture", package.len()),
            ),
            (
                resolve.clone(),
                file_metadata(COMMIT, "fixture", package.len()),
            ),
            (resolve, file_response(&package)),
        ]);
        let api = test_api(&cache, endpoint);
        let path = fetch(&api, &cache, &repo, None).unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), package);
        assert_eq!(cached(&cache, &repo, None).unwrap(), path);
    }

    #[test]
    fn online_cache_hit_resolves_revision_without_redownloading_weights() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), &fixture(&[])).unwrap();
        let (endpoint, worker) = mock_hub(vec![(
            "/api/models/owner/model/revision/gb10".into(),
            metadata_response(),
        )]);
        let path = fetch(&test_api(&cache, endpoint), &cache, &repo, None).unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fixture(&[]));
    }
    #[test]
    fn offline_shards_require_complete_snapshots_and_use_regular_file_views() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        let name = "model-data-01-of-01.safetensors";
        std::fs::write(directory.join("model.safetensors"), fixture(&[name])).unwrap();
        assert!(
            cached(&cache, &repo, None)
                .unwrap_err()
                .to_string()
                .contains("not cached")
        );
        let blob = temp.path().join("blob");
        std::fs::write(&blob, b"companion data").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&blob, directory.join(name)).unwrap();
        #[cfg(windows)]
        std::fs::hard_link(&blob, directory.join(name)).unwrap();
        let path = cached(&cache, &repo, None).unwrap();
        #[cfg(unix)]
        {
            assert_ne!(path, directory.join("model.safetensors"));
            assert_same_file(&blob, &path.parent().unwrap().join(name));
        }
        #[cfg(windows)]
        assert_eq!(path, directory.join("model.safetensors"));
        assert!(
            std::fs::symlink_metadata(path.parent().unwrap().join(name))
                .unwrap()
                .is_file()
        );
        assert_eq!(
            std::fs::read(path.parent().unwrap().join(name)).unwrap(),
            b"companion data"
        );
        assert_eq!(cached(&cache, &repo, None).unwrap(), path);
        assert_eq!(
            choose_file(["model.safetensors".into(), name.into()], None).unwrap(),
            "model.safetensors"
        );
        assert!(choose_file([], Some(name)).is_err());
    }
    #[test]
    fn companions_download_from_the_primary_commit() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let name = "model-data-01-of-01.safetensors";
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), fixture(&[name])).unwrap();
        let resolve = format!("/owner/model/resolve/{COMMIT}/{name}");
        let (endpoint, worker) = mock_hub(vec![
            (
                "/api/models/owner/model/revision/gb202".into(),
                metadata_response(),
            ),
            (resolve.clone(), file_metadata(COMMIT, "data-fixture", 4)),
            (resolve.clone(), file_metadata(COMMIT, "data-fixture", 4)),
            (resolve, file_response(b"data")),
        ]);
        let path = fetch(&test_api(&cache, endpoint), &cache, &repo, None).unwrap();
        worker.join().unwrap();
        assert_eq!(
            std::fs::read(path.parent().unwrap().join(name)).unwrap(),
            b"data"
        );
        assert_eq!(cached(&cache, &repo, None).unwrap(), path);
    }

    #[test]
    fn regular_snapshot_shards_load_in_place_offline() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
        let directory = snapshot(&cache, &repo, COMMIT).join("nested");
        std::fs::create_dir_all(&directory).unwrap();
        let name = format!("{}.safetensors", "a".repeat(64));
        let primary = directory.join("model.safetensors");
        std::fs::write(&primary, fixture(&[&name])).unwrap();
        std::fs::write(directory.join(&name), b"data").unwrap();
        assert_eq!(cached(&cache, &repo, None).unwrap(), primary);
        assert!(
            !temp
                .path()
                .join(repo.folder_name())
                .join("flare-packages")
                .exists()
        );
    }

    #[test]
    fn cached_primary_blob_is_linked_without_downloading_its_body() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let blob = cache.repo(repo.clone()).blob_path("fixture");
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        let package = fixture(&[]);
        std::fs::write(&blob, &package).unwrap();
        let (endpoint, worker) = mock_hub(vec![
            (
                "/api/models/owner/model/revision/gb10".into(),
                metadata_response(),
            ),
            (
                format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
                file_metadata(COMMIT, "fixture", package.len()),
            ),
        ]);
        let path = fetch(&test_api(&cache, endpoint), &cache, &repo, None).unwrap();
        worker.join().unwrap();
        assert_eq!(
            path,
            snapshot(&cache, &repo, COMMIT).join("model.safetensors")
        );
        assert_eq!(std::fs::read(&path).unwrap(), package);
        #[cfg(unix)]
        assert_same_file(&blob, &path);
        assert_eq!(cached(&cache, &repo, None).unwrap(), path);
    }

    #[test]
    fn revisions_share_downloaded_payload_without_replacing_the_blob() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let other_commit = "abcdef0123456789012345678901234567890123";
        let first_repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let second_repo =
            Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let name = format!("{}.safetensors", "a".repeat(64));
        for commit in [COMMIT, other_commit] {
            let directory = snapshot(&cache, &first_repo, commit);
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join("model.safetensors"), fixture(&[&name])).unwrap();
        }
        let resolve = format!("/owner/model/resolve/{COMMIT}/{name}");
        let (endpoint, worker) = mock_hub(vec![
            (
                "/api/models/owner/model/revision/gb10".into(),
                metadata_response(),
            ),
            (resolve.clone(), file_metadata(COMMIT, "data-fixture", 4)),
            (resolve.clone(), file_metadata(COMMIT, "data-fixture", 4)),
            (resolve, file_response(b"data")),
            (
                "/api/models/owner/model/revision/gb202".into(),
                revision_response(other_commit),
            ),
            (
                format!("/owner/model/resolve/{other_commit}/{name}"),
                file_metadata(other_commit, "data-fixture", 4),
            ),
        ]);
        let api = test_api(&cache, endpoint);
        let first = fetch(&api, &cache, &first_repo, None).unwrap();
        let second = fetch(&api, &cache, &second_repo, None).unwrap();
        worker.join().unwrap();
        assert_eq!(
            second,
            snapshot(&cache, &second_repo, other_commit).join("model.safetensors")
        );
        let blob = cache.repo(first_repo.clone()).blob_path("data-fixture");
        assert_eq!(std::fs::read(&blob).unwrap(), b"data");
        #[cfg(unix)]
        {
            assert_same_file(&blob, &first.parent().unwrap().join(&name));
            assert_same_file(&blob, &second.parent().unwrap().join(&name));
        }
        assert_eq!(cached(&cache, &first_repo, None).unwrap(), first);
        assert_eq!(cached(&cache, &second_repo, None).unwrap(), second);
    }

    #[test]
    fn concurrent_fetches_transfer_the_payload_once() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
        let blob = cache.repo(repo.clone()).blob_path("fixture");
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        let lock = blob_lock(&blob).unwrap();
        let package = fixture(&[]);
        let resolve = format!("/owner/model/resolve/{COMMIT}/model.safetensors");
        let (ready, waiting) = std::sync::mpsc::channel();
        let (endpoint, worker) = mock_hub_steps(
            vec![
                (
                    resolve.clone(),
                    file_metadata(COMMIT, "fixture", package.len()),
                ),
                (
                    resolve.clone(),
                    file_metadata(COMMIT, "fixture", package.len()),
                ),
                (
                    resolve.clone(),
                    file_metadata(COMMIT, "fixture", package.len()),
                ),
                (resolve, file_response(&package)),
            ],
            move |step| {
                if step == 1 {
                    ready.send(()).unwrap();
                }
            },
        );
        let clients: Vec<_> = (0..2)
            .map(|_| {
                let cache = cache.clone();
                let repo = repo.clone();
                let endpoint = endpoint.clone();
                thread::spawn(move || {
                    fetch_file(
                        &test_api(&cache, endpoint),
                        &cache,
                        &repo,
                        "model.safetensors",
                    )
                    .map_err(|error| error.to_string())
                })
            })
            .collect();
        // Both callers have missed the snapshot and fetched its identity before either
        // can publish a blob. The waiting caller must reuse the completed transfer.
        waiting.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(lock);
        for client in clients {
            let path = client.join().unwrap().unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), package);
            #[cfg(unix)]
            assert_same_file(&blob, &path);
        }
        worker.join().unwrap();
    }

    #[test]
    fn truncated_cached_blobs_are_rejected_without_modifying_them() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
        let blob = cache.repo(repo.clone()).blob_path("fixture");
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, b"truncated").unwrap();
        let (endpoint, worker) = mock_hub(vec![(
            format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
            file_metadata(COMMIT, "fixture", 100),
        )]);
        let error = fetch_file(
            &test_api(&cache, endpoint),
            &cache,
            &repo,
            "model.safetensors",
        )
        .unwrap_err();
        worker.join().unwrap();
        assert!(
            error
                .to_string()
                .contains("invalid cached Hugging Face blob")
        );
        assert_eq!(std::fs::read(blob).unwrap(), b"truncated");
        assert!(!snapshot(&cache, &repo, COMMIT).exists());
    }

    #[test]
    fn cache_lookup_rejects_escaping_etags_and_unpinned_commits() {
        for (commit, etag) in [
            (COMMIT, "../outside"),
            (COMMIT, ""),
            ("other-commit", "fixture"),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let cache = Cache::new(temp.path().to_owned());
            let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
            let (endpoint, worker) = mock_hub(vec![(
                format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
                file_metadata(commit, etag, 100),
            )]);
            let error = fetch_file(
                &test_api(&cache, endpoint),
                &cache,
                &repo,
                "model.safetensors",
            )
            .unwrap_err();
            worker.join().unwrap();
            assert!(
                error.to_string().contains("invalid file identity"),
                "{error}"
            );
            assert!(!snapshot(&cache, &repo, COMMIT).exists());
        }
    }
}
