//! Pinned primary metadata and payload resolution for selected deployments.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use prost::Message;

use super::{
    container::{self, Container, Slice},
    invalid,
    protection::VerifiedPackage,
    schema::{self, Catalog, wire},
};
use crate::Result;
use ring::digest;

pub const MANIFEST: &str = "flare.manifest";
pub const LOCATIONS: &str = "flare.locations";
pub const DERIVATIONS: &str = "flare.derivations";

/// Primary metadata is untrusted until its envelope has been authenticated.
pub struct Metadata {
    pub path: PathBuf,
    pub envelope: Vec<u8>,
    pub locations: wire::LocationSet,
    pub derivations: Option<wire::DerivationSet>,
    primary: Container,
}

/// The declared `__metadata__.format` of a safetensors header, read leniently within the largest
/// header this crate supports. Format dispatch uses this hint; the selected parser still validates
/// the complete framing and applies its own limits. Returns `None` for a header that cannot be
/// read or parsed, leaving the error to that parser.
pub fn declared_format(path: &Path) -> Result<Option<String>> {
    use std::io::{ErrorKind, Read as _};
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut prefix = [0u8; 8];
    if let Err(error) = file.read_exact(&mut prefix) {
        return if error.kind() == ErrorKind::UnexpectedEof {
            Ok(None)
        } else {
            Err(error.to_string().into())
        };
    }
    let length = u64::from_le_bytes(prefix);
    if length == 0 || length > container::MAX_HEADER {
        return Ok(None);
    }
    let mut header = vec![0u8; length as usize];
    if let Err(error) = file.read_exact(&mut header) {
        return if error.kind() == ErrorKind::UnexpectedEof {
            Ok(None)
        } else {
            Err(error.to_string().into())
        };
    }
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&header) else {
        return Ok(None);
    };
    Ok(value
        .get("__metadata__")
        .and_then(|metadata| metadata.get("format"))
        .and_then(|format| format.as_str())
        .map(str::to_owned))
}

impl Metadata {
    pub fn open(path: &Path) -> Result<Self> {
        let path = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.to_owned()
        };
        let primary = Container::open(&path)?;
        if primary.format() != container::PAYLOAD_FORMAT {
            return Err(invalid("expected a payload-format model package"));
        }
        let envelope = read_metadata(&primary.tensor(MANIFEST)?)?;
        let locations =
            wire::LocationSet::decode(read_metadata(&primary.tensor(LOCATIONS)?)?.as_slice())
                .map_err(|e| invalid(format!("invalid payload locations: {e}")))?;
        schema::validate_locations(&locations)?;
        let derivations = if primary.tensor_names().any(|name| name == DERIVATIONS) {
            Some(
                wire::DerivationSet::decode(
                    read_metadata(&primary.tensor(DERIVATIONS)?)?.as_slice(),
                )
                .map_err(|e| invalid(format!("invalid payload derivations: {e}")))?,
            )
        } else {
            None
        };
        Ok(Self {
            path,
            envelope,
            locations,
            derivations,
            primary,
        })
    }

    pub fn primary(&self) -> &Container {
        &self.primary
    }

    /// Open only the requested payloads, checking lengths against authenticated descriptors.
    pub fn payloads<'a>(
        &self,
        catalog: &Catalog,
        ids: impl IntoIterator<Item = &'a [u8]>,
    ) -> Result<PayloadFiles> {
        let locations: BTreeMap<_, _> = self
            .locations
            .locations
            .iter()
            .map(|location| (location.payload_id.as_slice(), location))
            .collect();
        let root = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut containers = BTreeMap::new();
        let mut payloads = BTreeMap::new();
        for id in ids {
            if payloads.contains_key(id) {
                continue;
            }
            let expected = catalog.length(id)?;
            let location = locations
                .get(id)
                .ok_or_else(|| invalid("selected deployment is missing a payload location"))?;
            let source = if location.file.is_empty() {
                self.primary.tensor(&location.tensor)?
            } else {
                if !containers.contains_key(&location.file) {
                    let (_, file) = container::open_relative(root, &location.file)?;
                    containers.insert(location.file.clone(), Container::open_pinned(file)?);
                }
                let companion = &containers[&location.file];
                if companion.format() != container::PAYLOAD_FORMAT {
                    return Err(invalid("invalid payload companion format"));
                }
                companion.tensor(&location.tensor)?
            };
            if source.len() != expected {
                return Err(invalid(
                    "payload tensor length differs from its signed descriptor",
                ));
            }
            payloads.insert(id.to_vec(), source);
        }
        Ok(PayloadFiles { payloads })
    }
}

/// Payloads consumed by the authenticated deployment and ranks, excluding acquisition hints.
pub fn required_payloads(package: &VerifiedPackage) -> Result<BTreeSet<Vec<u8>>> {
    if !package.is_executable() {
        return Err(invalid(
            "select a prepared deployment before resolving its payloads",
        ));
    }
    let mut ids = BTreeSet::new();
    let mut add = |source: Option<&wire::PayloadUse>| -> Result<()> {
        let slice = source
            .and_then(|source| source.slice.as_ref())
            .ok_or_else(|| invalid("payload consumer has no source slice"))?;
        package.catalog().check_slice(slice)?;
        ids.insert(slice.payload_id.clone());
        Ok(())
    };
    for asset in &package.package().assets {
        add(asset.payload.as_ref())?;
    }
    for rank in package.ranks() {
        for initializer in &rank.initializers {
            add(initializer.source.as_ref())?;
        }
        for module in &rank.modules {
            for global in &module.globals {
                add(global.contents.as_ref())?;
            }
        }
    }
    if let Some(engram) = package
        .deployment()
        .and_then(|deployment| deployment.engram.as_ref())
    {
        for table in &engram.tables {
            if table.shards.is_empty() {
                add(table.weights.as_ref())?;
                add(table.scales.as_ref())?;
            } else {
                for shard in &table.shards {
                    if package.rank(shard.rank).is_some() {
                        add(shard.weights.as_ref())?;
                        add(shard.scales.as_ref())?;
                    }
                }
            }
        }
    }
    for rank in package.ranks() {
        for resource in &rank.resources {
            let slice = resource
                .stored
                .as_ref()
                .ok_or_else(|| invalid("executable resource has no stored slice"))?;
            package.catalog().check_slice(slice)?;
            ids.insert(slice.payload_id.clone());
        }
    }
    Ok(ids)
}

/// Every slice owns its file handle independently of directory paths and metadata objects.
pub struct PayloadFiles {
    payloads: BTreeMap<Vec<u8>, Slice>,
}

impl PayloadFiles {
    pub(crate) fn from_slices(payloads: BTreeMap<Vec<u8>, Slice>) -> Self {
        Self { payloads }
    }

    pub fn get(&self, id: &[u8]) -> Result<&Slice> {
        self.payloads
            .get(id)
            .ok_or_else(|| invalid("payload is not in the selected closure"))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &Slice)> {
        self.payloads
            .iter()
            .map(|(id, slice)| (id.as_slice(), slice))
    }

    pub fn slice(&self, source: &wire::PayloadSlice) -> Result<Slice> {
        let payload = self.get(&source.payload_id)?;
        if source
            .offset
            .checked_add(source.nbytes)
            .is_none_or(|end| end > payload.len())
        {
            return Err(invalid("source slice exceeds payload tensor"));
        }
        Slice::new(
            payload.file().clone(),
            payload.offset() + source.offset,
            source.nbytes,
        )
    }

    pub fn verify(&self) -> Result<()> {
        self.verify_with_progress(&|_| {})
    }

    pub(crate) fn verify_with_progress(&self, progress: &(dyn Fn(u64) + Sync)) -> Result<()> {
        self.verify_with_workers_and_progress(verification_workers(self.payloads.len()), progress)
    }

    #[cfg(test)]
    fn verify_with_workers(&self, workers: usize) -> Result<()> {
        self.verify_with_workers_and_progress(workers, &|_| {})
    }

    fn verify_with_workers_and_progress(
        &self,
        workers: usize,
        progress: &(dyn Fn(u64) + Sync),
    ) -> Result<()> {
        let ids: Vec<&[u8]> = self.payloads.keys().map(Vec::as_slice).collect();
        if ids.is_empty() {
            return Ok(());
        }
        let workers = workers.max(1).min(ids.len());
        if workers == 1 {
            let mut buffer = vec![0u8; container::IO_CHUNK];
            for id in ids {
                verify_payload(self.payloads.get(id).expect("payload id from map"), id, &mut buffer, progress)?;
            }
            return Ok(());
        }

        use std::sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        };

        let cursor = AtomicUsize::new(0);
        let results = Mutex::new(
            (0..ids.len())
                .map(|_| None)
                .collect::<Vec<Option<Result<()>>>>(),
        );
        let mut spawn_error = None;
        std::thread::scope(|scope| {
            for _ in 0..workers {
                let worker = std::thread::Builder::new().spawn_scoped(scope, || {
                    let mut buffer = vec![0u8; container::IO_CHUNK];
                    loop {
                        let index = cursor.fetch_add(1, Ordering::Relaxed);
                        let Some(id) = ids.get(index) else {
                            break;
                        };
                        let source = self.payloads.get(*id).expect("payload id from map");
                        let result = verify_payload(source, id, &mut buffer, progress);
                        results.lock().expect("verification result mutex")[index] = Some(result);
                    }
                });
                if let Err(error) = worker {
                    spawn_error = Some(error);
                    break;
                }
            }
        });
        if let Some(error) = spawn_error {
            return Err(invalid(format!("spawn payload verifier: {error}")));
        }
        for result in results.into_inner().expect("verification result mutex") {
            result
                .expect("every payload verification worker must record a result")?;
        }
        Ok(())
    }
}

const MAX_VERIFY_WORKERS: usize = 8;

fn verification_workers(payloads: usize) -> usize {
    std::thread::available_parallelism()
        .map_or(1, |workers| workers.get())
        .min(MAX_VERIFY_WORKERS)
        .min(payloads)
}

fn verify_payload(
    source: &Slice,
    id: &[u8],
    buffer: &mut [u8],
    progress: &(dyn Fn(u64) + Sync),
) -> Result<()> {
    let mut context = digest::Context::new(&digest::SHA256);
    let mut offset = 0u64;
    while offset < source.len() {
        let count = (source.len() - offset).min(buffer.len() as u64) as usize;
        source.read_at(offset, &mut buffer[..count])?;
        context.update(&buffer[..count]);
        offset += count as u64;
        progress(count as u64);
    }
    if context.finish().as_ref() != id {
        return Err(invalid(
            "payload content does not match its signed identity",
        ));
    }
    Ok(())
}

fn read_metadata(source: &Slice) -> Result<Vec<u8>> {
    if source.is_empty() || source.len() > schema::MAX_BODY as u64 {
        return Err(invalid("invalid package metadata tensor length"));
    }
    let mut bytes = vec![0; source.len() as usize];
    source.read_at(0, &mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package::{
        PayloadId,
        container::{Source, Writer},
    };
    use std::{
        fs::{self, File},
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "flare-payload-reader-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn declared_format_leaves_truncated_prefix_header_and_oversized_headers_to_selected_parser() {
        let directory = Directory::new();
        let prefix = directory.0.join("prefix.safetensors");
        fs::write(&prefix, b"short").unwrap();
        assert_eq!(declared_format(&prefix).unwrap(), None);

        let header = directory.0.join("header.safetensors");
        let mut truncated = 128u64.to_le_bytes().to_vec();
        truncated.extend_from_slice(b"{}");
        fs::write(&header, truncated).unwrap();
        assert_eq!(declared_format(&header).unwrap(), None);

        let oversized = directory.0.join("oversized.safetensors");
        fs::write(&oversized, (container::MAX_HEADER + 1).to_le_bytes()).unwrap();
        assert_eq!(declared_format(&oversized).unwrap(), None);

        let missing = directory.0.join("missing-parent").join("package.safetensors");
        assert!(declared_format(&missing).is_err());
        assert!(
            declared_format(&directory.0).is_err(),
            "a real read error must not be treated as an unknown format"
        );
    }

    fn write(path: &Path, tensors: Vec<(&str, Vec<u8>)>) {
        let mut writer =
            Writer::new(File::create(path).unwrap(), container::PAYLOAD_FORMAT).unwrap();
        for (name, bytes) in tensors {
            writer
                .add(name, bytes.len() as u64, Source::Bytes(bytes))
                .unwrap();
        }
        writer.finish().unwrap();
    }

    fn primary(path: &Path, locations: Vec<wire::PayloadLocation>, payload: Option<Vec<u8>>) {
        let mut tensors = vec![
            (
                MANIFEST,
                wire::ProtectedPackage {
                    version: schema::FORMAT_VERSION,
                    ..Default::default()
                }
                .encode_to_vec(),
            ),
            (
                LOCATIONS,
                wire::LocationSet {
                    version: schema::FORMAT_VERSION,
                    locations,
                    available_deployments: vec![],
                }
                .encode_to_vec(),
            ),
        ];
        if let Some(bytes) = payload {
            tensors.push(("weights.layer-000", bytes));
        }
        write(path, tensors);
    }

    fn payload_files(
        directory: &Directory,
        entries: &[(PayloadId, Vec<u8>, bool)],
    ) -> PayloadFiles {
        let mut payloads = BTreeMap::new();
        for (index, (id, bytes, truncate)) in entries.iter().enumerate() {
            let path = directory.0.join(format!("payload-{index}.bin"));
            fs::write(&path, bytes).unwrap();
            let slice = Slice::new(
                std::sync::Arc::new(File::open(&path).unwrap()),
                0,
                bytes.len() as u64,
            )
            .unwrap();
            if *truncate {
                fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
            }
            payloads.insert(id.as_bytes().to_vec(), slice);
        }
        PayloadFiles { payloads }
    }

    fn sorted_payloads(count: u8) -> Vec<(PayloadId, Vec<u8>, bool)> {
        let mut entries: Vec<_> = (0..count)
            .map(|byte| {
                let bytes = vec![byte; 32];
                (PayloadId::of(&bytes), bytes, false)
            })
            .collect();
        entries.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
        entries
    }

    #[test]
    fn verification_reports_exact_bytes_with_bounded_chunks() {
        let directory = Directory::new();
        let entries: Vec<_> = (0..4u8).map(|byte| {
            let bytes = vec![byte; container::IO_CHUNK + 17];
            (PayloadId::of(&bytes), bytes, false)
        }).collect();
        let files = payload_files(&directory, &entries);
        for workers in [1, 4] {
            let completed = AtomicU64::new(0);
            files.verify_with_workers_and_progress(workers, &|bytes| {
                assert!(bytes > 0 && bytes <= container::IO_CHUNK as u64);
                completed.fetch_add(bytes, Ordering::Relaxed);
            }).unwrap();
            assert_eq!(completed.load(Ordering::Relaxed), 4 * (container::IO_CHUNK as u64 + 17));
        }
    }

    #[test]
    fn parallel_verification_rejects_nonfirst_corrupt_payload() {
        let directory = Directory::new();
        let mut entries = sorted_payloads(8);
        entries[3].1[0] ^= 0xff;
        let error = payload_files(&directory, &entries)
            .verify_with_workers(4)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("payload content does not match its signed identity"));
    }

    #[test]
    fn parallel_verification_rejects_nonfirst_truncated_payload() {
        let directory = Directory::new();
        let mut entries = sorted_payloads(8);
        entries[3].2 = true;
        let error = payload_files(&directory, &entries)
            .verify_with_workers(4)
            .unwrap_err();
        assert!(error.to_string().contains("read payload bytes"));
    }

    #[test]
    fn parallel_verification_checks_every_valid_payload() {
        let directory = Directory::new();
        let entries = sorted_payloads(8);
        payload_files(&directory, &entries)
            .verify_with_workers(4)
            .unwrap();
        for index in 0..entries.len() {
            let mut corrupt = entries.clone();
            corrupt[index].1[0] ^= 0xff;
            assert!(
                payload_files(&directory, &corrupt)
                    .verify_with_workers(4)
                    .is_err(),
                "payload {index} was not verified"
            );
        }
    }

    #[test]
    fn packed_payload_hash_and_slice_exclude_container_framing() {
        let directory = Directory::new();
        let bytes = b"prepared weights".to_vec();
        let id = PayloadId::of(&bytes);
        let path = directory.0.join("model.safetensors");
        primary(
            &path,
            vec![wire::PayloadLocation {
                payload_id: id.as_bytes().to_vec(),
                file: String::new(),
                tensor: "weights.layer-000".into(),
            }],
            Some(bytes.clone()),
        );
        let metadata = Metadata::open(&directory.0).unwrap();
        let mut catalog = Catalog::new();
        catalog
            .insert(&wire::PayloadDescriptor {
                id: id.as_bytes().to_vec(),
                nbytes: bytes.len() as u64,
            })
            .unwrap();
        let files = metadata
            .payloads(&catalog, [id.as_bytes().as_slice()])
            .unwrap();
        files.verify().unwrap();
        let slice = files
            .slice(&wire::PayloadSlice {
                payload_id: id.as_bytes().to_vec(),
                offset: 9,
                nbytes: 7,
            })
            .unwrap();
        let mut output = [0; 7];
        slice.read_at(0, &mut output).unwrap();
        assert_eq!(&output, b"weights");
        assert!(
            files
                .slice(&wire::PayloadSlice {
                    payload_id: id.as_bytes().to_vec(),
                    offset: u64::MAX,
                    nbytes: 1
                })
                .is_err()
        );
    }

    #[test]
    fn selected_closure_pins_companions_and_does_not_open_unselected_files() {
        let directory = Directory::new();
        let bytes = b"layer data".to_vec();
        let id = PayloadId::of(&bytes);
        let missing = PayloadId::of(b"absent deployment");
        let companion = directory.0.join("layer.safetensors");
        write(
            &companion,
            vec![(container::COMPANION_TENSOR, bytes.clone())],
        );
        primary(
            &directory.0.join("model.safetensors"),
            vec![
                wire::PayloadLocation {
                    payload_id: id.as_bytes().to_vec(),
                    file: "layer.safetensors".into(),
                    tensor: container::COMPANION_TENSOR.into(),
                },
                wire::PayloadLocation {
                    payload_id: missing.as_bytes().to_vec(),
                    file: "missing.safetensors".into(),
                    tensor: container::COMPANION_TENSOR.into(),
                },
            ],
            None,
        );
        let metadata = Metadata::open(&directory.0).unwrap();
        let mut catalog = Catalog::new();
        catalog
            .insert(&wire::PayloadDescriptor {
                id: id.as_bytes().to_vec(),
                nbytes: bytes.len() as u64,
            })
            .unwrap();
        catalog
            .insert(&wire::PayloadDescriptor {
                id: missing.as_bytes().to_vec(),
                nbytes: 17,
            })
            .unwrap();
        let files = metadata
            .payloads(&catalog, [id.as_bytes().as_slice()])
            .unwrap();
        assert!(
            metadata
                .payloads(&catalog, [missing.as_bytes().as_slice()])
                .is_err()
        );
        fs::rename(&companion, directory.0.join("retained.safetensors")).unwrap();
        write(
            &companion,
            vec![(container::COMPANION_TENSOR, b"wrong data".to_vec())],
        );
        files.verify().unwrap();
        assert!(
            metadata
                .payloads(&catalog, [id.as_bytes().as_slice()])
                .unwrap()
                .verify()
                .is_err()
        );
    }

    #[test]
    fn package_companions_lists_nested_payload_files() {
        let directory = Directory::new();
        let bytes = b"nested layer".to_vec();
        let id = PayloadId::of(&bytes);
        let nested = directory.0.join("weights/tp4/rank-0/layer.safetensors");
        fs::create_dir_all(nested.parent().unwrap()).unwrap();
        write(
            &nested,
            vec![(container::COMPANION_TENSOR, bytes.clone())],
        );
        primary(
            &directory.0.join("model.safetensors"),
            vec![wire::PayloadLocation {
                payload_id: id.as_bytes().to_vec(),
                file: "weights/tp4/rank-0/layer.safetensors".into(),
                tensor: container::COMPANION_TENSOR.into(),
            }],
            None,
        );
        assert_eq!(
            crate::runtime_package::package_companions(&directory.0).unwrap(),
            vec!["weights/tp4/rank-0/layer.safetensors"]
        );
    }
}
