use crate::{
    Result, invalid,
    record::{self, Descriptor},
};
use flare_eos::direct_update_file as open;
use flare_selene::shared::record::Header;
use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub use flare_selene::shared::record::DirectoryLock as RootLock;

pub struct DiskExtent {
    pub path: PathBuf,
    retired: AtomicBool,
}
impl DiskExtent {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            retired: AtomicBool::new(false),
        }
    }
    pub fn reader(&self) -> Result<Arc<File>> {
        open(&self.path, false).map(Arc::new)
    }
    pub fn keep(&self) {
        self.retired.store(false, Ordering::Release);
    }
    pub fn retire(&self) {
        self.retired.store(true, Ordering::Release);
    }
}
impl Drop for DiskExtent {
    fn drop(&mut self) {
        if self.retired.load(Ordering::Acquire) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
pub use flare_selene::shared::disk::{Faults, IoGate};
pub struct Recovered {
    pub descriptor: Descriptor,
    pub records: Vec<(Header, bool, Arc<DiskExtent>)>,
}

pub fn recover(root: &Path, pool_name: &str) -> Result<Vec<Recovered>> {
    use flare_selene::shared::record::{DirectoryLock, directory_name, scan};
    let mut namespaces = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let directory = entry.path();
        let lock = DirectoryLock::acquire(&directory).map_err(|e| invalid(e.to_string()))?;
        let mut header = None;
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() || entry.path().extension().is_none_or(|e| e != "ckpt")
            {
                continue;
            }
            let mut block = [0; 4096];
            match File::open(entry.path())?.read_exact(&mut block) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => continue,
                Err(e) => return Err(e),
            }
            if let Ok(value) = Header::decode(&block) {
                if entry.file_name().to_str().and_then(record::parse_file_name) == Some(value.key)
                    && entry.path().parent().unwrap().file_name().unwrap()
                        == directory_name(&value.identity, &value.fingerprint).as_str()
                {
                    header = Some(value);
                    break;
                }
            }
        }
        let identity = header.as_ref().map_or([0; 32], |h| h.identity);
        let fingerprint = header.as_ref().map_or([0; 16], |h| h.fingerprint);
        let records = scan(
            &lock,
            &identity,
            &fingerprint,
            header.as_ref().map_or(0, |h| h.page_tokens),
            header.as_ref().map_or(&[], |h| h.ranks.as_slice()),
        )
        .map_err(|e| invalid(e.to_string()))?;
        if let Some(header) = header {
            namespaces.push(Recovered {
                descriptor: Descriptor {
                    identity,
                    fingerprint,
                    page_tokens: header.page_tokens,
                    ranks: header.ranks,
                    pool_name: pool_name.to_owned(),
                },
                records: records
                    .into_iter()
                    .map(|r| (r.header, r.row, Arc::new(DiskExtent::new(r.path))))
                    .collect(),
            });
        }
    }
    Ok(namespaces)
}
