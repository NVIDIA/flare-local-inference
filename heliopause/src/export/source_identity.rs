//! Source compatibility identity for preparation metadata.

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

use super::{Result, invalid};

fn collect(directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound
                    | std::io::ErrorKind::NotADirectory
                    | std::io::ErrorKind::PermissionDenied
            ) =>
        {
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect(&path, files)?;
        } else if path.is_file()
            && path
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|value| matches!(value, "py" | "rs" | "c" | "cu" | "cuh" | "h"))
        {
            files.push(path);
        }
    }
    Ok(())
}

fn sort_key(path: &Path) -> Result<Vec<String>> {
    path.components()
        .map(|part| {
            let value = part
                .as_os_str()
                .to_str()
                .ok_or_else(|| invalid("source path is not UTF-8"))?;
            #[cfg(windows)]
            let value = value.to_lowercase();
            #[cfg(not(windows))]
            let value = value.to_owned();
            Ok(value)
        })
        .collect()
}

pub(crate) fn source_identity(roots: &[PathBuf]) -> Result<String> {
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 << 10];
    for root in roots {
        let mut files = Vec::new();
        collect(root, &mut files)?;
        let mut files = files
            .into_iter()
            .map(|path| Ok((sort_key(&path)?, path)))
            .collect::<Result<Vec<_>>>()?;
        files.sort_by(|left, right| left.0.cmp(&right.0));
        for (_, path) in files {
            let relative = path.strip_prefix(root)?;
            digest.update(
                relative
                    .to_str()
                    .ok_or_else(|| invalid("source path is not UTF-8"))?
                    .as_bytes(),
            );
            let mut source = File::open(path)?;
            loop {
                let count = source.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
            }
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_root_order_component_sort_and_source_suffixes() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("first");
        let second = directory.path().join("second");
        fs::create_dir_all(first.join("inner")).unwrap();
        fs::create_dir_all(&second).unwrap();
        fs::write(first.join("inner.py"), b"outer").unwrap();
        fs::write(first.join("inner").join("b.py"), b"nested").unwrap();
        fs::write(first.join("skip.txt"), b"excluded").unwrap();
        fs::write(second.join("a.rs"), vec![7; 200_000]).unwrap();
        let mut expected = Sha256::new();
        expected.update(Path::new("inner").join("b.py").to_str().unwrap().as_bytes());
        expected.update(b"nestedinner.pyoutera.rs");
        expected.update(vec![7; 200_000]);
        assert_eq!(
            source_identity(&[first.clone(), second.clone()]).unwrap(),
            format!("{:x}", expected.finalize())
        );
        assert_ne!(
            source_identity(&[first.clone(), second.clone()]).unwrap(),
            source_identity(&[second, first]).unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn hashes_file_symlinks_without_traversing_directory_symlinks() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("root");
        fs::create_dir(&root).unwrap();
        fs::write(directory.path().join("target.py"), b"payload").unwrap();
        symlink(directory.path().join("target.py"), root.join("link.py")).unwrap();
        symlink(directory.path(), root.join("cycle")).unwrap();
        assert_eq!(
            source_identity(&[root]).unwrap(),
            format!("{:x}", Sha256::digest(b"link.pypayload"))
        );
    }
}
