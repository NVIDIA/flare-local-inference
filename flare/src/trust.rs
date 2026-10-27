//! Publisher acceptance and the persistent signing-key trust store.
//!
//! Loading a package: read its unverified claim, resolve the signer's identity from the
//! publisher's repository whenever it can be consulted, verify the signatures, disclose
//! publisher, identity, source, and license, and obtain the acceptance string. Helios
//! requires it to match the signed publisher namespace exactly; it records trust and license
//! acceptance but is not executable key material. Remembered acceptance suppresses the
//! disclosure; it never skips the repository check.

#[cfg(unix)]
use std::fs::OpenOptions;
#[cfg(unix)]
use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};

use flare_helios::runtime_package::{AuthorizedPackage, PackageClaim, Publisher, VerifiedPackage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::identity::{self, Resolution};
use std::{
    fs::{self, File},
    io::{self, BufRead, IsTerminal, Read, Write},
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SLOT_SIZE: usize = 65536;

pub(crate) fn publishers(
    paths: &[PathBuf],
    development: bool,
    home: impl FnOnce() -> io::Result<PathBuf>,
) -> Result<Vec<Publisher>> {
    let default;
    let paths = if development && paths.is_empty() {
        default = home()?.join(".config/flare/development-publisher.json");
        if !default.try_exists()? {
            return Ok(Vec::new());
        }
        std::slice::from_ref(&default)
    } else {
        paths
    };
    let mut publishers = Vec::new();
    for path in paths {
        let mut bytes = Vec::new();
        File::open(path)
            .map_err(|error| format!("cannot read publisher profile {}: {error}", path.display()))?
            .take((SLOT_SIZE + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > SLOT_SIZE {
            return Err(
                format!("publisher profile file exceeds 64 KiB: {}", path.display()).into(),
            );
        }
        publishers.extend(
            serde_json::from_slice::<Vec<Publisher>>(&bytes).map_err(|error| {
                format!("invalid publisher profile {}: {error}", path.display())
            })?,
        );
    }
    unique_profiles(publishers)
}

fn unique_profiles(publishers: Vec<Publisher>) -> Result<Vec<Publisher>> {
    // One key may be listed under several namespaces: a publisher promoting a package keeps the
    // key and changes the namespace. Identity is the (key, namespace) pair, so a differing display
    // name for the same pair is the only real conflict; namespaces never merge into one another.
    let mut unique = std::collections::HashMap::new();
    let mut result = Vec::new();
    for publisher in publishers {
        publisher.validate()?;
        let identity = (publisher.fingerprint(), publisher.namespace.clone());
        if let Some(previous) = unique.insert(identity, publisher.clone()) {
            if previous != publisher {
                return Err(
                    "conflicting identities for the same publisher key and namespace".into(),
                );
            }
        } else {
            result.push(publisher);
        }
    }
    Ok(result)
}

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Store {
    trusted: Vec<Entry>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    fingerprint: String,
    namespace: String,
    /// The verification key, remembered so later loads verify signatures without a profile or
    /// a network request. Entries recorded before it was stored lack it and use profiles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    public_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    /// How the key's identity was established at acceptance: `huggingface.co` when the
    /// publisher's repository listed it, otherwise `profile`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    identity: Option<String>,
}
impl Entry {
    fn profile(&self) -> Option<Publisher> {
        let key = self.public_key.as_deref()?;
        let mut public_key = [0u8; 32];
        for (byte, pair) in public_key.iter_mut().zip(key.as_bytes().chunks(2)) {
            *byte = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
        }
        Some(Publisher {
            name: self
                .name
                .clone()
                .unwrap_or_else(|| "Remembered signer".into()),
            namespace: self.namespace.clone(),
            public_key,
            development: self.namespace == "local-development",
        })
    }
}
/// Public profiles remembered in an explicit trust store. Entries without a
/// stored key cannot verify signatures themselves and are omitted.
pub(crate) fn trusted_profiles(store: Option<&Path>) -> Result<Vec<Publisher>> {
    let store = load_store(&store_path(store)?)?;
    Ok(store.trusted.iter().filter_map(Entry::profile).collect())
}

/// The verification key remembered for the claim's signer and namespace when the store holds
/// one, so a package accepted earlier verifies without a profile or a network request.
pub(crate) fn remembered_profile(
    claim: &PackageClaim,
    store: Option<&Path>,
) -> Result<Option<Publisher>> {
    let store = load_store(&store_path(store)?)?;
    Ok(store
        .trusted
        .iter()
        .find(|e| e.fingerprint == claim.signer && e.namespace == claim.namespace)
        .and_then(Entry::profile))
}
fn fingerprint_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn store_path(given: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = given {
        return Ok(path.to_owned());
    }
    #[cfg(windows)]
    return Ok(crate::windows::trust_path()?);
    #[cfg(unix)]
    {
        let mut storage = vec![0u8; 65536];
        let mut record = std::mem::MaybeUninit::<libc::passwd>::uninit();
        let mut result = std::ptr::null_mut();
        // The passwd entry is copied before its backing buffer is released.
        let status = unsafe {
            libc::getpwuid_r(
                libc::getuid(),
                record.as_mut_ptr(),
                storage.as_mut_ptr().cast(),
                storage.len(),
                &mut result,
            )
        };
        if status != 0 || result.is_null() {
            return Err("cannot resolve home directory; specify --trust-store".into());
        }
        let home = unsafe { std::ffi::CStr::from_ptr((*result).pw_dir) }.to_str()?;
        Ok(Path::new(home).join(".config/flare/trust.toml"))
    }
}
fn load_store(path: &Path) -> Result<Store> {
    let bytes = match fs::read_to_string(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Store::default()),
        Err(error) => return Err(error.into()),
    };
    let store: Store = toml::from_str(&bytes)?;
    let mut seen = std::collections::HashSet::new();
    for entry in &store.trusted {
        if !fingerprint_valid(&entry.fingerprint)
            || !seen.insert((&entry.fingerprint, &entry.namespace))
            || entry
                .identity
                .as_deref()
                .is_some_and(|i| !matches!(i, "huggingface.co" | "profile"))
        {
            return Err("invalid or duplicate trust-store entry".into());
        }
        match entry.profile() {
            Some(profile) => {
                profile.validate()?;
                // The remembered key must be the one the fingerprint names.
                if hex(&Sha256::digest(profile.public_key)) != entry.fingerprint {
                    return Err("trust-store entry key does not match its fingerprint".into());
                }
            }
            None if entry.public_key.is_some() => {
                return Err("invalid trust-store entry key".into());
            }
            None => Publisher {
                development: false,
                name: "trust entry".into(),
                namespace: entry.namespace.clone(),
                public_key: [1; 32],
            }
            .validate()?,
        }
    }
    Ok(store)
}
fn modify(path: &Path, update: impl FnOnce(&mut Store)) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut lock_name = path.as_os_str().to_owned();
    lock_name.push(".lock");
    #[cfg(windows)]
    let lock = crate::windows::private_file(Path::new(&lock_name), false)?;
    #[cfg(unix)]
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(PathBuf::from(lock_name))?;
    // The lock serializes read/modify/rename across concurrent startup processes.
    #[cfg(unix)]
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    #[cfg(windows)]
    crate::windows::lock(&lock)?;
    let mut store = load_store(path)?;
    update(&mut store);
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(format!(".{}.tmp", std::process::id()));
    let temporary = PathBuf::from(temporary);
    #[cfg(windows)]
    let mut output = crate::windows::private_file(&temporary, true)?;
    #[cfg(unix)]
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary)?;
    let result = (|| -> Result<()> {
        output.write_all(toml::to_string(&store)?.as_bytes())?;
        output.sync_all()?;
        drop(output);
        #[cfg(unix)]
        {
            fs::rename(&temporary, path)?;
            File::open(parent)?.sync_all()?;
        }
        #[cfg(windows)]
        crate::windows::replace(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// How the signing key's identity was established for the disclosure.
enum Identity {
    /// The publisher's repository lists this key under the claimed namespace.
    Verified,
    /// Signatures verify against a local profile; the publisher's repository did not confirm it.
    Unverified(String),
    Development,
}

/// Verify, disclose, and accept a package, returning it ready to load along with its publisher.
/// Everything a package claim establishes before acceptance: the exact claim, the profiles that
/// can verify its signatures, the resolved publisher identity (including revocation), and the
/// remembered trust entry. Acquisition uses this to authenticate a package and bind its metadata
/// without accepting terms or decoding executables; acceptance stays in `load`/`authorize`.
pub(crate) struct ResolvedIdentity {
    pub claim: PackageClaim,
    pub profiles: Vec<Publisher>,
    pub identity: Identity,
    pub remembered: Option<Entry>,
    /// Trust store this resolution read, where acceptance is recorded.
    pub store_path: PathBuf,
}

pub(crate) fn resolve_identity(
    root: &Path,
    publisher_keys: &[PathBuf],
    hub: &crate::cli::HubOptions,
    trust_store: Option<&Path>,
) -> Result<ResolvedIdentity> {
    resolve_identity_with(root, publisher_keys, &[], hub, trust_store)
}

/// Resolve identity with profiles carried in an authenticated request. Profiles
/// are public keys, never private material or controller-local paths.
pub(crate) fn resolve_identity_with(
    root: &Path,
    publisher_keys: &[PathBuf],
    supplied: &[Publisher],
    hub: &crate::cli::HubOptions,
    trust_store: Option<&Path>,
) -> Result<ResolvedIdentity> {
    let claim = PackageClaim::read(root)?;
    let path = store_path(trust_store)?;
    let store = load_store(&path)?;
    let remembered = store
        .trusted
        .iter()
        .find(|e| e.fingerprint == claim.signer && e.namespace == claim.namespace)
        .cloned();
    let mut profiles = publishers(
        publisher_keys,
        claim.development,
        crate::environment::home_directory,
    )?;
    profiles.extend(supplied.iter().cloned());
    let mut profiles = unique_profiles(profiles)?;
    // The publisher's repository is consulted on every load that is not offline, so a key the
    // publisher removed from publisher.json stops loading everywhere, remembered or not. When it
    // cannot be consulted, a previously confirmed identity stands.
    let confirmed_earlier = remembered
        .as_ref()
        .is_some_and(|e| e.identity.as_deref() == Some("huggingface.co"));
    let identity = if claim.development {
        Identity::Development
    } else {
        match identity::resolve(&claim, hub)? {
            Resolution::Verified(profile) => {
                profiles.push(profile);
                Identity::Verified
            }
            _ if confirmed_earlier => Identity::Verified,
            Resolution::Absent => {
                Identity::Unverified(format!("no publisher.json at {}", claim.official_source))
            }
            Resolution::Unreachable(reason) => Identity::Unverified(format!(
                "{}: {reason}",
                identity::profile_url(&claim.official_source)
            )),
        }
    };
    // The remembered key verifies signatures only when no fresher profile names the same key and
    // namespace, so a publisher changing its display name does not conflict with what was
    // remembered, and a key promoted to another namespace still needs a profile for that
    // namespace.
    if !profiles
        .iter()
        .any(|p| p.fingerprint() == claim.signer && p.namespace == claim.namespace)
    {
        if let Some(profile) = remembered.as_ref().and_then(Entry::profile) {
            profiles.push(profile);
        }
    }
    Ok(ResolvedIdentity {
        claim,
        profiles,
        identity,
        remembered,
        store_path: path,
    })
}

pub fn load(
    root: &Path,
    args: &crate::cli::Run,
    progress: &mut flare_helios::startup::StartupProgress,
    origin: Option<&str>,
) -> Result<(AuthorizedPackage, Publisher)> {
    load_with(root, args, progress, origin, true)
}

/// Verify a package's signed metadata without reading payload bytes. A controller that mirrors a
/// node-held package holds only the index and companion headers.
pub fn load_metadata(
    root: &Path,
    args: &crate::cli::Run,
    progress: &mut flare_helios::startup::StartupProgress,
    origin: Option<&str>,
) -> Result<(AuthorizedPackage, Publisher)> {
    load_with(root, args, progress, origin, false)
}

fn load_with(
    root: &Path,
    args: &crate::cli::Run,
    progress: &mut flare_helios::startup::StartupProgress,
    origin: Option<&str>,
    verify_payloads: bool,
) -> Result<(AuthorizedPackage, Publisher)> {
    let phase = progress.phase("Verify package", None);
    let ResolvedIdentity {
        claim,
        profiles,
        identity,
        remembered,
        store_path: path,
    } = resolve_identity(
        root,
        &args.publisher_keys,
        &args.hub,
        args.trust_store.as_deref(),
    )?;
    // A peer of a multi-host package holds only its rank's bytes, so it verifies that rank alone.
    let rank = args.rank.filter(|&rank| rank != 0);
    // Signed topology selects the deployment. A single requested size is authoritative; without
    // one, a single-host run may infer it from the selected GPU count, while a multi-host peer
    // cannot (its local device count is not the world size) and must name it when the package
    // holds several deployments.
    let requested = match args.hub.tensor_parallel.as_slice() {
        [size] => Some(*size),
        [] => {
            if args.rank.is_none() && !args.gpus.is_empty() {
                u32::try_from(args.gpus.len()).ok()
            } else {
                None
            }
        }
        _ => None,
    };
    if args.hub.tensor_parallel.len() > 1 {
        return Err(
            "select one tensor-parallel size for a run; the package holds one deployment per size"
                .into(),
        );
    }
    let selected_rank = rank
        .map(|rank| u32::try_from(rank).map_err(|_| "rank id exceeds the package rank domain"))
        .transpose()?;
    // Only payload packages describe several deployments in one file; a legacy package is already
    // a single prepared size, so topology selection must not reroute it to the payload reader.
    let payload_format = matches!(
        flare_helios::runtime_package::NativeResources::package_format(root),
        Ok(flare_helios::runtime_package::PackageFormat::Payload)
    );
    let package = match requested {
        Some(size) if payload_format => {
            VerifiedPackage::open_tensor_parallel(root, &profiles, size, selected_rank)
        }
        _ => VerifiedPackage::open(root, &profiles, rank),
    }
    .map_err(|error| -> Box<dyn std::error::Error> {
        if error.to_string().contains("unknown package publisher") {
            if let Identity::Unverified(reason) = &identity {
                return format!(
                    "cannot verify signing key {}: {reason}; publish the key in publisher.json \
                     at the root of {} or supply its profile with --publisher-keys",
                    claim.signer, claim.official_source
                )
                .into();
            }
        }
        error.into()
    })?;
    phase.finish("signatures");
    if let Some(origin) = origin {
        let downloaded = format!("https://huggingface.co/{origin}");
        if downloaded != package.official_source() {
            progress.finish_animation();
            eprintln!(
                "\nWarning: downloaded from {downloaded}; the publisher signed {} as its source.",
                package.official_source()
            );
        }
    }
    let publisher = package.publisher().clone();
    let authorized = authorize(
        package,
        &claim,
        args,
        progress,
        &path,
        remembered,
        identity,
        verify_payloads,
    )?;
    Ok((authorized, publisher))
}

fn authorize(
    package: VerifiedPackage,
    claim: &PackageClaim,
    args: &crate::cli::Run,
    progress: &mut flare_helios::startup::StartupProgress,
    path: &Path,
    remembered: Option<Entry>,
    identity: Identity,
    verify_payloads: bool,
) -> Result<AuthorizedPackage> {
    let fingerprint = &claim.signer;
    let namespace = &claim.namespace;
    // Publisher disclosure and exact namespace acceptance protect user safety
    // before executing model package code.
    let explicit = match (&args.dangerously_approve_key, &args.dangerously_accept_publisher) {
        (Some(key), Some(owner)) if key == fingerprint && owner == namespace => Some(owner.clone()),
        (None, None) => None,
        _ => return Err("--dangerously-approve-key and --dangerously-accept-publisher must exactly match the authenticated key and namespace".into()),
    };
    if remembered.is_none() {
        progress.finish_animation();
        eprintln!(
            "\n{}",
            signing_info(&package, fingerprint, &identity, args.verify_weights)
        );
    }
    let accepted = match (&remembered, explicit) {
        (Some(entry), _) => entry.namespace.clone(),
        (None, Some(owner)) => owner,
        (None, None) => {
            if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
                return Err("unattended first use requires --dangerously-approve-key FINGERPRINT --dangerously-accept-publisher NAMESPACE".into());
            }
            match package.license() {
                Some(license) => eprint!(
                    "\nType \x1b[1m{namespace}\x1b[0m to agree to the {} terms and trust this signing key: ",
                    license.name
                ),
                None => eprint!("\nType \x1b[1m{namespace}\x1b[0m to trust this signing key: "),
            }
            io::stderr().flush()?;
            let response = acceptance_line(io::stdin().lock())?;
            if &response != namespace {
                return Err("publisher acceptance declined or did not match".into());
            }
            response
        }
    };
    let publisher = package.publisher().clone();
    let authorized = package.authorize_with_progress(
        &accepted,
        args.verify_weights,
        verify_payloads,
        Some(progress),
    )?;
    let entry = Entry {
        fingerprint: fingerprint.clone(),
        namespace: accepted,
        public_key: Some(hex(&publisher.public_key)),
        name: Some(publisher.name.clone()),
        identity: Some(
            match identity {
                Identity::Verified => "huggingface.co",
                _ => "profile",
            }
            .into(),
        ),
    };
    let current = remembered.as_ref().is_some_and(|e| {
        e.public_key == entry.public_key && e.name == entry.name && e.identity == entry.identity
    });
    if !current {
        modify(path, |store| {
            store.trusted.retain(|e| {
                !(e.fingerprint == entry.fingerprint && e.namespace == entry.namespace)
            });
            store.trusted.push(entry);
        })?;
    }
    if remembered.is_none() {
        eprintln!("Signing key accepted for {namespace}.\n");
    }
    Ok(authorized)
}

fn signing_info(
    package: &VerifiedPackage,
    fingerprint: &str,
    identity: &Identity,
    verify_weights: bool,
) -> String {
    let publisher = package.publisher();
    let mut lines = vec![
        format!("Publisher:  {} ({})", publisher.name, publisher.namespace),
        format!(
            "Identity:   {}",
            match identity {
                Identity::Verified =>
                    format!("Verified via huggingface.co/{}", publisher.namespace),
                Identity::Unverified(reason) => format!("Unverified; {reason}"),
                Identity::Development => "Local development key; verified by no publisher".into(),
            }
        ),
        "Signature:  Valid".into(),
        format!("Source:     {}", package.official_source()),
    ];
    if let Some(license) = package.license() {
        lines.push(format!("License:    {}", license.name));
        if !license.link.is_empty() {
            lines.push(format!("            {}", license.link));
        }
    }
    lines.extend([
        format!(
            "Stored data: {}",
            if verify_weights {
                "Verification enabled"
            } else {
                "Not checked (--verify-weights to check)"
            }
        ),
        String::new(),
        "Signing key:".into(),
        fingerprint.to_owned(),
        String::new(),
        "Executable GPU code can access your process and data.".into(),
    ]);
    if matches!(identity, Identity::Unverified(_)) {
        lines.push("A valid signature does not verify the claimed name or source.".into());
    }
    if publisher.development {
        lines.push("Local development package; not an official published checkpoint.".into());
    } else {
        lines.push("Do not proceed if this package came from another source.".into());
    }
    if package.license().is_some() {
        lines.push(
            "Typing the publisher namespace agrees to the license terms at the link above.".into(),
        );
    }
    lines.push("Only approve a signing key you independently trust.".into());
    if !io::stderr().is_terminal() {
        return format!("Signing key acceptance\n{}", lines.join("\n"));
    }
    #[cfg(windows)]
    let columns = crate::windows::terminal_columns();
    #[cfg(unix)]
    let columns = {
        let mut size: libc::winsize = unsafe { std::mem::zeroed() };
        // SAFETY: ioctl writes terminal dimensions into a valid winsize.
        unsafe { libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size) };
        if size.ws_col == 0 {
            80
        } else {
            usize::from(size.ws_col)
        }
    };
    if columns < 28 {
        return format!("Signing key acceptance\n{}", lines.join("\n"));
    }
    let width = columns.min(80) - 4;
    // Only a publisher-confirmed identity earns the calm frame; development and unknown keys warn.
    let color = if matches!(identity, Identity::Verified) {
        "36"
    } else {
        "33"
    };
    let border = format!("\x1b[{color}m");
    let title = " Signing key acceptance ";
    let mut output = format!(
        "{border}╭─{title}{}╮\x1b[0m\n",
        "─".repeat(width + 1 - title.len())
    );
    for line in lines {
        let mut remaining = line.as_str();
        while remaining.chars().count() > width {
            let end = remaining.char_indices().nth(width).unwrap().0;
            let split = remaining[..end]
                .rfind(' ')
                .filter(|n| *n > width / 2)
                .unwrap_or(end);
            let row = remaining[..split].trim_end();
            output.push_str(&format!(
                "{border}│\x1b[0m {row}{} {border}│\x1b[0m\n",
                " ".repeat(width - row.chars().count())
            ));
            remaining = remaining[split..].trim_start();
        }
        output.push_str(&format!(
            "{border}│\x1b[0m {remaining}{} {border}│\x1b[0m\n",
            " ".repeat(width - remaining.chars().count())
        ));
    }
    output.push_str(&format!("{border}╰{}╯\x1b[0m", "─".repeat(width + 2)));
    output
}

fn acceptance_line(input: impl BufRead) -> io::Result<String> {
    let mut response = String::new();
    input.take(256).read_line(&mut response)?;
    if !response.ends_with('\n') {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "publisher acceptance requires a complete line",
        ));
    }
    response.pop();
    if response.ends_with('\r') {
        response.pop();
    }
    Ok(response)
}

pub fn revoke(fingerprint: &str, given: Option<&Path>) -> Result<()> {
    if !fingerprint_valid(fingerprint) {
        return Err("a full lowercase SHA-256 fingerprint is required".into());
    }
    modify(&store_path(given)?, |store| {
        store
            .trusted
            .retain(|entry| entry.fingerprint != fingerprint)
    })?;
    eprintln!("Removed trust for signer {fingerprint}.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn development_publisher() -> Publisher {
        Publisher {
            name: "Local development signer".into(),
            namespace: "local-development".into(),
            public_key: [7; 32],
            development: true,
        }
    }

    #[test]
    fn development_packages_find_the_default_public_profile() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join(".config/flare/development-publisher.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let publisher = development_publisher();
        fs::write(&path, serde_json::to_vec(&[&publisher]).unwrap()).unwrap();
        assert_eq!(
            publishers(&[], true, || Ok(home.path().to_owned())).unwrap(),
            vec![publisher]
        );
    }

    #[test]
    fn one_key_may_hold_several_namespaces_but_not_conflicting_identities() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("publisher.json");
        let mut second = development_publisher();
        second.name = "Example".into();
        second.namespace = "example".into();
        second.development = false;
        let mut promoted = second.clone();
        promoted.namespace = "example-next".into();
        fs::write(&path, serde_json::to_vec(&[&second, &promoted]).unwrap()).unwrap();
        assert_eq!(
            publishers(std::slice::from_ref(&path), false, || unreachable!()).unwrap(),
            vec![second.clone(), promoted]
        );
        // The same key and namespace with different attributes is one ambiguity, not two profiles.
        let mut renamed = second.clone();
        renamed.name = "Someone Else".into();
        fs::write(&path, serde_json::to_vec(&[&second, &renamed]).unwrap()).unwrap();
        assert!(publishers(std::slice::from_ref(&path), false, || unreachable!()).is_err());
    }

    #[test]
    fn explicit_profiles_and_production_packages_skip_the_default_lookup() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("publisher.json");
        let publisher = development_publisher();
        fs::write(&path, serde_json::to_vec(&[&publisher]).unwrap()).unwrap();
        for development in [false, true] {
            assert_eq!(
                publishers(std::slice::from_ref(&path), development, || {
                    panic!("explicit profiles must not resolve the default directory")
                })
                .unwrap(),
                vec![publisher.clone()]
            );
        }
        assert!(
            publishers(&[], false, || {
                panic!("production packages must not resolve the default directory")
            })
            .unwrap()
            .is_empty()
        );
        fs::write(&path, b"[]").unwrap();
        assert!(
            publishers(std::slice::from_ref(&path), true, || {
                panic!("an empty explicit profile must not trigger the default lookup")
            })
            .unwrap()
            .is_empty()
        );
        fs::remove_file(&path).unwrap();
        assert!(
            publishers(&[path], true, || {
                panic!("a missing explicit profile must not trigger the default lookup")
            })
            .is_err()
        );
    }

    #[test]
    fn a_missing_development_profile_leaves_other_keys_available() {
        let home = tempfile::tempdir().unwrap();
        assert!(
            publishers(&[], true, || Ok(home.path().to_owned()))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn malformed_and_oversized_defaults_report_the_profile_path() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join(".config/flare/development-publisher.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        for (bytes, message) in [
            (b"not json".to_vec(), "invalid publisher profile"),
            (vec![b' '; SLOT_SIZE + 1], "exceeds 64 KiB"),
        ] {
            fs::write(&path, bytes).unwrap();
            let error = publishers(&[], true, || Ok(home.path().to_owned()))
                .unwrap_err()
                .to_string();
            assert!(error.contains(message), "{error}");
            assert!(error.contains(path.to_str().unwrap()), "{error}");
        }
    }

    #[test]
    fn default_profiles_validate_keys_and_reject_conflicting_identities() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join(".config/flare/development-publisher.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let publisher = development_publisher();
        let mut invalid = publisher.clone();
        invalid.public_key = [0; 32];
        fs::write(&path, serde_json::to_vec(&[invalid]).unwrap()).unwrap();
        assert!(publishers(&[], true, || Ok(home.path().to_owned())).is_err());
        let mut conflicting = publisher.clone();
        conflicting.name = "Another signer".into();
        fs::write(
            &path,
            serde_json::to_vec(&[publisher, conflicting]).unwrap(),
        )
        .unwrap();
        let error = publishers(&[], true, || Ok(home.path().to_owned()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicting identities"), "{error}");
    }

    #[test]
    fn acceptance_is_exact_and_requires_a_complete_bounded_line() {
        assert_eq!(
            acceptance_line(&b"example-publisher\n"[..]).unwrap(),
            "example-publisher"
        );
        assert_eq!(
            acceptance_line(&b"example-publisher\r\n"[..]).unwrap(),
            "example-publisher"
        );
        assert_eq!(
            acceptance_line(&b" example-publisher \n"[..]).unwrap(),
            " example-publisher "
        );
        assert!(acceptance_line(&b"example-publisher"[..]).is_err());
        assert!(acceptance_line([vec![b'x'; 256], vec![b'\n']].concat().as_slice()).is_err());
        assert!(acceptance_line(&b""[..]).is_err());
    }
}

#[cfg(all(test, windows))]
mod windows_tests {
    use super::*;

    #[test]
    fn concurrent_updates_preserve_entries_and_replace_existing_files() {
        let directory =
            std::env::temp_dir().join(format!("flare-trust-用户-{}", std::process::id()));
        let path = directory.join("trust.toml");
        std::thread::scope(|scope| {
            for index in 0..8 {
                let path = &path;
                scope.spawn(move || {
                    modify(path, |store| {
                        store.trusted.push(Entry {
                            fingerprint: format!("{index:064x}"),
                            namespace: "test-publisher".into(),
                            public_key: None,
                            name: None,
                            identity: None,
                        })
                    })
                    .unwrap()
                });
            }
        });
        assert_eq!(load_store(&path).unwrap().trusted.len(), 8);
        modify(&path, |store| store.trusted.clear()).unwrap();
        assert!(load_store(&path).unwrap().trusted.is_empty());
        fs::remove_dir_all(directory).unwrap();
    }
}
