//! Shared-memory regions and restart recovery for the manager.

use crate::{
    Result,
    arena::{Arena, Extent},
    invalid,
};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
    sync::Arc,
};

pub const SHM_ROOT: &str = "/dev/shm/selene";
pub const PAGE_BYTES: u64 = 4096;
use flare_selene::shared::region::{ARENA_FILE, Region, STAGING_FILE};

/// One manager instance's shared-memory regions and the windows granted from them.
pub struct Local {
    instance: String,
    directory: PathBuf,
    _lock: File,
    arena: Arena,
    staging: Arena,
    arena_path: PathBuf,
    staging_path: PathBuf,
    arena_size: u64,
    staging_size: u64,
    region_files: [(u64, u64); 2],
    holders: HashMap<u64, u32>,
    /// Identifies this set of region files. It changes exactly when the files are recreated, so
    /// a process that registered the old files can tell the new ones are different.
    regions_generation: u64,
    /// Processes of the previous run that may still access the reused regions.
    prior: Option<Prior>,
    /// True while `prior` holds the regions; the pool reads it to refuse claims distinctly.
    held: Arc<std::sync::atomic::AtomicBool>,
}

/// The reused regions' previous run: nothing in them is granted until every process that held
/// a session then has reconnected (so it has drained) or exited. See PROTOCOL.md section 8.2.
struct Prior {
    /// Pid to a descriptor that reports its exit; `None` when the process cannot be watched,
    /// which holds until it reconnects.
    holders: HashMap<u32, Option<OwnedFd>>,
    /// The whole of both regions, so no grant or store allocation can land in them.
    _extents: Vec<Extent>,
}

/// Fixed file names let a restart find the regions it left.
const GENERATION_FILE: &str = "generation";
const HOLDERS_FILE: &str = "holders";
const LOCK_FILE: &str = "lock";

/// Names the startup step and path an error came from.
fn step_error(step: &str, path: &Path, error: io::Error) -> io::Error {
    io::Error::new(error.kind(), format!("{step} {}: {error}", path.display()))
}

/// Keep one manager per pool name on this host, independently of its region namespace.
pub fn lock_pool(name: &str, socket: &Path) -> Result<File> {
    use std::os::unix::fs::MetadataExt;

    validate_instance(name)?;
    let root = Path::new(SHM_ROOT);
    create_private_directory(root).map_err(|error| step_error("create", root, error))?;
    let path = root.join(format!("{name}.lock"));
    check_owned_file(&path)?;
    let mut lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .mode(0o660)
        .open(&path)
        .map_err(|error| step_error("open", &path, error))?;
    let metadata = lock.metadata()?;
    if !metadata.is_file() || metadata.uid() != unsafe { libc::geteuid() } {
        return Err(invalid(
            "pool name lock is not a regular file owned by this manager",
        ));
    }
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let error = io::Error::last_os_error();
        let mut holder = String::new();
        lock.read_to_string(&mut holder)?;
        return Err(io::Error::new(
            error.kind(),
            format!(
                "pool '{name}' is already in use by {}: {error}",
                holder.trim()
            ),
        ));
    }
    lock.set_len(0)?;
    writeln!(lock, "pid {} at {}", std::process::id(), socket.display())?;
    Ok(lock)
}

impl Local {
    /// Owns `/dev/shm/selene/<instance>`: takes the exclusive lock, reclaims the debris a
    /// crashed owner left behind and reuses its arena and staging files when they have the
    /// configured sizes (same files, inodes and offsets), else creates fresh, preallocated
    /// ones under a new generation.
    pub fn open(instance: &str, arena_bytes: usize, staging_bytes: usize) -> Result<Self> {
        validate_instance(instance)?;
        let root = Path::new(SHM_ROOT);
        create_private_directory(root).map_err(|error| step_error("create", root, error))?;
        let directory = root.join(instance);
        create_private_directory(&directory)
            .map_err(|error| step_error("create", &directory, error))?;

        use std::os::unix::fs::MetadataExt;
        let metadata = fs::symlink_metadata(&directory)?;
        if !metadata.is_dir() || metadata.uid() != unsafe { libc::geteuid() } {
            return Err(invalid(
                "shared region directory is not owned by this manager",
            ));
        }

        let lock_path = directory.join(LOCK_FILE);
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .mode(0o660)
            .open(&lock_path)
            .map_err(|error| step_error("open", &lock_path, error))?;
        if lock.metadata()?.uid() != unsafe { libc::geteuid() } {
            return Err(invalid("shared region lock belongs to another user"));
        }
        if unsafe { libc::fchmod(lock.as_raw_fd(), 0o660) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // The open flags already include O_CLOEXEC; set the descriptor flag as well so no child
        // can keep the instance lock alive across an exec.
        if unsafe { libc::fcntl(lock.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } == -1 {
            return Err(io::Error::last_os_error());
        }
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let error = io::Error::last_os_error();
            return Err(io::Error::new(
                error.kind(),
                format!("instance '{instance}' is already in use: {error}"),
            ));
        }

        for name in [
            LOCK_FILE,
            ARENA_FILE,
            STAGING_FILE,
            GENERATION_FILE,
            HOLDERS_FILE,
        ] {
            check_owned_file(&directory.join(name))?;
        }
        let reclaimed = reclaim_stale(&directory)
            .map_err(|error| step_error("reclaim in", &directory, error))?;
        if reclaimed > 0 {
            eprintln!("selene: reclaimed stale instance '{instance}' ({reclaimed} files)");
        }

        let arena_size = region_size_of(ARENA_FILE, arena_bytes)?;
        let staging_size = region_size_of(STAGING_FILE, staging_bytes)?;
        let arena_path = directory.join(ARENA_FILE);
        let staging_path = directory.join(STAGING_FILE);
        let reused = match reusable(&directory, arena_size, staging_size) {
            Ok(generation) => Some(generation),
            Err(reason) => {
                let existed = arena_path.exists() || staging_path.exists();
                for name in [ARENA_FILE, STAGING_FILE, GENERATION_FILE, HOLDERS_FILE] {
                    let _ = fs::remove_file(directory.join(name));
                }
                if existed {
                    eprintln!(
                        "selene: instance '{instance}' recreated its shared regions ({reason}); \
                         running instances must restart to use the pool"
                    );
                }
                None
            }
        };
        let regions_generation = match reused {
            Some(generation) => generation,
            None => {
                create_region(&directory, ARENA_FILE, arena_size)
                    .map_err(|error| step_error("create the arena in", &directory, error))?;
                create_region(&directory, STAGING_FILE, staging_size)
                    .map_err(|error| step_error("create staging in", &directory, error))?;
                let generation = new_generation()?;
                write_atomically(
                    &directory.join(GENERATION_FILE),
                    format!("{generation:016x}\n").as_bytes(),
                )
                .map_err(|error| step_error("write the generation in", &directory, error))?;
                generation
            }
        };
        let arena = Arena::open(&arena_path, arena_size)?;
        let staging = Arena::open(&staging_path, staging_size)?;
        if reused.is_some() {
            eprintln!(
                "selene: instance '{instance}' reused its shared regions (generation \
                 {regions_generation:016x})"
            );
        }
        let prior = if reused.is_some() {
            Prior::read(&directory, &arena, &staging, instance)
        } else {
            None
        };

        let region_files = [&arena_path, &staging_path].map(|path| {
            fs::symlink_metadata(path).map(|metadata| (metadata.dev(), metadata.ino()))
        });
        let [arena_file, staging_file] = region_files;
        let local = Self {
            instance: instance.to_owned(),
            directory,
            _lock: lock,
            arena,
            staging,
            arena_path,
            staging_path,
            arena_size: arena_size as u64,
            staging_size: staging_size as u64,
            region_files: [arena_file?, staging_file?],
            holders: HashMap::new(),
            regions_generation,
            held: Arc::new(std::sync::atomic::AtomicBool::new(prior.is_some())),
            prior,
        };
        local.persist_holders();
        Ok(local)
    }

    /// Identifies the region files. Equal values mean the same files at the same offsets; a
    /// process that mapped and registered them can keep using them.
    pub fn regions_generation(&self) -> u64 {
        self.regions_generation
    }

    /// The flag that is true while the regions are held for the previous run.
    pub fn held_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.held.clone()
    }

    /// Whether the regions are still held for processes of the previous run.
    pub fn held_for_previous_run(&self) -> bool {
        self.prior.is_some()
    }

    /// Writes the pids with a live local session, and the previous run's still-unresolved
    /// holders, so a restart after a crash knows whose copies could still be in flight.
    fn persist_holders(&self) {
        let mut pids: Vec<u32> = self.holders.values().copied().collect();
        if let Some(prior) = &self.prior {
            pids.extend(prior.holders.keys());
        }
        pids.sort_unstable();
        pids.dedup();
        let mut text = String::new();
        for pid in pids {
            if let Some(start) = process_start(pid) {
                text.push_str(&format!("{pid} {start}\n"));
            }
        }
        let _ = write_atomically(&self.directory.join(HOLDERS_FILE), text.as_bytes());
    }

    /// Releases the previous run's hold once none of its processes can still be copying.
    fn settle_prior(&mut self) {
        if self
            .prior
            .as_ref()
            .is_some_and(|prior| prior.holders.is_empty())
        {
            self.prior = None;
            self.held.store(false, std::sync::atomic::Ordering::Release);
            eprintln!(
                "selene: instance '{}' released the shared regions held for the previous run",
                self.instance
            );
        }
    }

    pub(crate) fn check_regions(&self) -> Result<()> {
        for ((path, size), identity) in [
            (&self.arena_path, self.arena_size),
            (&self.staging_path, self.staging_size),
        ]
        .into_iter()
        .zip(self.region_files)
        {
            let valid = fs::symlink_metadata(path).is_ok_and(|metadata| {
                metadata.is_file()
                    && metadata.len() == size
                    && (metadata.dev(), metadata.ino()) == identity
            });
            if !valid {
                return Err(invalid(format!(
                    "shared-memory region {} vanished or was replaced; systemd-logind RemoveIPC \
                     may have removed it; stopping the manager; use a system user/service or \
                     set RemoveIPC=no before restarting the manager and instances",
                    path.display()
                )));
            }
        }
        Ok(())
    }

    pub fn instance(&self) -> &str {
        &self.instance
    }
    pub fn directory(&self) -> &Path {
        &self.directory
    }
    pub fn arena_path(&self) -> &Path {
        &self.arena_path
    }
    pub fn staging_path(&self) -> &Path {
        &self.staging_path
    }
    pub fn arena_size(&self) -> u64 {
        self.arena_size
    }
    pub fn staging_size(&self) -> u64 {
        self.staging_size
    }
    pub fn arena(&self) -> Arena {
        self.arena.clone()
    }
    pub fn staging(&self) -> Arena {
        self.staging.clone()
    }
    fn region_size(&self, region: Region) -> u64 {
        match region {
            Region::Arena => self.arena_size,
            Region::Staging => self.staging_size,
        }
    }
    pub fn arena_used(&self) -> usize {
        self.arena.used()
    }
    pub fn staging_used(&self) -> usize {
        self.staging.used()
    }

    /// Records a live pool process for restart recovery; the pool owns its pidfd.
    pub fn accept(&mut self, connection: u64, pid: u32) -> Result<()> {
        if pid == 0 {
            return Ok(());
        }
        self.holders.insert(connection, pid);
        // A process of the previous run that reconnects has drained what it did under the
        // dead manager before it did (PROTOCOL.md section 8.1), so its hold is over.
        if let Some(prior) = &mut self.prior {
            if prior.holders.remove(&pid).is_some() {
                eprintln!(
                    "selene: instance '{}': process {pid} of the previous run reconnected, {} left",
                    self.instance,
                    prior.holders.len()
                );
            }
            self.settle_prior();
        }
        self.persist_holders();
        Ok(())
    }

    /// The pool has proved this process no longer accesses the regions.
    pub fn close(&mut self, connection: u64) {
        self.holders.remove(&connection);
        self.persist_holders();
    }

    pub fn poll_prior(&mut self) {
        if let Some(prior) = &mut self.prior {
            let before = prior.holders.len();
            let instance = &self.instance;
            prior.holders.retain(|pid, pidfd| {
                let gone = pidfd.as_ref().is_some_and(pidfd_exited);
                if gone {
                    eprintln!(
                        "selene: instance '{instance}': process {pid} of the previous run exited"
                    );
                }
                !gone
            });
            if prior.holders.len() != before {
                self.settle_prior();
                self.persist_holders();
            }
        }
    }

    /// Checks the view before exposing a stored buffer to the pool.
    pub fn offset(
        &self,
        region: Region,
        extent: &Extent,
        within: usize,
        length: u64,
    ) -> Result<u64> {
        let length = usize::try_from(length).map_err(|_| invalid("window length overflows"))?;
        if !within.is_multiple_of(PAGE_BYTES as usize)
            || !length.is_multiple_of(PAGE_BYTES as usize)
            || within
                .checked_add(length)
                .is_none_or(|end| end > extent.bytes().len())
        {
            return Err(invalid("window exceeds its extent or is not page aligned"));
        }
        let offset = extent
            .offset()
            .checked_add(within)
            .ok_or_else(|| invalid("window offset overflows"))? as u64;
        if offset
            .checked_add(length as u64)
            .is_none_or(|end| end > self.region_size(region))
        {
            return Err(invalid("window exceeds its region"));
        }
        Ok(offset)
    }
}

/// Creates `directory` with mode 0770 when it is missing. An existing directory
/// keeps its mode and ownership: the manager only sets a mode on a directory it
/// created. A directory this process cannot write is reported with its path and
/// owner instead of a bare `EPERM`, which usually only says the caller is not
/// the owner.
fn create_private_directory(directory: &Path) -> Result<()> {
    match fs::create_dir(directory) {
        Ok(()) => return fs::set_permissions(directory, fs::Permissions::from_mode(0o770)),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    if directory_writable(directory) {
        return Ok(());
    }
    let owner = fs::metadata(directory)
        .map(|metadata| {
            use std::os::unix::fs::MetadataExt;
            format!(" (owned by uid {} gid {})", metadata.uid(), metadata.gid())
        })
        .unwrap_or_default();
    Err(invalid(format!(
        "{}: existing directory is not writable by this process{owner}",
        directory.display()
    )))
}

/// Whether this process may create files in `directory`, judged by its
/// effective uid and gid.
fn directory_writable(directory: &Path) -> bool {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let Ok(path) = CString::new(directory.as_os_str().as_bytes()) else {
        return false;
    };
    unsafe { libc::faccessat(libc::AT_FDCWD, path.as_ptr(), libc::W_OK, libc::AT_EACCESS) == 0 }
}

fn validate_instance(instance: &str) -> Result<()> {
    if instance.is_empty()
        || instance == "."
        || instance == ".."
        || instance.contains('/')
        || instance.contains('\0')
    {
        return Err(invalid(format!("invalid instance name '{instance}'")));
    }
    Ok(())
}

/// Refuse foreign files before startup can replace or remove them.
fn check_owned_file(path: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && metadata.uid() == unsafe { libc::geteuid() } => {
            Ok(())
        }
        Ok(_) => Err(invalid(format!(
            "{} is not a regular file owned by this manager",
            path.display()
        ))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// With the instance lock held, remove only this manager's unfinished metadata writes.
fn reclaim_stale(directory: &Path) -> Result<usize> {
    let mut removed = 0;
    for name in ["generation.tmp", "holders.tmp"] {
        let path = directory.join(name);
        check_owned_file(&path)?;
        match fs::remove_file(path) {
            Ok(()) => removed += 1,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(removed)
}

/// A region's size: the configured bytes rounded down to a page.
fn region_size_of(name: &str, bytes: usize) -> Result<usize> {
    let size = bytes / PAGE_BYTES as usize * PAGE_BYTES as usize;
    if size == 0 {
        return Err(invalid(format!(
            "{name} region must hold at least {PAGE_BYTES} bytes"
        )));
    }
    Ok(size)
}

/// Creates a fresh region file, sets 0660 and `fallocate`s it to its full size.
fn create_region(directory: &Path, name: &str, size: usize) -> Result<PathBuf> {
    let path = directory.join(name);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .mode(0o660)
        .open(&path)?;
    if unsafe { libc::fchmod(file.as_raw_fd(), 0o660) } != 0 {
        return Err(io::Error::last_os_error());
    }
    while unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, size as libc::off_t) } != 0 {
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        let _ = fs::remove_file(&path);
        return Err(error);
    }
    Ok(path)
}

/// The generation of the region files a previous run left, if they can be used as they are:
/// regular files of this user, exactly the configured sizes, with their memory reserved, and a
/// readable generation. Otherwise why not.
fn reusable(
    directory: &Path,
    arena_size: usize,
    staging_size: usize,
) -> std::result::Result<u64, String> {
    use std::os::unix::fs::MetadataExt;
    for (name, size) in [(ARENA_FILE, arena_size), (STAGING_FILE, staging_size)] {
        let path = directory.join(name);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(format!("no {name} file"));
            }
            Err(error) => return Err(format!("{name}: {error}")),
        };
        if !metadata.is_file() {
            return Err(format!("{name} is not a regular file"));
        }
        if metadata.uid() != unsafe { libc::geteuid() } {
            return Err(format!("{name} belongs to another user"));
        }
        if metadata.len() != size as u64 {
            return Err(format!(
                "{name} is {} bytes and {size} are configured",
                metadata.len()
            ));
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&path)
            .map_err(|error| format!("{name}: {error}"))?;
        if unsafe { libc::fchmod(file.as_raw_fd(), 0o660) } != 0 {
            return Err(format!("{name}: {}", io::Error::last_os_error()));
        }
        if unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, size as libc::off_t) } != 0 {
            return Err(format!("{name}: {}", io::Error::last_os_error()));
        }
    }
    let text = fs::read_to_string(directory.join(GENERATION_FILE))
        .map_err(|error| format!("generation: {error}"))?;
    match u64::from_str_radix(text.trim(), 16) {
        Ok(generation) if generation != 0 => Ok(generation),
        _ => Err("the generation file is unreadable".to_owned()),
    }
}

/// A random nonzero 64-bit identity for a new set of region files.
fn new_generation() -> Result<u64> {
    loop {
        let mut bytes = [0u8; 8];
        File::open("/dev/urandom")?.read_exact(&mut bytes)?;
        let value = u64::from_le_bytes(bytes);
        if value != 0 {
            return Ok(value);
        }
    }
}

/// Replaces `path` with `contents` in one step, so a crash leaves the old file or the new one.
fn write_atomically(path: &Path, contents: &[u8]) -> io::Result<()> {
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, contents)?;
    fs::set_permissions(&temporary, fs::Permissions::from_mode(0o660))?;
    fs::rename(&temporary, path)
}

/// When a process started, in clock ticks since boot: with its pid it names one process, so a
/// recycled pid is not mistaken for the one that held a session.
fn process_start(pid: u32) -> Option<u64> {
    let text = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The name may contain spaces and parentheses; everything after the last ')' is plain.
    let rest = &text[text.rfind(')')? + 1..];
    rest.split_whitespace().nth(19)?.parse().ok()
}

impl Prior {
    /// The processes the last run recorded that are still the same live processes. `None` when
    /// there are none, so nothing needs holding.
    fn read(directory: &Path, arena: &Arena, staging: &Arena, instance: &str) -> Option<Self> {
        let text = fs::read_to_string(directory.join(HOLDERS_FILE)).ok()?;
        let mut holders = HashMap::new();
        for line in text.lines() {
            let mut fields = line.split_whitespace();
            let (Some(pid), Some(start)) = (
                fields.next().and_then(|field| field.parse::<u32>().ok()),
                fields.next().and_then(|field| field.parse::<u64>().ok()),
            ) else {
                continue;
            };
            if process_start(pid) != Some(start) {
                continue;
            }
            let (pidfd, exited) = watch_pid(pid);
            if !exited {
                holders.insert(pid, pidfd);
            }
        }
        if holders.is_empty() {
            return None;
        }
        let mut pids: Vec<_> = holders.keys().copied().collect();
        pids.sort_unstable();
        eprintln!(
            "selene: instance '{instance}' holds its shared regions until {} process(es) of the \
             previous run reconnect or exit (pids {pids:?})",
            holders.len()
        );
        let mut extents = arena.reserve_all();
        extents.extend(staging.reserve_all());
        Some(Self {
            holders,
            _extents: extents,
        })
    }
}

/// A missing descriptor is not proof of exit unless pidfd_open returned ESRCH.
pub(crate) fn watch_pid(pid: u32) -> (Option<OwnedFd>, bool) {
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0u32) } as i32;
    if fd >= 0 {
        return (Some(unsafe { OwnedFd::from_raw_fd(fd) }), false);
    }
    (
        None,
        io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH),
    )
}

pub(crate) fn pidfd_exited(pidfd: &OwnedFd) -> bool {
    let mut poll = libc::pollfd {
        fd: pidfd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let result = unsafe { libc::poll(&mut poll, 1, 0) };
    result > 0 && poll.revents != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        os::unix::fs::PermissionsExt,
        process::{Child, Command},
        sync::atomic::{AtomicU64, Ordering},
        thread,
        time::{Duration, Instant},
    };

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn instance(tag: &str) -> String {
        format!(
            "selene-local-{}-{tag}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )
    }

    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn local(name: &str, arena: usize, staging: usize) -> (Local, Cleanup) {
        let local = Local::open(name, arena, staging).unwrap();
        let cleanup = Cleanup(local.directory().to_path_buf());
        (local, cleanup)
    }

    /// Reopening an instance right after its owner was dropped can transiently observe the
    /// lock still held by a child this process spawned for another test: the child inherits
    /// the lock descriptor and keeps the flock until its exec closes it (CLOEXEC). Retry that
    /// window instead of assuming close and release are the same instant.
    fn reopen(name: &str) -> Local {
        reopen_sized(name, 64 * 1024, 64 * 1024)
    }

    fn reopen_sized(name: &str, arena: usize, staging: usize) -> Local {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match Local::open(name, arena, staging) {
                Ok(local) => return local,
                Err(error)
                    if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
                {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("reopening instance '{name}' failed: {error}"),
            }
        }
    }

    struct ChildGuard(Child);
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    fn spawn_sleeper() -> ChildGuard {
        ChildGuard(
            Command::new("/bin/sh")
                .args(["-c", "sleep 30"])
                .spawn()
                .unwrap(),
        )
    }

    #[test]
    fn the_same_instance_cannot_be_opened_twice() {
        let name = instance("lock");
        let (first, _cleanup) = local(&name, 64 * 1024, 64 * 1024);
        let error = Local::open(&name, 64 * 1024, 64 * 1024).err().unwrap();
        assert!(
            error.to_string().contains("already in use"),
            "unexpected error: {error}"
        );
        drop(first);
        let second = reopen(&name);
        drop(second);
    }

    #[test]
    fn vanished_or_replaced_regions_are_reported() {
        for name in [ARENA_FILE, STAGING_FILE] {
            let instance = instance("removed-region");
            let (local, _cleanup) = local(&instance, 64 * 1024, 64 * 1024);
            local.check_regions().unwrap();
            let path = local.directory().join(name);
            fs::remove_file(&path).unwrap();
            assert!(
                local
                    .check_regions()
                    .unwrap_err()
                    .to_string()
                    .contains("RemoveIPC")
            );
            create_region(local.directory(), name, 64 * 1024).unwrap();
            assert!(
                local.check_regions().is_err(),
                "a replacement is not the mapped file"
            );
        }
    }

    #[test]
    fn stale_region_debris_is_removed() {
        let name = instance("stale");
        let directory;
        {
            let first = Local::open(&name, 64 * 1024, 64 * 1024).unwrap();
            directory = first.directory().to_path_buf();
            assert!(first.arena_path().exists());
            assert!(first.staging_path().exists());
        }
        fs::write(directory.join("generation.tmp"), b"stale").unwrap();
        fs::write(directory.join("holders.tmp"), b"stale").unwrap();
        fs::write(directory.join("leftover.tmp"), b"stale").unwrap();
        let second = reopen(&name);
        let _cleanup = Cleanup(directory.clone());
        assert!(!directory.join("generation.tmp").exists());
        assert!(!directory.join("holders.tmp").exists());
        assert!(directory.join("leftover.tmp").exists());
        assert!(directory.join("lock").exists());
        assert!(second.arena_path().exists());
        assert_eq!(
            fs::metadata(second.arena_path()).unwrap().len(),
            second.arena_size()
        );
        assert_eq!(
            fs::metadata(second.staging_path()).unwrap().len(),
            second.staging_size()
        );
        let mode = fs::metadata(second.arena_path())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o660);
    }

    #[test]
    fn an_existing_instance_directory_keeps_its_mode() {
        let name = instance("mode");
        let directory = Path::new(SHM_ROOT).join(&name);
        fs::create_dir_all(&directory).unwrap();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o755)).unwrap();
        let local = Local::open(&name, 64 * 1024, 64 * 1024).unwrap();
        let _cleanup = Cleanup(directory.clone());
        let mode = fs::metadata(local.directory())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o755, "an existing directory keeps its mode");
    }

    #[test]
    fn an_unwritable_instance_directory_names_itself() {
        if unsafe { libc::geteuid() } == 0 {
            // Root may write any directory, so this failure cannot be staged.
            return;
        }
        let name = instance("unwritable");
        let directory = Path::new(SHM_ROOT).join(&name);
        fs::create_dir_all(&directory).unwrap();
        let _cleanup = Cleanup(directory.clone());
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o500)).unwrap();
        let error = Local::open(&name, 64 * 1024, 64 * 1024)
            .err()
            .unwrap()
            .to_string();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        assert!(
            error.contains(&directory.display().to_string()) && error.contains("uid"),
            "{error}"
        );
    }

    fn inode(path: &Path) -> u64 {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(path).unwrap().ino()
    }

    #[test]
    fn a_restart_reuses_the_same_files_and_offsets() {
        let name = instance("reuse");
        let (first, cleanup) = local(&name, 64 * 1024, 64 * 1024);
        let mut first = first;
        first.accept(1, std::process::id()).unwrap();
        let grant = first.staging().allocate(4096).unwrap();
        let offset = grant.offset();
        drop(grant);
        let (arena, staging) = (
            first.arena_path().to_path_buf(),
            first.staging_path().to_path_buf(),
        );
        let (generation, inodes) = (first.regions_generation(), (inode(&arena), inode(&staging)));
        assert_ne!(generation, 0);
        // The window is still granted when the manager goes away.
        drop(first);

        let mut second = reopen(&name);
        assert_eq!(second.arena_path(), arena);
        assert_eq!(second.staging_path(), staging);
        assert_eq!(
            (inode(&arena), inode(&staging)),
            inodes,
            "the files were reused"
        );
        assert_eq!(second.regions_generation(), generation);
        // The test process is the holder that was recorded, so it is held until it reconnects.
        assert!(second.held_for_previous_run());
        second.accept(1, std::process::id()).unwrap();
        assert!(!second.held_for_previous_run());
        let again = second.staging().allocate(4096).unwrap();
        assert_eq!(again.offset(), offset, "regions keep the same offsets");
        drop(cleanup);
    }

    #[test]
    fn a_changed_size_recreates_the_files_under_a_new_generation() {
        let name = instance("resize");
        let (first, cleanup) = local(&name, 64 * 1024, 64 * 1024);
        let generation = first.regions_generation();
        let directory = first.directory().to_path_buf();
        drop(first);

        let second = reopen_sized(&name, 64 * 1024, 128 * 1024);
        let resized = second.regions_generation();
        assert_ne!(resized, generation);
        assert_eq!(second.staging_size(), 128 * 1024);
        assert_eq!(
            fs::metadata(second.staging_path()).unwrap().len(),
            128 * 1024
        );
        assert_eq!(
            fs::read_to_string(directory.join("generation"))
                .unwrap()
                .trim(),
            format!("{:016x}", second.regions_generation())
        );
        assert!(
            !second.held_for_previous_run(),
            "new files have no previous holders"
        );
        drop(second);
        // The same size again is a plain reuse.
        let third = reopen_sized(&name, 64 * 1024, 128 * 1024);
        assert_eq!(third.regions_generation(), resized);
        drop(cleanup);
    }

    #[test]
    fn an_unusable_file_is_recreated_under_a_new_generation() {
        let name = instance("unusable");
        let (first, cleanup) = local(&name, 64 * 1024, 64 * 1024);
        let generation = first.regions_generation();
        let staging = first.staging_path().to_path_buf();
        drop(first);
        // Truncated behind the manager's back.
        fs::OpenOptions::new()
            .write(true)
            .open(&staging)
            .unwrap()
            .set_len(4096)
            .unwrap();
        let second = reopen(&name);
        assert_ne!(second.regions_generation(), generation);
        assert_eq!(fs::metadata(&staging).unwrap().len(), 64 * 1024);
        drop(second);

        // A generation nobody can read is as unusable as a missing file.
        let directory = cleanup.0.clone();
        let second_generation = fs::read_to_string(directory.join("generation")).unwrap();
        fs::write(directory.join("generation"), b"not a number").unwrap();
        let third = reopen(&name);
        assert_ne!(
            format!("{:016x}\n", third.regions_generation()),
            second_generation
        );
        drop(third);
        fs::remove_file(directory.join("generation")).unwrap();
        let fourth = reopen(&name);
        assert!(directory.join("generation").exists());
        drop(fourth);
    }

    #[test]
    fn cleanup_never_removes_reused_or_live_or_another_instances_files() {
        let name = instance("clean-a");
        let other = instance("clean-b");
        let (first, cleanup) = local(&name, 64 * 1024, 64 * 1024);
        let (neighbour, other_cleanup) = local(&other, 64 * 1024, 64 * 1024);
        let directory = first.directory().to_path_buf();
        let (arena, staging) = (
            first.arena_path().to_path_buf(),
            first.staging_path().to_path_buf(),
        );
        let inodes = (inode(&arena), inode(&staging));

        // A second manager of the same instance is refused and touches nothing.
        fs::write(directory.join("holders.tmp"), b"x").unwrap();
        let refused = Local::open(&name, 64 * 1024, 64 * 1024).err().unwrap();
        assert!(refused.to_string().contains("already in use"));
        assert!(
            directory.join("holders.tmp").exists(),
            "a live instance is not reclaimed"
        );
        assert_eq!((inode(&arena), inode(&staging)), inodes);

        // After it stops, the debris goes and the reused files stay.
        drop(first);
        let second = reopen(&name);
        assert!(!directory.join("holders.tmp").exists());
        assert_eq!((inode(&arena), inode(&staging)), inodes);
        // The neighbour, live throughout, kept every file.
        for file in ["arena", "staging", "generation", "lock"] {
            assert!(neighbour.directory().join(file).exists(), "{file}");
        }
        drop(second);
        drop(neighbour);
        drop(cleanup);
        drop(other_cleanup);
    }

    #[test]
    fn startup_refuses_a_region_linked_from_another_manager() {
        let name = instance("foreign");
        let (first, _cleanup) = local(&name, 64 * 1024, 64 * 1024);
        let (neighbour, _other_cleanup) = local(&instance("owner"), 64 * 1024, 64 * 1024);
        let path = first.arena_path().to_path_buf();
        let other = neighbour.arena_path();
        let original = inode(other);
        drop(first);
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(other, &path).unwrap();
        let error = Local::open(&name, 64 * 1024, 64 * 1024).err().unwrap();
        assert!(error.to_string().contains("not a regular file owned"));
        assert!(fs::symlink_metadata(path).unwrap().file_type().is_symlink());
        assert_eq!(inode(other), original);
    }

    #[test]
    fn a_previous_runs_live_holder_keeps_the_regions_until_it_reconnects_or_exits() {
        let mut gone = spawn_sleeper();
        let mut back = spawn_sleeper();
        let name = instance("prior");
        let (mut first, cleanup) = local(&name, 64 * 1024, 64 * 1024);
        first.accept(1, gone.0.id()).unwrap();
        first.accept(2, back.0.id()).unwrap();
        // The manager dies without closing anything.
        drop(first);

        let mut second = reopen(&name);
        assert!(second.held_for_previous_run());
        let flag = second.held_flag();
        assert!(flag.load(std::sync::atomic::Ordering::Acquire));
        second.accept(9, std::process::id()).unwrap();
        assert!(second.staging().allocate(4096).is_none());
        assert!(second.arena().allocate(4096).is_none());
        // One holder reconnects: it drained before it did, but the other has not been heard from.
        second.accept(3, back.0.id()).unwrap();
        assert!(second.held_for_previous_run());
        // The other exits.
        gone.0.kill().unwrap();
        gone.0.wait().unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while second.held_for_previous_run() {
            second.poll_prior();
            assert!(Instant::now() < deadline, "the exit was never seen");
            thread::sleep(Duration::from_millis(2));
        }
        assert!(!flag.load(std::sync::atomic::Ordering::Acquire));
        assert!(second.staging().allocate(4096).is_some());
        assert!(second.arena().allocate(4096).is_some());
        drop(second);
        let _ = back.0.kill();
        let _ = back.0.wait();
        drop(cleanup);
    }

    #[test]
    fn buffer_views_are_bounds_checked_and_hold_the_allocation() {
        let name = instance("views");
        let (local, _cleanup) = local(&name, 8192, 8192);
        let extent = Arc::new(local.staging().allocate(4096).unwrap());
        assert_eq!(
            local.offset(Region::Staging, &extent, 0, 4096).unwrap(),
            extent.offset() as u64
        );
        assert!(local.offset(Region::Staging, &extent, 1, 4096).is_err());
        assert!(local.offset(Region::Staging, &extent, 0, 4097).is_err());
        assert!(local.offset(Region::Staging, &extent, 4096, 4096).is_err());
        let hold = extent.clone();
        drop(extent);
        assert_eq!(local.staging_used(), 4096);
        drop(hold);
        assert_eq!(local.staging_used(), 0);
    }
}
