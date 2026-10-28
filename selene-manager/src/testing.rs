//! In-process manager used by client and native integration tests.
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
static HOST_BUSY: AtomicBool = AtomicBool::new(false);

// Serialises manager-backed tests: each manager owns region files, a disk
// worker and a socket, and running twenty of them at once exhausts container
// file descriptors.
thread_local! {
    static HOST_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct HostSlot {
    root: bool,
}

impl HostSlot {
    fn acquire() -> Self {
        let root = HOST_DEPTH.with(|depth| {
            let value = depth.get();
            depth.set(value + 1);
            value == 0
        });
        if root {
            while HOST_BUSY.swap(true, Ordering::Acquire) {
                thread::sleep(Duration::from_millis(5));
            }
        }
        Self { root }
    }
}

impl Drop for HostSlot {
    fn drop(&mut self) {
        HOST_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
        if self.root {
            HOST_BUSY.store(false, Ordering::Release);
        }
    }
}

pub struct Manager {
    pub config: crate::store::Config,
    pub base: std::path::PathBuf,
    _slot: HostSlot,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<std::io::Result<()>>>,
}
impl Manager {
    pub fn new() -> Self {
        Self::sized(8 << 20, 8 << 20)
    }
    pub fn start() -> Self {
        Self::with_memory(16 << 20)
    }
    pub fn with_memory(memory: usize) -> Self {
        let mut manager = Self::sized(memory, 16 << 20);
        manager.run();
        manager
    }
    fn sized(memory: usize, staging: u64) -> Self {
        static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let name = format!(
            "flare-pool-startup-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        );
        let directory = std::env::temp_dir().join(&name);
        std::fs::create_dir_all(&directory).unwrap();
        let mut config = crate::store::Config::dram_only(directory.join("manager.sock"), memory);
        config.name = name;
        config.staging = staging;
        Self {
            config,
            base: directory,
            _slot: HostSlot::acquire(),
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
        }
    }
    pub fn run(&mut self) {
        for attempt in 0..4 {
            if attempt > 0 {
                // A failed attempt may have lost its metrics port after the probe.
                self.config.metrics = Some(free_address());
            }
            match self.try_start() {
                Ok(()) => {
                    // A manager that failed after creating its socket is not serving.
                    thread::sleep(Duration::from_millis(20));
                    if self
                        .thread
                        .as_ref()
                        .is_some_and(|thread| !thread.is_finished())
                    {
                        for address in [
                            self.config.listen,
                            self.config.listen_data,
                            self.config.metrics,
                        ]
                        .into_iter()
                        .flatten()
                        {
                            wait("manager listener", || {
                                std::net::TcpStream::connect(address).is_ok()
                            });
                        }
                        return;
                    }
                    if let Some(thread) = self.thread.take() {
                        let result = thread.join().unwrap();
                        assert!(
                            result
                                .is_err_and(|error| error.kind() == std::io::ErrorKind::AddrInUse)
                        );
                    }
                }
                Err(error) => {
                    assert_eq!(
                        error.kind(),
                        std::io::ErrorKind::AddrInUse,
                        "manager did not start: {error}"
                    );
                }
            }
        }
        panic!("manager did not start");
    }

    pub fn try_start(&mut self) -> std::io::Result<()> {
        self.stop.store(false, Ordering::Release);
        let config = self.config.clone();
        let stop = self.stop.clone();
        self.thread = Some(thread::spawn(move || crate::control::run(config, stop)));
        let deadline = Instant::now() + Duration::from_secs(15);
        while !self.config.socket.exists() {
            if self
                .thread
                .as_ref()
                .is_some_and(thread::JoinHandle::is_finished)
            {
                let handle = self.thread.take().unwrap();
                return handle.join().unwrap();
            }
            if Instant::now() >= deadline {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "manager did not start",
                ));
            }
            thread::sleep(Duration::from_millis(5));
        }
        Ok(())
    }

    /// Asks the manager to stop without waiting for it.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap().unwrap();
        }
        if let Some(root) = &self.config.root {
            wait("root lock", || crate::disk::RootLock::acquire(root).is_ok());
        }
    }

    pub fn restart(&mut self) {
        self.stop();
        self.run();
    }
    pub fn with_storage(memory: usize, nvme: u64) -> Self {
        let mut manager = Self::sized(memory, 16 << 20);
        manager.config.root = Some(manager.base.join("records"));
        manager.config.nvme = nvme;
        manager.config.metrics = Some(free_address());
        manager
    }
    pub fn dram_only(memory: usize) -> Self {
        let mut manager = Self::sized(memory, 16 << 20);
        manager.config.metrics = Some(free_address());
        manager
    }
    pub fn open_config(mut config: crate::store::Config) -> Self {
        config.metrics.get_or_insert_with(free_address);
        let base = config
            .root
            .as_ref()
            .and_then(|root| root.parent())
            .or_else(|| config.socket.parent().and_then(|path| path.parent()))
            .expect("a private test directory")
            .to_path_buf();
        Self {
            config,
            base,
            _slot: HostSlot::acquire(),
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
        }
    }
    pub fn wait(&self, mut condition: impl FnMut(&Self) -> bool) {
        wait("manager condition", || condition(self));
    }
    pub fn metrics(&self) -> String {
        use std::io::{Read, Write};
        let mut stream = std::net::TcpStream::connect(self.config.metrics.unwrap()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        stream
            .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        let mut text = String::new();
        stream.read_to_string(&mut text).unwrap();
        text
    }
    pub fn gauge(&self, name: &str) -> u64 {
        self.metrics()
            .lines()
            .find_map(|line| line.strip_prefix(&format!("selene_{name} ")))
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(0)
    }
    pub fn shm_directory(&self) -> std::path::PathBuf {
        self.config.shm_directory().unwrap()
    }
    pub fn address(&self) -> String {
        format!("unix:{}", self.config.socket.display())
    }
}
impl Drop for Manager {
    fn drop(&mut self) {
        self.stop();
        if self.base.file_name().is_some_and(|name| {
            let name = name.to_string_lossy();
            name.starts_with("selene-") || name.starts_with("flare-pool-startup-")
        }) {
            let _ = std::fs::remove_dir_all(&self.base);
        }
        let regions = self.config.shm_directory().unwrap();
        clean_regions(&regions, true);
    }
}
/// Remove only known test region files, and only while their manager lock is held.
/// Keep the lock inode so another opener cannot acquire a different lock during cleanup.
pub fn remove_regions(directory: &std::path::Path) {
    clean_regions(directory, false);
}

fn clean_regions(directory: &std::path::Path, remove_directory: bool) {
    use std::os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    };
    let Ok(lock) = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(directory.join("lock"))
    else {
        return;
    };
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return;
    }
    for name in [
        "arena",
        "staging",
        "generation",
        "holders",
        "generation.tmp",
        "holders.tmp",
    ] {
        let path = directory.join(name);
        if std::fs::symlink_metadata(&path)
            .is_ok_and(|m| m.is_file() && m.uid() == unsafe { libc::geteuid() })
        {
            let _ = std::fs::remove_file(path);
        }
    }
    if remove_directory {
        // This fixture's unique namespace is never opened again after Drop.
        std::fs::remove_file(directory.join("lock")).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
}

pub fn wait(what: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(10));
    }
}

pub fn free_address() -> std::net::SocketAddr {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

pub fn certificate() -> rcgen::CertifiedKey<rcgen::KeyPair> {
    rcgen::generate_simple_self_signed(vec!["localhost".to_owned()]).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn managers_with_different_pool_names_keep_separate_regions() {
        let mut first = Manager::start();
        let directory = first.config.shm_directory().unwrap();
        let arena = directory.join("arena");
        let inode = std::fs::metadata(&arena).unwrap().ino();
        let mut second = Manager::new();
        second.run();
        let other = second.config.shm_directory().unwrap();
        assert_ne!(directory, other);
        assert_ne!(inode, std::fs::metadata(other.join("arena")).unwrap().ino());

        // Neither a rejected startup nor test cleanup may unlink a live manager's files.
        let error = crate::control::serve(first.config.clone(), Arc::new(AtomicBool::new(false)))
            .unwrap_err();
        assert!(error.to_string().contains("already in use"), "{error}");
        remove_regions(&directory);
        assert_eq!(std::fs::metadata(&arena).unwrap().ino(), inode);
        drop(second);
        assert!(!other.exists());
        assert!(arena.exists());
        first.restart();
        assert_eq!(std::fs::metadata(&arena).unwrap().ino(), inode);
        drop(first);
        assert!(!directory.exists());
    }
}
