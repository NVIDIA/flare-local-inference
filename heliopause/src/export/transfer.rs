//! Authenticated preparation-rank export handoff over established fabric streams.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
#[cfg(test)]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use ring::hmac;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{RankOutput, Result, invalid};

const BLOCK_BYTES: usize = 4 << 20;
const METADATA_BYTES: usize = 256 << 20;
const PROTOCOL: u32 = 1;

#[derive(Clone, Copy)]
enum FileDirection {
    Send,
    Receive,
}
struct FileTransactions {
    send: usize,
    receive: usize,
    reserved: usize,
    capacity: usize,
}
static FILE_TRANSACTIONS: (Mutex<FileTransactions>, Condvar) = (
    Mutex::new(FileTransactions {
        send: 0,
        receive: 0,
        reserved: 0,
        capacity: 0,
    }),
    Condvar::new(),
);
struct FileTransaction {
    direction: FileDirection,
    descriptors: usize,
}

fn descriptor_headroom() -> Result<usize> {
    let mut limit = std::mem::MaybeUninit::<libc::rlimit>::uninit();
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, limit.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let limit = unsafe { limit.assume_init() }.rlim_cur;
    let open = fs::read_dir("/proc/self/fd")?
        .collect::<std::io::Result<Vec<_>>>()?
        .len()
        .saturating_sub(1);
    Ok(usize::try_from(limit)
        .unwrap_or(usize::MAX)
        .saturating_sub(open)
        .saturating_sub(2))
}
impl FileTransaction {
    fn acquire(direction: FileDirection, descriptors: usize) -> Result<Self> {
        let (lock, ready) = &FILE_TRANSACTIONS;
        let mut state = lock.lock().unwrap();
        loop {
            if state.reserved == 0 {
                state.capacity = descriptor_headroom()?;
            }
            // Network writes hold no file transaction; receivers leave room for source reads.
            let peer_reserve = match direction {
                FileDirection::Receive => 3,
                FileDirection::Send => 0,
            };
            if state.reserved + descriptors + peer_reserve <= state.capacity {
                state.reserved += descriptors;
                match direction {
                    FileDirection::Send => state.send += 1,
                    FileDirection::Receive => state.receive += 1,
                }
                ready.notify_all();
                return Ok(Self {
                    direction,
                    descriptors,
                });
            }
            if state.reserved == 0 {
                return Err(invalid(format!(
                    "insufficient file-descriptor headroom for rank export: {} available",
                    state.capacity
                )));
            }
            state = ready.wait(state).unwrap();
        }
    }
}
impl Drop for FileTransaction {
    fn drop(&mut self) {
        let (lock, ready) = &FILE_TRANSACTIONS;
        let mut state = lock.lock().unwrap();
        state.reserved -= self.descriptors;
        match self.direction {
            FileDirection::Send => state.send -= 1,
            FileDirection::Receive => state.receive -= 1,
        }
        ready.notify_all();
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ChannelState {
    pub fd: i32,
    pub key: Vec<u8>,
    pub send_role: String,
    pub receive_role: String,
    pub sent: u64,
    pub received: u64,
}

// The binding retains the Python socket owners until every scoped worker has joined.
#[derive(Clone)]
struct Socket(i32);

impl Socket {
    fn borrow(fd: i32) -> Result<Self> {
        if unsafe { libc::fcntl(fd, libc::F_GETFD) } < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Self(fd))
    }
    fn abort(&self) {
        unsafe {
            libc::shutdown(self.0, libc::SHUT_RDWR);
        }
    }
    fn ready(&self, events: i16) -> std::io::Result<()> {
        let mut poll = libc::pollfd {
            fd: self.0,
            events,
            revents: 0,
        };
        loop {
            let result = unsafe { libc::poll(&mut poll, 1, 30_000) };
            if result > 0 {
                return Ok(());
            }
            if result == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "rank export socket timed out",
                ));
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
}
impl Read for Socket {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        loop {
            self.ready(libc::POLLIN)?;
            let size = unsafe {
                libc::recv(
                    self.0,
                    bytes.as_mut_ptr().cast(),
                    bytes.len(),
                    libc::MSG_DONTWAIT,
                )
            };
            if size >= 0 {
                return Ok(size as usize);
            }
            let error = std::io::Error::last_os_error();
            if !matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
            ) {
                return Err(error);
            }
        }
    }
}
impl Write for Socket {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        loop {
            self.ready(libc::POLLOUT)?;
            let size = unsafe {
                libc::send(
                    self.0,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL,
                )
            };
            if size >= 0 {
                return Ok(size as usize);
            }
            let error = std::io::Error::last_os_error();
            if !matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
            ) {
                return Err(error);
            }
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct Channel {
    socket: Socket,
    state: ChannelState,
}
impl Channel {
    fn new(state: ChannelState) -> Result<Self> {
        if state.key.len() != 32
            || !matches!(
                (state.send_role.as_str(), state.receive_role.as_str()),
                ("root", "peer") | ("peer", "root")
            )
        {
            return Err(invalid("invalid authenticated export channel handoff"));
        }
        Ok(Self {
            socket: Socket::borrow(state.fd)?,
            state,
        })
    }
    fn send(&mut self, mut header: Value, payload: &[u8]) -> Result<()> {
        if payload.len() > BLOCK_BYTES {
            return Err(invalid("export frame exceeds capacity"));
        }
        header["sequence"] = self.state.sent.into();
        header["bytes"] = payload.len().into();
        let header = serde_json::to_vec(&header)?;
        if header.len() > 65536 {
            return Err(invalid("export frame header exceeds capacity"));
        }
        let key = hmac::Key::new(hmac::HMAC_SHA256, &self.state.key);
        let mut signature = hmac::Context::with_key(&key);
        signature.update(self.state.send_role.as_bytes());
        signature.update(&header);
        signature.update(payload);
        self.socket
            .write_all(&(header.len() as u32).to_le_bytes())?;
        self.socket.write_all(&header)?;
        self.socket.write_all(payload)?;
        self.socket.write_all(signature.sign().as_ref())?;
        self.state.sent = self
            .state
            .sent
            .checked_add(1)
            .ok_or_else(|| invalid("export sequence overflow"))?;
        Ok(())
    }
    fn header(&mut self) -> Result<(Value, Vec<u8>, usize)> {
        let mut length = [0; 4];
        self.socket.read_exact(&mut length)?;
        let length = u32::from_le_bytes(length) as usize;
        if length > 65536 {
            return Err(invalid("export frame header exceeds capacity"));
        }
        let mut header = vec![0; length];
        self.socket.read_exact(&mut header)?;
        let value: Value = serde_json::from_slice(&header)?;
        let length = value["bytes"]
            .as_u64()
            .filter(|&n| n <= BLOCK_BYTES as u64)
            .ok_or_else(|| invalid("invalid export frame length"))? as usize;
        Ok((value, header, length))
    }
    fn payload(
        &mut self,
        kind: &str,
        value: &Value,
        header: &[u8],
        payload: &mut [u8],
    ) -> Result<()> {
        let mut signature = [0; 32];
        self.socket.read_exact(payload)?;
        self.socket.read_exact(&mut signature)?;
        let key = hmac::Key::new(hmac::HMAC_SHA256, &self.state.key);
        let mut signed = hmac::Context::with_key(&key);
        signed.update(self.state.receive_role.as_bytes());
        signed.update(header);
        signed.update(payload);
        #[allow(deprecated)]
        ring::constant_time::verify_slices_are_equal(signed.sign().as_ref(), &signature)
            .map_err(|_| invalid("rank export message authentication failed"))?;
        if value["sequence"].as_u64() != Some(self.state.received) {
            return Err(invalid("rank export message sequence differs"));
        }
        if value["kind"] != kind {
            return Err(invalid(format!(
                "expected export {kind}, received {}",
                value["kind"]
            )));
        }
        self.state.received = self
            .state
            .received
            .checked_add(1)
            .ok_or_else(|| invalid("export sequence overflow"))?;
        Ok(())
    }
    fn receive_into(&mut self, kind: &str, payload: &mut [u8]) -> Result<Value> {
        let (value, header, length) = self.header()?;
        if length != payload.len() {
            return Err(invalid("rank export payload range differs"));
        }
        self.payload(kind, &value, &header, payload)?;
        Ok(value)
    }
    #[cfg(test)]
    fn receive(&mut self, kind: &str) -> Result<(Value, Vec<u8>)> {
        let (value, header, length) = self.header()?;
        let mut payload = vec![0; length];
        self.payload(kind, &value, &header, &mut payload)?;
        Ok((value, payload))
    }
}

fn send_control(socket: &mut Socket, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > METADATA_BYTES {
        return Err(invalid("rank export manifest exceeds capacity"));
    }
    socket.write_all(b"N")?;
    socket.write_all(&(bytes.len() as u64).to_le_bytes())?;
    socket.write_all(&bytes)?;
    Ok(())
}
fn receive_control<T: serde::de::DeserializeOwned>(socket: &mut Socket) -> Result<T> {
    let mut header = [0; 9];
    socket.read_exact(&mut header)?;
    let length = u64::from_le_bytes(header[1..].try_into()?);
    if header[0] != b'N' || length > METADATA_BYTES as u64 {
        return Err(invalid("incompatible native rank export protocol"));
    }
    let mut bytes = vec![0; length as usize];
    socket.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn digest_bytes(hex: &str) -> Result<[u8; 32]> {
    if hex.len() != 64 {
        return Err(invalid("invalid rank payload digest"));
    }
    let mut bytes = [0; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)?;
    }
    Ok(bytes)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Source {
    size: u64,
    sha256: String,
    blob: bool,
}
#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    output: RankOutput,
    files: BTreeMap<PathBuf, Source>,
    inventory: Value,
}
#[derive(Serialize, Deserialize)]
struct Request {
    version: u32,
    files: Vec<PathBuf>,
}

fn streams<T: Send>(
    states: Vec<ChannelState>,
    action: impl Fn(usize, &mut Channel) -> Result<T> + Sync,
) -> Result<(Vec<T>, Vec<ChannelState>)> {
    if states.is_empty() || states.len() > 64 {
        return Err(invalid(
            "rank export requires 1..64 established fabric streams",
        ));
    }
    let channels: Vec<_> = states
        .into_iter()
        .map(Channel::new)
        .collect::<Result<_>>()?;
    streams_ready(channels, action)
}

fn streams_ready<T: Send>(
    channels: Vec<Channel>,
    action: impl Fn(usize, &mut Channel) -> Result<T> + Sync,
) -> Result<(Vec<T>, Vec<ChannelState>)> {
    let abort: Vec<_> = channels
        .iter()
        .map(|channel| channel.socket.clone())
        .collect();
    let failure = Mutex::new(None);
    std::thread::scope(|scope| {
        let handles: Vec<_> = channels
            .into_iter()
            .enumerate()
            .map(|(stream, mut channel)| {
                let action = &action;
                let failure = &failure;
                let abort = &abort;
                scope.spawn(move || match action(stream, &mut channel) {
                    Ok(value) => Some((value, channel.state)),
                    Err(error) => {
                        let mut first = failure.lock().unwrap();
                        if first.is_none() {
                            eprintln!("rank export failed: {error}");
                            *first = Some(error);
                            for socket in abort {
                                socket.abort();
                            }
                        }
                        None
                    }
                })
            })
            .collect();
        let mut values = Vec::new();
        let mut states = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(Some((value, state))) => {
                    values.push(value);
                    states.push(state);
                }
                Ok(None) => {}
                Err(_) => {
                    for socket in &abort {
                        socket.abort();
                    }
                    return Err(invalid("rank export worker panicked"));
                }
            }
        }
        if let Some(error) = failure.lock().unwrap().take() {
            return Err(error);
        }
        Ok((values, states))
    })
}

fn completion(
    channel: &mut Channel,
    names: &[PathBuf],
    files: &BTreeMap<PathBuf, Source>,
    sender: bool,
) -> Result<()> {
    let stats = json!({"files": names.len(), "payload_bytes": names.iter().map(|n| files[n].size).sum::<u64>()});
    if sender {
        channel.send(json!({"kind":"export-finished", "stats": stats}), &[])?;
    }
    let message = channel.receive_into("export-finished", &mut [])?;
    if message["stats"] != stats {
        return Err(invalid("rank export completion accounting differs"));
    }
    if !sender {
        channel.send(json!({"kind":"export-finished", "stats": stats}), &[])?;
    }
    Ok(())
}

pub(crate) fn send_rank(
    output: RankOutput,
    inventory: Value,
    control_fd: i32,
    channels: Vec<ChannelState>,
) -> Result<Vec<ChannelState>> {
    if channels.is_empty() {
        return Err(invalid(
            "rank export requires established fabric channels; management payload transfer is forbidden",
        ));
    }
    let budget = super::io::IoBudget::for_ranks(1)?;
    let progress = helios_core::startup::StartupProgress::silent();
    let scan_phase = progress.phase("Hash rank export sources", None);
    let mut hashed_bytes = 0;
    let mut files = BTreeMap::new();
    let mut identities = BTreeMap::new();
    for part in &output.payload {
        let path = &part.source.path;
        if files.contains_key(path) {
            continue;
        }
        let _transaction = FileTransaction::acquire(FileDirection::Send, 3)?;
        let file = Arc::new(super::io::open_read(path)?);
        let before = super::io::Identity::of(&file)?;
        let mut digest = Sha256::new();
        let source = super::io::Source::from_file(file.clone(), 0, file.metadata()?.len())?;
        budget.scan(&source, |bytes| {
            digest.update(bytes);
            hashed_bytes += bytes.len() as u64;
            scan_phase.advance(bytes.len() as u64);
            scan_phase.detail(format!("{hashed_bytes} bytes hashed"));
            Ok(())
        })?;
        if super::io::Identity::of(&file)? != before {
            return Err(invalid("rank export source changed during metadata scan"));
        }
        let sha256 = format!("{:x}", digest.finalize());
        let blob =
            path.file_name().and_then(|n| n.to_str()) == Some(&format!("{sha256}.safetensors"));
        files.insert(
            path.clone(),
            Source {
                size: file.metadata()?.len(),
                sha256,
                blob,
            },
        );
        identities.insert(path.clone(), before);
    }
    scan_phase.finish(format!(
        "{} files, {hashed_bytes} bytes hashed",
        files.len()
    ));
    let mut control = Socket::borrow(control_fd)?;
    send_control(
        &mut control,
        &Manifest {
            version: PROTOCOL,
            output,
            files: files.clone(),
            inventory,
        },
    )?;
    let request: Request = receive_control(&mut control)?;
    let unique: BTreeSet<_> = request.files.iter().collect();
    if request.version != PROTOCOL
        || unique.len() != request.files.len()
        || request.files.iter().any(|n| !files.contains_key(n))
    {
        return Err(invalid("invalid requested rank export sources"));
    }
    let count = channels.len();
    let total: u64 = request.files.iter().map(|name| files[name].size).sum();
    let phase = progress.phase("Send export payload", Some(total));
    let (_, states) = streams(channels, |stream, channel| {
        let names: Vec<_> = request
            .files
            .iter()
            .skip(stream)
            .step_by(count)
            .cloned()
            .collect();
        let mut buffer = budget.buffer()?;
        for name in &names {
            let size = files[name].size;
            let mut offset = 0;
            while offset < size {
                let length = (size - offset).min(BLOCK_BYTES as u64) as usize;
                {
                    let _transaction = FileTransaction::acquire(FileDirection::Send, 3)?;
                    let source = Arc::new(super::io::open_read(name)?);
                    if super::io::Identity::of(&source)? != identities[name] {
                        return Err(invalid("rank export source changed before transfer"));
                    }
                    let pinned = super::io::Source::from_file(source.clone(), 0, size)?;
                    pinned.read_at(offset, &mut buffer.bytes_mut()[..length])?;
                    if super::io::Identity::of(&source)? != identities[name] {
                        return Err(invalid("rank export source changed during transfer"));
                    }
                }
                channel.send(
                    json!({"kind":"export-data", "name":name, "offset":offset}),
                    &buffer.bytes()[..length],
                )?;
                offset += length as u64;
                phase.advance_bytes(length as u64);
            }
        }
        completion(channel, &names, &files, true)
    })?;
    phase.finish(format!("{total} bytes sent over {count} streams"));
    Ok(states)
}

#[derive(Deserialize)]
pub(crate) struct ReceiveRequest {
    pub control_fd: i32,
    pub channels: Vec<ChannelState>,
    pub destination: PathBuf,
}
#[derive(Serialize)]
pub(crate) struct ReceiveOutput {
    #[cfg(test)]
    #[serde(skip_serializing)]
    pub output: RankOutput,
    pub inventory: Value,
    pub channels: Vec<ChannelState>,
    pub stats: Value,
}

struct ReceiveProgress {
    phase: helios_core::startup::Phase,
    bytes: Mutex<u64>,
}
impl ReceiveProgress {
    fn advance(&self, count: u64) {
        let mut bytes = self.bytes.lock().unwrap();
        *bytes += count;
        self.phase.advance(count);
        self.phase.detail(format!("{} bytes received", *bytes));
    }
}

fn receive_rank(
    request: ReceiveRequest,
    mut control: Socket,
    channels: Vec<Channel>,
    cache: &Path,
    no_weights: bool,
    budget: &Arc<super::io::IoBudget>,
    aggregate: &ReceiveProgress,
) -> Result<ReceiveOutput> {
    if request.channels.is_empty() {
        return Err(invalid(
            "rank export requires established fabric channels; management payload transfer is forbidden",
        ));
    }
    let mut manifest: Manifest = receive_control(&mut control)?;
    if manifest.version != PROTOCOL {
        return Err(invalid("incompatible native rank export protocol version"));
    }
    let mut logical_end = 0;
    for part in &manifest.output.payload {
        if part.offset < logical_end || part.source.offset.checked_add(part.source.nbytes).is_none()
        {
            return Err(invalid(
                "rank export payload ranges must be ordered and disjoint",
            ));
        }
        logical_end = part
            .offset
            .checked_add(part.source.nbytes)
            .ok_or_else(|| invalid("rank export payload length overflow"))?;
    }
    super::package::validate_index(&manifest.output.index, logical_end)?;
    fs::create_dir_all(
        request
            .destination
            .parent()
            .ok_or_else(|| invalid("rank export destination has no parent"))?,
    )?;
    let store = super::blob::BlobStore::new(cache, budget.clone())?;
    let progress = helios_core::startup::StartupProgress::silent();
    let cache_phase = progress.phase("Verify rank export cache", None);
    let mut cached_bytes = 0;
    let mut cached_files = 0;
    let mut targets = BTreeMap::new();
    let mut requested = Vec::new();
    let referenced: BTreeSet<_> = manifest
        .output
        .payload
        .iter()
        .map(|p| p.source.path.clone())
        .collect();
    if referenced != manifest.files.keys().cloned().collect() {
        return Err(invalid(
            "rank export file inventory differs from payload sources",
        ));
    }
    for (name, info) in &manifest.files {
        if info.sha256.len() != 64
            || !info
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || info.size == 0
            || info.size > 64 << 30
        {
            return Err(invalid("invalid rank export file digest or size"));
        }
        let target = if info.blob {
            store.root.join(format!("{}.safetensors", info.sha256))
        } else {
            request
                .destination
                .parent()
                .unwrap()
                .join(format!("{}.source", info.sha256))
        };
        if target.try_exists()? {
            let _transaction = FileTransaction::acquire(FileDirection::Receive, 3)?;
            super::blob::verify_file(&target, digest_bytes(&info.sha256)?, info.size, &budget)?;
            cached_bytes += info.size;
            cached_files += 1;
            cache_phase.advance(info.size);
            cache_phase.detail(format!(
                "{cached_files} cached files, {cached_bytes} bytes verified"
            ));
        } else {
            if info.blob && no_weights {
                return Err(invalid(
                    "--no-weights requires the peer stored-data blob in the coordinator cache",
                ));
            }
            requested.push(name.clone());
        }
        targets.insert(name.clone(), target);
    }
    for part in &manifest.output.payload {
        if part
            .source
            .offset
            .checked_add(part.source.nbytes)
            .filter(|&end| end <= manifest.files[&part.source.path].size)
            .is_none()
        {
            return Err(invalid("rank export source range exceeds its file"));
        }
    }
    cache_phase.finish(format!(
        "{cached_files} cached files, {cached_bytes} bytes verified"
    ));
    send_control(
        &mut control,
        &Request {
            version: PROTOCOL,
            files: requested.clone(),
        },
    )?;
    let started = Instant::now();
    let count = request.channels.len();
    let total: u64 = requested.iter().map(|name| manifest.files[name].size).sum();
    let phase = progress.phase("Receive export payload", Some(total));
    let (_, channels) = streams_ready(channels, |stream, channel| {
        let names: Vec<_> = requested
            .iter()
            .skip(stream)
            .step_by(count)
            .cloned()
            .collect();
        for name in &names {
            let info = &manifest.files[name];
            let target = &targets[name];
            channel.socket.ready(libc::POLLIN)?;
            let _transaction = FileTransaction::acquire(FileDirection::Receive, 4)?;
            let (file, partial) =
                super::archive::temporary(target.parent().unwrap(), ".rank-blob-")?;
            let result = (|| -> Result<()> {
                let mut digest = Sha256::new();
                budget.write(&file, 0, info.size, |position, bytes| {
                    let mut filled = 0;
                    while filled < bytes.len() {
                        let offset = position + filled as u64;
                        let length = (info.size - offset).min(BLOCK_BYTES as u64) as usize;
                        if length > bytes.len() - filled {
                            return Err(invalid("rank export payload range differs"));
                        }
                        let payload = &mut bytes[filled..filled + length];
                        let message = channel.receive_into("export-data", payload)?;
                        if message["name"].as_str() != name.to_str()
                            || message["offset"].as_u64() != Some(offset)
                        {
                            return Err(invalid("rank export payload range differs"));
                        }
                        digest.update(payload);
                        filled += length;
                        phase.advance_bytes(length as u64);
                        aggregate.advance(length as u64);
                    }
                    Ok(())
                })?;
                if format!("{:x}", digest.finalize()) != info.sha256 {
                    return Err(invalid("rank payload digest mismatch"));
                }
                file.sync_all()?;
                if info.blob {
                    file.set_permissions({
                        use std::os::unix::fs::PermissionsExt;
                        fs::Permissions::from_mode(0o444)
                    })?;
                }
                match fs::hard_link(&partial, target) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        super::blob::verify_file(
                            target,
                            digest_bytes(&info.sha256)?,
                            info.size,
                            &budget,
                        )?;
                    }
                    Err(error) => return Err(error.into()),
                }
                Ok(())
            })();
            let _ = fs::remove_file(&partial);
            result?;
        }
        completion(channel, &names, &manifest.files, false)
    })?;
    for part in &mut manifest.output.payload {
        part.source.path = targets[&part.source.path].clone();
    }
    manifest.output.staging = None;
    {
        let _transaction = FileTransaction::acquire(FileDirection::Receive, 1)?;
        super::package::write_sources(&request.destination, &manifest.output)?;
    }
    let seconds = started.elapsed().as_secs_f64();
    let size: u64 = requested.iter().map(|n| manifest.files[n].size).sum();
    phase.finish(format!(
        "{size} bytes received, {cached_files} cached files ({cached_bytes} bytes)"
    ));
    Ok(ReceiveOutput {
        #[cfg(test)]
        output: manifest.output,
        inventory: manifest.inventory,
        channels,
        stats: json!({"payload_bytes":size,"payload_seconds":seconds,"payload_gbps":size as f64 * 8.0 / seconds.max(f64::MIN_POSITIVE) / 1e9,"files":requested.len(),"streams":count,"transport":"fabric-tcp","cached_bytes":cached_bytes,"cached_files":cached_files}),
    })
}

pub(crate) fn receive_ranks(
    requests: Vec<ReceiveRequest>,
    cache: &Path,
    no_weights: bool,
) -> Result<Vec<ReceiveOutput>> {
    let budget = super::io::IoBudget::for_ranks(1)?;
    let progress = helios_core::startup::StartupProgress::stderr_rows();
    let aggregate = ReceiveProgress {
        phase: progress.phase("Receive rank export payloads", None),
        bytes: Mutex::new(0),
    };
    let jobs = requests
        .into_iter()
        .map(|request| -> Result<_> {
            if request.channels.len() > 64 {
                return Err(invalid(
                    "rank export requires 1..64 established fabric streams",
                ));
            }
            let control = Socket::borrow(request.control_fd)?;
            let channels = request
                .channels
                .iter()
                .cloned()
                .map(Channel::new)
                .collect::<Result<Vec<_>>>()?;
            Ok((request, control, channels))
        })
        .collect::<Result<Vec<_>>>()?;
    let abort: Vec<_> = jobs
        .iter()
        .flat_map(|(_, control, channels)| {
            std::iter::once(control.clone())
                .chain(channels.iter().map(|channel| channel.socket.clone()))
        })
        .collect();
    let failure = Mutex::new(None);
    let outputs = std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .into_iter()
            .map(|(request, control, channels)| {
                let failure = &failure;
                let abort = &abort;
                let budget = &budget;
                let aggregate = &aggregate;
                scope.spawn(move || {
                    match receive_rank(
                        request, control, channels, cache, no_weights, budget, aggregate,
                    ) {
                        Ok(output) => Some(output),
                        Err(error) => {
                            let mut first = failure.lock().unwrap();
                            if first.is_none() {
                                eprintln!("rank export failed: {error}");
                                *first = Some(error);
                                for socket in abort {
                                    socket.abort();
                                }
                            }
                            None
                        }
                    }
                })
            })
            .collect();
        let mut outputs = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(Some(output)) => outputs.push(output),
                Ok(None) => {}
                Err(_) => {
                    for socket in &abort {
                        socket.abort();
                    }
                    return Err(invalid("rank export receiver panicked"));
                }
            }
        }
        if let Some(error) = failure.lock().unwrap().take() {
            return Err(error);
        }
        Ok(outputs)
    })?;
    let received: u64 = outputs
        .iter()
        .map(|output| output.stats["payload_bytes"].as_u64().unwrap_or(0))
        .sum();
    let cached: u64 = outputs
        .iter()
        .map(|output| output.stats["cached_bytes"].as_u64().unwrap_or(0))
        .sum();
    aggregate.phase.finish(format!(
        "{received} bytes received, {cached} cached bytes across {} ranks",
        outputs.len()
    ));
    Ok(outputs)
}

#[pyo3::pyfunction]
fn export_send_rank(
    py: pyo3::Python<'_>,
    output: pyo3::PyRef<'_, super::binding::NativeRankOutput>,
    inventory: &str,
    control_fd: i32,
    channels: &str,
    owners: pyo3::Py<pyo3::PyAny>,
) -> pyo3::PyResult<String> {
    use pyo3::types::PyAnyMethods;
    if owners.bind(py).is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rank export requires retained socket owners with exclusive use until return",
        ));
    }
    let parse = || -> Result<_> {
        Ok((
            serde_json::from_str(inventory)?,
            serde_json::from_str(channels)?,
        ))
    };
    let (inventory, channels) =
        parse().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let output = output.output.clone();
    let mut lease = crate::lifetime::OwnerLease(Some(owners));
    let result = py.detach(move || -> Result<String> {
        Ok(serde_json::to_string(&send_rank(
            output, inventory, control_fd, channels,
        )?)?)
    });
    drop(lease.take_after_drain());
    result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyo3::pyfunction]
fn export_receive_ranks(
    py: pyo3::Python<'_>,
    requests: &str,
    cache: PathBuf,
    no_weights: bool,
    owners: pyo3::Py<pyo3::PyAny>,
) -> pyo3::PyResult<String> {
    use pyo3::types::PyAnyMethods;
    if owners.bind(py).is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rank export requires retained socket owners with exclusive use until return",
        ));
    }
    let requests = serde_json::from_str(requests)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let mut lease = crate::lifetime::OwnerLease(Some(owners));
    let result = py.detach(move || -> Result<String> {
        Ok(serde_json::to_string(&receive_ranks(
            requests, &cache, no_weights,
        )?)?)
    });
    drop(lease.take_after_drain());
    result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

pub(crate) fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    module.add_function(pyo3::wrap_pyfunction!(export_send_rank, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(export_receive_ranks, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicU64, Ordering};

    static SERIAL: AtomicU64 = AtomicU64::new(0);
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "flare-export-transfer-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
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
    fn sockets() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (root, _) = listener.accept().unwrap();
        (root, peer)
    }
    fn state(socket: &TcpStream, server: bool) -> ChannelState {
        ChannelState {
            fd: socket.as_raw_fd(),
            key: vec![23; 32],
            send_role: if server { "root" } else { "peer" }.into(),
            receive_role: if server { "peer" } else { "root" }.into(),
            sent: 17,
            received: 17,
        }
    }
    fn output(path: PathBuf, length: u64) -> RankOutput {
        RankOutput {
            index: super::super::wire::ModelIndex {
                version: 4,
                ranks: vec![Default::default()],
                ..Default::default()
            },
            payload: vec![super::super::FilePart {
                offset: 0,
                source: super::super::FileSource {
                    path,
                    offset: 0,
                    nbytes: length,
                },
            }],
            allocations: vec![],
            module_globals: vec![],
            external: vec![],
            staging: None,
        }
    }
    fn roundtrip(
        directory: &Directory,
        name: &str,
        source: &RankOutput,
        no_weights: bool,
    ) -> Result<ReceiveOutput> {
        let (control_root, control_peer) = sockets();
        let (root, peer) = sockets();
        std::thread::scope(|scope| {
            let sender = scope.spawn(|| {
                send_rank(
                    source.clone(),
                    json!({"test":true}),
                    control_peer.as_raw_fd(),
                    vec![state(&peer, false)],
                )
            });
            let receiver = receive_ranks(
                vec![ReceiveRequest {
                    control_fd: control_root.as_raw_fd(),
                    channels: vec![state(&root, true)],
                    destination: directory.0.join(name),
                }],
                &directory.0.join("cache"),
                no_weights,
            );
            let sent = sender.join().unwrap();
            match receiver {
                Ok(mut outputs) => {
                    sent?;
                    Ok(outputs.remove(0))
                }
                Err(error) => Err(error),
            }
        })
    }
    #[test]
    fn loopback_streams_payload_and_preserves_sequence() {
        let directory = Directory::new();
        let bytes = vec![71; BLOCK_BYTES + 37];
        let path = directory.0.join("resource");
        fs::write(&path, &bytes).unwrap();
        let received = roundtrip(
            &directory,
            "rank.json",
            &output(path, bytes.len() as u64),
            false,
        )
        .unwrap();
        assert_eq!(
            fs::read(&received.output.payload[0].source.path).unwrap(),
            bytes
        );
        assert_eq!(received.stats["payload_bytes"], bytes.len());
        assert_eq!(received.channels[0].received, 20);
        assert_eq!(received.channels[0].sent, 18);
        assert!(directory.0.join("rank.json").exists());
    }
    #[test]
    fn multiple_ranks_and_streams_share_cache_and_drain() {
        let directory = Directory::new();
        let shared = vec![87; 16_385];
        let shared_path = directory
            .0
            .join(format!("{:x}.safetensors", Sha256::digest(&shared)));
        fs::write(&shared_path, &shared).unwrap();
        let mut requests = Vec::new();
        let mut peers = Vec::new();
        let mut retained = Vec::new();
        for rank in 0..2 {
            let (control, peer_control) = sockets();
            let mut roots = Vec::new();
            let mut rank_peers = Vec::new();
            for _ in 0..2 {
                let (root, peer) = sockets();
                roots.push(state(&root, true));
                retained.push(root);
                rank_peers.push(peer);
            }
            requests.push(ReceiveRequest {
                control_fd: control.as_raw_fd(),
                channels: roots,
                destination: directory.0.join(format!("rank-{rank}.json")),
            });
            retained.push(control);
            let mut prepared = output(shared_path.clone(), shared.len() as u64);
            let mut cursor = shared.len() as u64;
            for item in 0..2 {
                let data = vec![rank as u8 + item as u8; 8193];
                let path = directory.0.join(format!("rank-{rank}-resource-{item}"));
                fs::write(&path, &data).unwrap();
                prepared.payload.push(super::super::FilePart {
                    offset: cursor,
                    source: super::super::FileSource {
                        path,
                        offset: 0,
                        nbytes: data.len() as u64,
                    },
                });
                cursor += data.len() as u64;
            }
            peers.push((peer_control, rank_peers, prepared, rank));
        }
        std::thread::scope(|scope| {
            let senders: Vec<_> = peers
                .into_iter()
                .map(|(control, channels, prepared, rank)| {
                    scope.spawn(move || {
                        let states = channels.iter().map(|c| state(c, false)).collect();
                        send_rank(prepared, json!({"rank":rank}), control.as_raw_fd(), states)
                    })
                })
                .collect();
            let received = receive_ranks(requests, &directory.0.join("cache"), false).unwrap();
            for sender in senders {
                assert_eq!(sender.join().unwrap().unwrap().len(), 2);
            }
            assert_eq!(received.len(), 2);
            for (rank, result) in received.iter().enumerate() {
                assert_eq!(result.inventory["rank"], rank);
                assert_eq!(result.channels.len(), 2);
                assert_eq!(
                    fs::read(&result.output.payload[0].source.path).unwrap(),
                    shared
                );
            }
            assert_eq!(
                received[0].output.payload[0].source.path,
                received[1].output.payload[0].source.path
            );
        });
    }
    #[test]
    fn verified_cache_hit_transfers_no_weights() {
        let directory = Directory::new();
        let bytes = vec![19; 8193];
        let name = format!("{:x}.safetensors", Sha256::digest(&bytes));
        let path = directory.0.join(name);
        fs::write(&path, &bytes).unwrap();
        let rank = output(path, bytes.len() as u64);
        roundtrip(&directory, "first.json", &rank, false).unwrap();
        let result = roundtrip(&directory, "second.json", &rank, true).unwrap();
        assert_eq!(result.stats["payload_bytes"], 0);
        assert_eq!(result.channels[0].received, 18);
    }
    #[test]
    fn missing_weights_fail_without_receiving_payload() {
        let directory = Directory::new();
        let bytes = vec![19; 8193];
        let path = directory
            .0
            .join(format!("{:x}.safetensors", Sha256::digest(&bytes)));
        fs::write(&path, &bytes).unwrap();
        let error = roundtrip(
            &directory,
            "rank.json",
            &output(path, bytes.len() as u64),
            true,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("--no-weights"));
        assert!(!directory.0.join("rank.json").exists());
    }
    #[test]
    fn corrupt_cached_blob_is_rejected() {
        let directory = Directory::new();
        let bytes = vec![19; 8193];
        let name = format!("{:x}.safetensors", Sha256::digest(&bytes));
        let path = directory.0.join(&name);
        fs::write(&path, &bytes).unwrap();
        fs::create_dir_all(directory.0.join("cache/blobs")).unwrap();
        fs::write(
            directory.0.join("cache/blobs").join(name),
            vec![20; bytes.len()],
        )
        .unwrap();
        assert!(
            roundtrip(
                &directory,
                "rank.json",
                &output(path, bytes.len() as u64),
                false
            )
            .is_err()
        );
        assert!(!directory.0.join("rank.json").exists());
    }
    #[test]
    fn duplicate_requested_sources_are_rejected() {
        let directory = Directory::new();
        let path = directory.0.join("source");
        fs::write(&path, b"payload").unwrap();
        let (control_root, control_peer) = sockets();
        let (_root, peer) = sockets();
        std::thread::scope(|scope| {
            let sender = scope.spawn(|| {
                send_rank(
                    output(path.clone(), 7),
                    Value::Null,
                    control_peer.as_raw_fd(),
                    vec![state(&peer, false)],
                )
            });
            let mut control = Socket::borrow(control_root.as_raw_fd()).unwrap();
            let manifest: Manifest = receive_control(&mut control).unwrap();
            let name = manifest.files.keys().next().unwrap().clone();
            send_control(
                &mut control,
                &Request {
                    version: PROTOCOL,
                    files: vec![name.clone(), name],
                },
            )
            .unwrap();
            assert!(
                sender
                    .join()
                    .unwrap()
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("invalid requested")
            );
        });
    }
    #[test]
    fn incompatible_export_version_is_rejected() {
        let directory = Directory::new();
        let (control_root, control_peer) = sockets();
        let (root, _peer) = sockets();
        let mut control = Socket::borrow(control_peer.as_raw_fd()).unwrap();
        send_control(
            &mut control,
            &Manifest {
                version: PROTOCOL + 1,
                output: output(directory.0.join("unused"), 7),
                files: BTreeMap::new(),
                inventory: Value::Null,
            },
        )
        .unwrap();
        let error = receive_ranks(
            vec![ReceiveRequest {
                control_fd: control_root.as_raw_fd(),
                channels: vec![state(&root, true)],
                destination: directory.0.join("rank.json"),
            }],
            &directory.0.join("cache"),
            false,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("protocol version"));
    }
    #[test]
    fn incorrect_authentication_is_rejected() {
        let (root, peer) = sockets();
        let mut sender = Channel::new(state(&peer, false)).unwrap();
        let mut root_state = state(&root, true);
        root_state.key[0] ^= 1;
        let mut receiver = Channel::new(root_state).unwrap();
        sender.send(json!({"kind":"export-data"}), b"test").unwrap();
        assert!(
            receiver
                .receive("export-data")
                .err()
                .unwrap()
                .to_string()
                .contains("authentication")
        );
    }
    #[test]
    fn replayed_sequence_is_rejected() {
        let (root, peer) = sockets();
        let mut sender = Channel::new(state(&peer, false)).unwrap();
        let mut receiver = Channel::new(state(&root, true)).unwrap();
        sender
            .send(json!({"kind":"export-data"}), b"first")
            .unwrap();
        receiver.receive("export-data").unwrap();
        sender.state.sent -= 1;
        sender
            .send(json!({"kind":"export-data"}), b"replay")
            .unwrap();
        assert!(
            receiver
                .receive("export-data")
                .err()
                .unwrap()
                .to_string()
                .contains("sequence")
        );
    }
    #[test]
    fn truncated_frame_and_disconnect_are_rejected() {
        for bytes in [&b""[..], &b"\x08\x00\x00\x00{}"[..]] {
            let (root, mut peer) = sockets();
            let mut receiver = Channel::new(state(&root, true)).unwrap();
            peer.write_all(bytes).unwrap();
            drop(peer);
            assert!(receiver.receive("export-data").is_err());
        }
    }
    #[test]
    fn damaged_payloads_never_publish_files() {
        for truncated in [false, true] {
            let directory = Directory::new();
            let (control_root, control_peer) = sockets();
            let (root, peer) = sockets();
            let source = directory.0.join("source");
            let bytes = vec![41; 8193];
            let digest = format!("{:x}", Sha256::digest(&bytes));
            let mut files = BTreeMap::new();
            files.insert(
                source.clone(),
                Source {
                    size: bytes.len() as u64,
                    sha256: digest.clone(),
                    blob: false,
                },
            );
            std::thread::scope(|scope| {
                let sender = scope.spawn(move || {
                    let mut control = Socket::borrow(control_peer.as_raw_fd()).unwrap();
                    send_control(&mut control, &Manifest { version: PROTOCOL, output: output(source.clone(), bytes.len() as u64), files, inventory: Value::Null }).unwrap();
                    let _: Request = receive_control(&mut control).unwrap();
                    let mut channel = Channel::new(state(&peer, false)).unwrap();
                    if truncated {
                        let header = serde_json::to_vec(&json!({"kind":"export-data","name":source,"offset":0,"sequence":17,"bytes":bytes.len()})).unwrap();
                        channel.socket.write_all(&(header.len() as u32).to_le_bytes()).unwrap();
                        channel.socket.write_all(&header).unwrap();
                        channel.socket.write_all(b"truncated").unwrap();
                    } else {
                        channel.send(json!({"kind":"export-data","name":source,"offset":0}), &vec![42; bytes.len()]).unwrap();
                    }
                });
                let result = receive_ranks(
                    vec![ReceiveRequest {
                        control_fd: control_root.as_raw_fd(),
                        channels: vec![state(&root, true)],
                        destination: directory.0.join("rank.json"),
                    }],
                    &directory.0.join("cache"),
                    false,
                );
                sender.join().unwrap();
                assert!(result.is_err());
                if !truncated {
                    assert!(
                        result
                            .err()
                            .unwrap()
                            .to_string()
                            .contains("digest mismatch")
                    );
                }
            });
            assert!(!directory.0.join("rank.json").exists());
            assert!(!directory.0.join(format!("{digest}.source")).exists());
            assert!(!fs::read_dir(&directory.0).unwrap().any(|entry| {
                entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".rank-blob-")
            }));
        }
    }
    #[test]
    fn frame_size_is_bounded_before_allocation() {
        let (root, mut peer) = sockets();
        let mut receiver = Channel::new(state(&root, true)).unwrap();
        peer.write_all(&65537u32.to_le_bytes()).unwrap();
        assert!(
            receiver
                .receive("export-data")
                .err()
                .unwrap()
                .to_string()
                .contains("capacity")
        );
    }
}
