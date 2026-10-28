//! The data channel instances on other hosts copy through.
//!
//! An instance without shared memory cannot map the manager's windows, so the
//! manager moves their bytes itself, over one connection per transfer. A claim's
//! windows (filled by the store, in the manager's own regions) are written to the
//! instance in order; a put's windows (granted by the store) are filled from what
//! the instance writes. The framing is `flare_selene::shared::frame`.
//!
//! The pool actor owns every decision. It opens a transfer and gets its token,
//! queues windows or grants as the store produces them, and stops the transfer
//! when the claim or put ends. A handler task serves the transfer's connection
//! and reports what it did through [`Event`]. Physical fencing is the handler's
//! exit: `Closed` is sent only after the connection has been dropped, so no byte
//! of a window is read or written afterwards. The actor must not release a
//! window to the store before `Closed` (or before `stop` answered that no handler
//! ever ran).

use std::{
    collections::{HashMap, VecDeque},
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use flare_selene::shared::{
    SliceLayout,
    frame::{Carrier, NO_SLOT, Shape, TOKEN_BYTES, ToInstance, ToManager, Window, within},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    runtime::Handle,
    sync::{Notify, mpsc, watch},
    time::timeout,
};
use tokio_rustls::TlsAcceptor;

use super::{Region, invalid};
use crate::regions::LocalRegions;

/// A connection that has not presented its token by then is dropped.
const TOKEN_TIMEOUT: Duration = Duration::from_secs(5);
/// A write or a read that makes no progress for this long ends the transfer.
const IO_TIMEOUT: Duration = Duration::from_secs(20);
/// The most a closing connection may take to say goodbye.
const GOODBYE: Duration = Duration::from_millis(200);
const SOCKET_BUFFER: u64 = 16 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The manager sends windows of a claim.
    Get,
    /// The manager receives windows of a put.
    Put,
}

/// A claim's window, filled by the store, with the checksums stored with its
/// record. The window lies in one of the manager's regions.
#[derive(Clone, Debug)]
pub struct WindowOut {
    pub shape: Shape,
    pub page_xxh3: Vec<u64>,
    pub row_xxh3: u64,
    pub region: Region,
    pub offset: u64,
}

/// A put's window, granted by the store for the instance to fill.
#[derive(Clone, Copy, Debug)]
pub struct GrantOut {
    pub shape: Shape,
    pub region: Region,
    pub offset: u64,
}

/// What a handler reports. `Closed` is always last.
#[derive(Debug)]
pub enum Event {
    /// A claim's window is in the socket; the window may be released.
    Sent { transfer: u64, index: u32 },
    /// A put's window is filled, with the checksums the instance stated.
    Filled {
        transfer: u64,
        index: u32,
        row_xxh3: Option<u64>,
        page_xxh3: Vec<u64>,
    },
    /// The handler ended the transfer on an error.
    Failed { transfer: u64, reason: String },
    /// The handler has exited and its connection is closed.
    Closed { transfer: u64 },
}

struct Queues {
    windows: VecDeque<WindowOut>,
    grants: VecDeque<GrantOut>,
    finished: bool,
    attached: bool,
}

struct Transfer {
    kind: Kind,
    /// A leg of a multi-process instance's rank: the transfer carries only that rank's
    /// slice of each window (`Shape::length` is the slice's length), gathered out of the
    /// window before it is sent and scattered into it after it is received.
    slicing: Option<(Arc<SliceLayout>, usize)>,
    token: [u8; TOKEN_BYTES],
    queues: Mutex<Queues>,
    wake: Notify,
    stopped: watch::Sender<bool>,
}

#[derive(Default)]
struct State {
    tokens: HashMap<[u8; TOKEN_BYTES], u64>,
    transfers: HashMap<u64, Arc<Transfer>>,
    open: bool,
}

/// The manager's RDMA device. A transfer whose instance asks for RDMA gets a queue pair
/// from it; the pair's description goes to the instance and the instance's comes back.
pub trait Rdma: Send + Sync {
    fn pair(&self) -> io::Result<Box<dyn Pair>>;
}

/// A queue pair made for one transfer, not yet connected.
pub trait Pair: Send {
    /// This end's description for the peer.
    fn handshake(&self) -> Vec<u8>;
    fn connect(&mut self, peer: &[u8]) -> io::Result<()>;
    /// Post one signalled write into the peer's staging. Poll it before posting another.
    fn write(
        &mut self,
        region: Region,
        offset: u64,
        len: u64,
        remote_offset: u64,
    ) -> io::Result<()>;
    fn poll_done(&mut self) -> io::Result<bool>;
    /// Destroy the QP before any window is released.
    fn close(&mut self) -> io::Result<()>;
}

/// Bytes and windows moved, by carrier, for the metrics.
#[derive(Default)]
struct Counters {
    windows: [AtomicU64; 2],
    bytes: [AtomicU64; 2],
}

impl Counters {
    fn add(&self, carrier: usize, bytes: u64) {
        self.windows[carrier].fetch_add(1, Ordering::Relaxed);
        self.bytes[carrier].fetch_add(bytes, Ordering::Relaxed);
    }
}

/// What the hub has moved: `[stream, rdma]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Moved {
    pub windows: [u64; 2],
    pub bytes: [u64; 2],
}

struct Inner {
    rdma: Option<Arc<dyn Rdma>>,
    counters: Counters,
    address: String,
    regions: Arc<LocalRegions>,
    acceptor: Option<TlsAcceptor>,
    max_pages: u32,
    events: mpsc::UnboundedSender<Event>,
    state: Mutex<State>,
    next: AtomicU64,
    handle: Handle,
    /// Tests hold back `Closed` to play a handler whose exit cannot be observed.
    #[cfg(test)]
    closed_delay: Mutex<Duration>,
}

/// The listener's state and the transfers it serves. Cheap to clone.
#[derive(Clone)]
pub struct Hub {
    inner: Arc<Inner>,
    closing: watch::Sender<bool>,
}

fn random_token() -> io::Result<[u8; TOKEN_BYTES]> {
    use std::io::Read;
    let mut token = [0u8; TOKEN_BYTES];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut token)?;
    Ok(token)
}

impl Hub {
    /// Serve the data channel on `listener`. `max_pages` bounds the checksum
    /// table a window may announce. Without `tls` the connection is plaintext.
    pub fn start(
        handle: &Handle,
        listener: std::net::TcpListener,
        regions: Arc<LocalRegions>,
        tls: Option<Arc<rustls::ServerConfig>>,
        max_pages: u32,
        rdma: Option<Arc<dyn Rdma>>,
    ) -> io::Result<(Hub, mpsc::UnboundedReceiver<Event>)> {
        if rdma.is_some() && tls.is_some() {
            return Err(invalid(
                "RDMA transfers are not encrypted, so they cannot be offered with TLS",
            ));
        }
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?.to_string();
        let (events, receiver) = mpsc::unbounded_channel();
        let (closing, closed) = watch::channel(false);
        let inner = Arc::new(Inner {
            rdma,
            counters: Counters::default(),
            address,
            regions,
            acceptor: tls.map(TlsAcceptor::from),
            max_pages,
            events,
            state: Mutex::new(State {
                open: true,
                ..State::default()
            }),
            next: AtomicU64::new(1),
            handle: handle.clone(),
            #[cfg(test)]
            closed_delay: Mutex::new(Duration::ZERO),
        });
        let accepting = inner.clone();
        let listener = {
            let _guard = handle.enter();
            TcpListener::from_std(listener)?
        };
        handle.spawn(accept_loop(accepting, listener, closed));
        Ok((Hub { inner, closing }, receiver))
    }

    /// The `host:port` clients are told to connect to. A host that is the
    /// unspecified address means "the host you reached me by".
    pub fn address(&self) -> &str {
        &self.inner.address
    }

    pub fn rdma_transfer(&self, token: &[u8]) -> bool {
        let state = self.inner.state.lock().unwrap();
        self.offers_rdma()
            && state
                .tokens
                .get(token)
                .and_then(|id| state.transfers.get(id))
                .is_some_and(|entry| {
                    entry.kind == Kind::Get
                        && entry
                            .slicing
                            .as_ref()
                            .is_none_or(|(layout, _)| layout.world() == 1)
                })
    }

    /// Whether instances may ask for the RDMA carrier.
    pub fn offers_rdma(&self) -> bool {
        self.inner.rdma.is_some()
    }

    /// What has been moved so far, by carrier.
    pub fn moved(&self) -> Moved {
        let load = |counters: &[AtomicU64; 2]| {
            [
                counters[0].load(Ordering::Relaxed),
                counters[1].load(Ordering::Relaxed),
            ]
        };
        Moved {
            windows: load(&self.inner.counters.windows),
            bytes: load(&self.inner.counters.bytes),
        }
    }

    /// Whether connections are encrypted.
    #[cfg(test)]
    fn encrypted(&self) -> bool {
        self.inner.acceptor.is_some()
    }

    /// Begin a transfer. Its token authenticates the one connection that serves it.
    pub fn open(&self, kind: Kind) -> io::Result<(u64, [u8; TOKEN_BYTES])> {
        self.open_with(kind, None)
    }

    /// Begin the leg of rank `rank` of a multi-process instance: only that rank's slices of
    /// the windows travel on it.
    pub fn open_rank(
        &self,
        kind: Kind,
        layout: Arc<SliceLayout>,
        rank: usize,
    ) -> io::Result<(u64, [u8; TOKEN_BYTES])> {
        if rank >= layout.world() {
            return Err(invalid("no such rank"));
        }
        self.open_with(kind, Some((layout, rank)))
    }

    fn open_with(
        &self,
        kind: Kind,
        slicing: Option<(Arc<SliceLayout>, usize)>,
    ) -> io::Result<(u64, [u8; TOKEN_BYTES])> {
        let token = random_token()?;
        let transfer = self.inner.next.fetch_add(1, Ordering::Relaxed);
        let (stopped, _) = watch::channel(false);
        let entry = Arc::new(Transfer {
            kind,
            slicing,
            token,
            queues: Mutex::new(Queues {
                windows: VecDeque::new(),
                grants: VecDeque::new(),
                finished: false,
                attached: false,
            }),
            wake: Notify::new(),
            stopped,
        });
        let mut state = self.inner.state.lock().unwrap();
        if !state.open {
            return Err(io::Error::other("the data channel is shut down"));
        }
        state.tokens.insert(token, transfer);
        state.transfers.insert(transfer, entry);
        Ok((transfer, token))
    }

    fn transfer(&self, transfer: u64) -> Option<Arc<Transfer>> {
        self.inner
            .state
            .lock()
            .unwrap()
            .transfers
            .get(&transfer)
            .cloned()
    }

    /// Queue a claim's filled window for the instance.
    pub fn push_window(&self, transfer: u64, window: WindowOut) {
        if let Some(entry) = self.transfer(transfer) {
            entry.queues.lock().unwrap().windows.push_back(window);
            entry.wake.notify_one();
        }
    }

    /// Queue a put's granted window for the instance to fill.
    pub fn push_grant(&self, transfer: u64, grant: GrantOut) {
        if let Some(entry) = self.transfer(transfer) {
            entry.queues.lock().unwrap().grants.push_back(grant);
            entry.wake.notify_one();
        }
    }

    /// No window follows the ones queued: the handler sends END after them.
    pub fn finish(&self, transfer: u64) {
        if let Some(entry) = self.transfer(transfer) {
            entry.queues.lock().unwrap().finished = true;
            entry.wake.notify_one();
        }
    }

    /// End the transfer now. Returns `true` when no handler ever ran or it has
    /// already exited, so nothing can touch the transfer's windows any more;
    /// otherwise `Closed` follows once the handler has dropped its connection.
    pub fn stop(&self, transfer: u64) -> bool {
        let mut state = self.inner.state.lock().unwrap();
        let Some(entry) = state.transfers.get(&transfer).cloned() else {
            return true;
        };
        // Taking the token and the stop flag in one critical section with the
        // handler's attach means a connection can never attach to a stopped transfer.
        state.tokens.remove(&entry.token);
        let _ = entry.stopped.send(true);
        let attached = entry.queues.lock().unwrap().attached;
        if !attached {
            state.transfers.remove(&transfer);
        }
        !attached
    }

    /// Forget a transfer whose handler has exited.
    pub fn forget(&self, transfer: u64) {
        self.inner.state.lock().unwrap().transfers.remove(&transfer);
    }

    /// Stop accepting and stop every transfer.
    pub fn shutdown(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.open = false;
        state.tokens.clear();
        for entry in state.transfers.values() {
            let _ = entry.stopped.send(true);
        }
        let _ = self.closing.send(true);
    }

    /// Hold back every later `Closed` by `delay`, as if the handler could not be seen to exit.
    #[cfg(test)]
    pub fn delay_closed(&self, delay: Duration) {
        *self.inner.closed_delay.lock().unwrap() = delay;
    }

    /// Transfers with a live handler, for tests and metrics.
    pub fn handlers(&self) -> usize {
        self.inner
            .state
            .lock()
            .unwrap()
            .transfers
            .values()
            .filter(|entry| entry.queues.lock().unwrap().attached)
            .count()
    }
}

async fn accept_loop(inner: Arc<Inner>, listener: TcpListener, mut closing: watch::Receiver<bool>) {
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    let inner = inner.clone();
                    let handle = inner.handle.clone();
                    handle.spawn(async move { serve(inner, stream).await });
                }
                Err(error) => {
                    // A transient accept failure (descriptor exhaustion) must not end the listener.
                    eprintln!("selene: warning: the data listener could not accept: {error}");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            },
            _ = closing.changed() => return,
        }
    }
}

pub(crate) trait Conn: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Conn for T {}

#[cfg(target_os = "linux")]
fn set_buffers(stream: &TcpStream) {
    let socket = socket2::SockRef::from(stream);
    let _ = socket.set_send_buffer_size(SOCKET_BUFFER as usize);
    let _ = socket.set_recv_buffer_size(SOCKET_BUFFER as usize);
}

#[cfg(not(target_os = "linux"))]
fn set_buffers(_stream: &TcpStream) {}

/// Why a handler stopped.
enum End {
    /// The actor stopped the transfer.
    Stopped,
    /// The transfer is complete.
    Done,
    Failed(String),
}

fn failure(error: impl std::fmt::Display) -> End {
    End::Failed(error.to_string())
}

async fn serve(inner: Arc<Inner>, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    set_buffers(&stream);
    // The handshake and the token share one budget.
    let accepted = timeout(TOKEN_TIMEOUT, async {
        let mut conn: Box<dyn Conn> = match &inner.acceptor {
            Some(acceptor) => Box::new(acceptor.accept(stream).await?),
            None => Box::new(stream),
        };
        let mut token = [0u8; TOKEN_BYTES];
        conn.read_exact(&mut token).await?;
        Ok::<_, io::Error>((conn, token))
    })
    .await;
    let Ok(Ok((mut conn, token))) = accepted else {
        return;
    };
    // Look the token up and attach in one critical section with `stop`, so a
    // stopped transfer (its token already removed) can never gain a handler.
    // A token is good for one connection, whatever becomes of it.
    let (transfer, entry) = {
        let mut state = inner.state.lock().unwrap();
        let Some(transfer) = state.tokens.remove(&token) else {
            return;
        };
        let Some(entry) = state.transfers.get(&transfer).cloned() else {
            return;
        };
        if *entry.stopped.borrow() {
            return;
        }
        entry.queues.lock().unwrap().attached = true;
        (transfer, entry)
    };
    let mut stopped = entry.stopped.subscribe();
    // An RDMA transfer's channel lives here, outside the handler, so it can be fenced
    // properly whichever way the handler ends.
    let mut rdma: Option<Box<dyn Pair>> = None;
    let result = tokio::select! {
        result = run(&inner, transfer, &entry, &mut conn, &mut rdma) => result,
        _ = async { let _ = stopped.wait_for(|stop| *stop).await; } => End::Stopped,
    };
    // Fence the queue pair: it moves to the error state, its requests are flushed and it is
    // destroyed, on a thread that may block for the device.
    if let Some(channel) = rdma.take() {
        let fenced = tokio::task::spawn_blocking(move || close_rdma(channel)).await;
        if !matches!(fenced, Ok(Ok(()))) {
            let reason = "RDMA teardown failed; transfer quarantined".to_string();
            eprintln!("warning: {reason}");
            let _ = inner.events.send(Event::Failed { transfer, reason });
            // Keep `attached` set and withhold Closed: the store must retain its windows.
            // The failed channel retains the registration and mappings too.
            return;
        }
    }
    // Say goodbye if the peer will listen (a TLS peer needs the close notice to tell an
    // end from a cut), then drop the connection: only now is the transfer's memory quiet.
    let _ = timeout(GOODBYE, conn.shutdown()).await;
    drop(conn);
    #[cfg(test)]
    {
        let delay = *inner.closed_delay.lock().unwrap();
        tokio::time::sleep(delay).await;
    }
    entry.queues.lock().unwrap().attached = false;
    if let End::Failed(reason) = result {
        let _ = inner.events.send(Event::Failed { transfer, reason });
    }
    let _ = inner.events.send(Event::Closed { transfer });
}

fn close_rdma(mut channel: Box<dyn Pair>) -> io::Result<()> {
    match channel.close() {
        Ok(()) => Ok(()),
        Err(error) => {
            std::mem::forget(channel);
            Err(error)
        }
    }
}

async fn run(
    inner: &Inner,
    transfer: u64,
    entry: &Transfer,
    conn: &mut Box<dyn Conn>,
    rdma: &mut Option<Box<dyn Pair>>,
) -> End {
    let hello = match timeout(IO_TIMEOUT, ToManager::read(&mut *conn, inner.max_pages)).await {
        Ok(Ok(frame)) => frame,
        Ok(Err(error)) => return failure(error),
        Err(_) => return End::Failed("the instance sent no hello".into()),
    };
    match hello {
        ToManager::Hello(Carrier::Stream) => {}
        ToManager::Hello(Carrier::Rdma(peer))
            if entry.kind == Kind::Get
                && entry
                    .slicing
                    .as_ref()
                    .is_none_or(|(layout, _)| layout.world() == 1) =>
        {
            match open_rdma(inner, conn, &peer).await {
                Ok(channel) => *rdma = Some(channel),
                Err(error) => {
                    // Say why before the connection closes, so the instance falls back.
                    let _ = ToInstance::Abort(error.to_string()).write(conn).await;
                    return failure(error);
                }
            }
        }
        ToManager::Hello(Carrier::Rdma(_)) => {
            let message = "RDMA carries only unsliced claim windows";
            let _ = ToInstance::Abort(message.into()).write(conn).await;
            return End::Failed(message.into());
        }
        _ => return End::Failed("the instance did not open with a hello".into()),
    }
    let result = match entry.kind {
        Kind::Get => send_windows(inner, transfer, entry, conn, rdma.as_mut()).await,
        Kind::Put => receive_windows(inner, transfer, entry, conn).await,
    };
    match result {
        Ok(()) => End::Done,
        Err(error) => failure(error),
    }
}

/// Make a queue pair for the instance's, answer with ours and bring it up.
async fn open_rdma(
    inner: &Inner,
    conn: &mut Box<dyn Conn>,
    peer: &[u8],
) -> io::Result<Box<dyn Pair>> {
    let rdma = inner
        .rdma
        .as_ref()
        .ok_or_else(|| invalid("this manager does not offer RDMA"))?;
    let mut pair = rdma.pair()?;
    let handshake = pair.handshake();
    within(IO_TIMEOUT, "the instance stopped reading", async {
        ToInstance::Accept(handshake).write(conn).await?;
        conn.flush().await
    })
    .await?;
    pair.connect(peer)?;
    Ok(pair)
}

async fn send_windows(
    inner: &Inner,
    transfer: u64,
    entry: &Transfer,
    conn: &mut Box<dyn Conn>,
    mut rdma: Option<&mut Box<dyn Pair>>,
) -> io::Result<()> {
    loop {
        let next = {
            let mut queues = entry.queues.lock().unwrap();
            match queues.windows.pop_front() {
                Some(window) => Some(Some(window)),
                None if queues.finished => Some(None),
                None => None,
            }
        };
        let Some(next) = next else {
            entry.wake.notified().await;
            continue;
        };
        let Some(window) = next else {
            within(IO_TIMEOUT, "the instance stopped reading", async {
                ToInstance::End.write(conn).await?;
                conn.flush().await
            })
            .await?;
            let _ = timeout(GOODBYE, conn.shutdown()).await;
            return Ok(());
        };
        let slot = if let Some(channel) = rdma.as_mut() {
            let ToManager::Credit { offset, length } = within(
                IO_TIMEOUT,
                "the instance lent no slot",
                ToManager::read(conn, inner.max_pages),
            )
            .await?
            else {
                return Err(invalid("a claim's instance sends only credits"));
            };
            if window.shape.length > length || !offset.is_multiple_of(4096) {
                return Err(invalid("a window does not fit the slot the instance lent"));
            }
            channel.write(window.region, window.offset, window.shape.length, offset)?;
            rdma_done(channel).await?;
            offset
        } else {
            NO_SLOT
        };
        // SAFETY: the handler owns this window until `Closed`; the slice is only read.
        let packed;
        let payload: &[u8] = if slot != NO_SLOT {
            &[]
        } else {
            match &entry.slicing {
                None => unsafe {
                    inner
                        .regions
                        .bytes(window.region, window.offset, window.shape.length)?
                },
                Some((layout, rank)) => {
                    let (pages, row) = (window.shape.page_count, window.shape.row);
                    if window.shape.length != layout.slice_bytes(*rank, pages, row) {
                        return Err(invalid("a rank window's length is not its slice's"));
                    }
                    // SAFETY: as above, over the whole window the slice lies in.
                    let whole = unsafe {
                        inner.regions.bytes(
                            window.region,
                            window.offset,
                            layout.window_bytes(pages, row),
                        )?
                    };
                    let mut bytes = vec![0u8; window.shape.length as usize];
                    layout
                        .gather(*rank, whole, pages, row, &mut bytes)
                        .map_err(|error| invalid(error.to_string()))?;
                    packed = bytes;
                    &packed
                }
            }
        };
        let frame = ToInstance::Window(Window {
            shape: window.shape,
            page_xxh3: window.page_xxh3,
            row_xxh3: window.row_xxh3,
            slot,
        });
        within(IO_TIMEOUT, "the instance stopped reading", async {
            frame.write(conn).await?;
            conn.write_all(payload).await?;
            conn.flush().await
        })
        .await?;
        inner
            .counters
            .add(usize::from(rdma.is_some()), window.shape.length);
        let _ = inner.events.send(Event::Sent {
            transfer,
            index: window.shape.index,
        });
    }
}

/// How long one RDMA window may take to complete.
const RDMA_WINDOW: Duration = Duration::from_secs(10);

/// Poll the single write: a few spins on the fast path, then short sleeps.
async fn rdma_done(channel: &mut Box<dyn Pair>) -> io::Result<()> {
    let deadline = tokio::time::Instant::now() + RDMA_WINDOW;
    let mut spins = 0u32;
    loop {
        if channel.poll_done()? {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "an RDMA window did not complete",
            ));
        }
        spins += 1;
        if spins < 200 {
            tokio::task::yield_now().await;
        } else {
            tokio::time::sleep(Duration::from_micros(200)).await;
        }
    }
}

async fn receive_windows(
    inner: &Inner,
    transfer: u64,
    entry: &Transfer,
    conn: &mut Box<dyn Conn>,
) -> io::Result<()> {
    let (mut reader, mut writer) = tokio::io::split(conn);
    let (expect, mut expected) = mpsc::unbounded_channel::<GrantOut>();
    let writing = async {
        loop {
            let grants: Vec<GrantOut> = entry.queues.lock().unwrap().grants.drain(..).collect();
            if grants.is_empty() {
                entry.wake.notified().await;
                continue;
            }
            for grant in grants {
                // Tell the reader first: the instance may answer before this write returns.
                let _ = expect.send(grant);
                within(IO_TIMEOUT, "the instance stopped reading", async {
                    ToInstance::Grant(grant.shape).write(&mut writer).await?;
                    writer.flush().await
                })
                .await?;
            }
        }
    };
    let reading = async {
        while let Some(grant) = expected.recv().await {
            let frame = within(
                IO_TIMEOUT,
                "the instance sent nothing",
                ToManager::read(&mut reader, inner.max_pages),
            )
            .await?;
            let ToManager::Filled {
                index,
                row_xxh3,
                page_xxh3,
            } = frame
            else {
                return Err(invalid("the instance sent a frame a put does not take"));
            };
            if index != grant.shape.index
                || page_xxh3.len() != grant.shape.page_count as usize
                || row_xxh3.is_some() != grant.shape.row
            {
                return Err(invalid("a filled window does not match its grant"));
            }
            // SAFETY: the handler owns this window until `Closed`; the store leased this window to the put.
            match &entry.slicing {
                None => {
                    let payload = unsafe {
                        inner
                            .regions
                            .bytes(grant.region, grant.offset, grant.shape.length)?
                    };
                    within(
                        IO_TIMEOUT,
                        "the instance stopped writing",
                        reader.read_exact(payload),
                    )
                    .await?;
                }
                Some((layout, rank)) => {
                    let (pages, row) = (grant.shape.page_count, grant.shape.row);
                    if grant.shape.length != layout.slice_bytes(*rank, pages, row) {
                        return Err(invalid("a rank window's length is not its slice's"));
                    }
                    let mut packed = vec![0u8; grant.shape.length as usize];
                    within(
                        IO_TIMEOUT,
                        "the instance stopped writing",
                        reader.read_exact(&mut packed),
                    )
                    .await?;
                    // SAFETY: as above, over the whole window the slice lies in.
                    let whole = unsafe {
                        inner.regions.bytes(
                            grant.region,
                            grant.offset,
                            layout.window_bytes(pages, row),
                        )?
                    };
                    layout
                        .scatter(*rank, whole, pages, row, &packed)
                        .map_err(|error| invalid(error.to_string()))?;
                }
            }
            inner.counters.add(0, grant.shape.length);
            let _ = inner.events.send(Event::Filled {
                transfer,
                index,
                row_xxh3,
                page_xxh3,
            });
        }
        Ok(())
    };
    // Either direction ending, normally or not, ends the handler.
    tokio::select! {
        result = writing => result,
        result = reading => result,
    }
}

/// Parse the manager's certificate chain and key into a TLS server configuration.
pub fn server_config(
    certificate_pem: &[u8],
    key_pem: &[u8],
) -> io::Result<Arc<rustls::ServerConfig>> {
    use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};
    let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(certificate_pem)
        .collect::<Result<_, _>>()
        .map_err(|error| invalid(format!("the TLS certificate: {error}")))?;
    if chain.is_empty() {
        return Err(invalid("the TLS certificate file holds no certificate"));
    }
    let key = PrivateKeyDer::from_pem_slice(key_pem)
        .map_err(|error| invalid(format!("the TLS key: {error}")))?;
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|error| invalid(format!("TLS: {error}")))?
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .map_err(|error| invalid(format!("TLS: {error}")))?;
    config.alpn_protocols = vec![b"h2".to_vec()];
    Ok(Arc::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use flare_selene::shared::frame::TOKEN_BYTES;
    use std::net::TcpListener as StdListener;

    const PAGE: usize = 4096;

    /// Runs a test's future on its own runtime and fails with a message, instead of hanging,
    /// if it has not finished in [`TEST_LIMIT`]: every wait in these tests is for a peer that
    /// is in the same test, so a stuck one is a bug to report, never a reason to wait.
    const TEST_LIMIT: Duration = Duration::from_secs(60);

    fn bounded(test: impl std::future::Future<Output = ()>) {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let name = std::thread::current()
            .name()
            .unwrap_or("a test")
            .to_string();
        runtime.block_on(async {
            if timeout(TEST_LIMIT, test).await.is_err() {
                panic!("{name} did not finish within {TEST_LIMIT:?}: a connection or event it waits for never came");
            }
        });
        // A stuck background task must not keep the process either.
        runtime.shutdown_timeout(Duration::from_secs(5));
    }

    fn regions() -> Arc<LocalRegions> {
        Arc::new(LocalRegions::new(
            Arena::new(1 << 20).unwrap(),
            Arena::new(1 << 20).unwrap(),
        ))
    }

    fn start(
        regions: Arc<LocalRegions>,
        tls: Option<Arc<rustls::ServerConfig>>,
    ) -> (Hub, mpsc::UnboundedReceiver<Event>, String) {
        let listener = StdListener::bind("127.0.0.1:0").unwrap();
        let (hub, events) =
            Hub::start(&Handle::current(), listener, regions, tls, 64, None).unwrap();
        let address = hub.address().to_string();
        (hub, events, address)
    }

    fn fill(regions: &LocalRegions, region: Region, offset: u64, bytes: &[u8]) {
        // SAFETY: the test owns this window for the borrow.
        unsafe {
            let dst = regions.bytes(region, offset, bytes.len() as u64).unwrap();
            dst.copy_from_slice(bytes);
        }
    }

    fn shape(index: u32, pages: u32, row: bool) -> Shape {
        Shape {
            index,
            record: 0,
            first_page: 0,
            page_count: pages,
            row,
            length: u64::from(pages) * PAGE as u64 + if row { PAGE as u64 } else { 0 },
        }
    }

    async fn connect(address: &str, token: &[u8; TOKEN_BYTES]) -> Box<dyn Conn> {
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream.write_all(token).await.unwrap();
        ToManager::Hello(Carrier::Stream)
            .write(&mut stream)
            .await
            .unwrap();
        Box::new(stream)
    }

    async fn next(events: &mut mpsc::UnboundedReceiver<Event>) -> Event {
        timeout(Duration::from_secs(5), events.recv())
            .await
            .expect("an event in time")
            .expect("the hub is alive")
    }

    #[test]
    fn a_claims_windows_arrive_in_order_with_their_checksums_and_end() {
        bounded(async {
            let regions = regions();
            let (hub, mut events, address) = start(regions.clone(), None);
            let (transfer, token) = hub.open(Kind::Get).unwrap();
            let first: Vec<u8> = (0..2 * PAGE + PAGE).map(|i| (i % 251) as u8).collect();
            let second = vec![7u8; PAGE];
            fill(&regions, Region::Arena, 0, &first);
            fill(&regions, Region::Staging, 8192, &second);
            let mut conn = connect(&address, &token).await;
            hub.push_window(
                transfer,
                WindowOut {
                    shape: shape(0, 2, true),
                    page_xxh3: vec![11, 12],
                    row_xxh3: 13,
                    region: Region::Arena,
                    offset: 0,
                },
            );
            hub.push_window(
                transfer,
                WindowOut {
                    shape: shape(1, 1, false),
                    page_xxh3: vec![21],
                    row_xxh3: 0,
                    region: Region::Staging,
                    offset: 8192,
                },
            );
            hub.finish(transfer);
            let mut payload = vec![0u8; first.len()];
            let ToInstance::Window(window) = ToInstance::read(&mut conn, 64).await.unwrap() else {
                panic!("expected a window");
            };
            assert_eq!(window.shape, shape(0, 2, true));
            assert_eq!(
                (window.page_xxh3.clone(), window.row_xxh3),
                (vec![11, 12], 13)
            );
            conn.read_exact(&mut payload).await.unwrap();
            assert_eq!(payload, first);
            let ToInstance::Window(window) = ToInstance::read(&mut conn, 64).await.unwrap() else {
                panic!("expected a second window");
            };
            assert_eq!(window.shape.index, 1);
            let mut payload = vec![0u8; PAGE];
            conn.read_exact(&mut payload).await.unwrap();
            assert_eq!(payload, second);
            assert_eq!(
                ToInstance::read(&mut conn, 64).await.unwrap(),
                ToInstance::End
            );
            // Sent twice, then the handler leaves on its own.
            assert!(matches!(
                next(&mut events).await,
                Event::Sent { index: 0, .. }
            ));
            assert!(matches!(
                next(&mut events).await,
                Event::Sent { index: 1, .. }
            ));
            assert!(matches!(next(&mut events).await, Event::Closed { .. }));
            assert!(
                hub.stop(transfer),
                "an exited handler leaves nothing to wait for"
            );
        })
    }

    #[test]
    fn a_token_serves_one_connection_and_a_wrong_token_none() {
        bounded(async {
            let (hub, mut events, address) = start(regions(), None);
            let (transfer, token) = hub.open(Kind::Get).unwrap();
            let _first = connect(&address, &token).await;
            // Wait until the first connection has been attached, or the second could win the race.
            while hub.handlers() == 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            // The token is spent: a second connection is closed without a byte.
            let mut second = connect(&address, &token).await;
            assert!(ToInstance::read(&mut second, 64).await.is_err());
            let mut wrong = connect(&address, &[9; TOKEN_BYTES]).await;
            assert!(ToInstance::read(&mut wrong, 64).await.is_err());
            assert!(!hub.stop(transfer), "a handler is attached");
            assert!(matches!(next(&mut events).await, Event::Closed { .. }));
        })
    }

    #[test]
    fn stopping_before_any_connection_needs_no_wait_and_blocks_a_late_one() {
        bounded(async {
            let (hub, _events, address) = start(regions(), None);
            let (transfer, token) = hub.open(Kind::Get).unwrap();
            assert!(hub.stop(transfer), "no handler ever ran");
            let mut late = connect(&address, &token).await;
            assert!(ToInstance::read(&mut late, 64).await.is_err());
            assert_eq!(hub.handlers(), 0);
        })
    }

    #[test]
    fn stopping_a_stalled_transfer_ends_its_handler_and_reports_closed() {
        bounded(async {
            let regions = regions();
            let (hub, mut events, address) = start(regions.clone(), None);
            let (transfer, token) = hub.open(Kind::Get).unwrap();
            // The instance connects and never reads: the socket buffers fill.
            let _conn = connect(&address, &token).await;
            fill(&regions, Region::Arena, 0, &vec![1u8; 1 << 20]);
            for index in 0..64 {
                hub.push_window(
                    transfer,
                    WindowOut {
                        shape: Shape {
                            length: 1 << 20,
                            page_count: 256,
                            ..shape(index, 0, false)
                        },
                        page_xxh3: vec![0; 256],
                        row_xxh3: 0,
                        region: Region::Arena,
                        offset: 0,
                    },
                );
            }
            // Wait until the handler is stuck in a write (some windows went, not all).
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert_eq!(hub.handlers(), 1);
            assert!(!hub.stop(transfer));
            loop {
                match next(&mut events).await {
                    Event::Closed { transfer: closed } => {
                        assert_eq!(closed, transfer);
                        break;
                    }
                    Event::Sent { .. } => {}
                    other => panic!("unexpected {other:?}"),
                }
            }
            assert_eq!(hub.handlers(), 0);
        })
    }

    #[test]
    fn a_peer_that_drops_mid_transfer_fails_it_and_closes_the_handler() {
        bounded(async {
            let regions = regions();
            let (hub, mut events, address) = start(regions.clone(), None);
            let (transfer, token) = hub.open(Kind::Get).unwrap();
            let conn = connect(&address, &token).await;
            drop(conn);
            fill(&regions, Region::Arena, 0, &vec![1u8; 1 << 20]);
            for index in 0..32 {
                hub.push_window(
                    transfer,
                    WindowOut {
                        shape: Shape {
                            length: 1 << 20,
                            page_count: 256,
                            ..shape(index, 0, false)
                        },
                        page_xxh3: vec![0; 256],
                        row_xxh3: 0,
                        region: Region::Arena,
                        offset: 0,
                    },
                );
            }
            let mut failed = false;
            loop {
                match next(&mut events).await {
                    Event::Failed { .. } => failed = true,
                    Event::Closed { .. } => break,
                    Event::Sent { .. } => {}
                    other => panic!("unexpected {other:?}"),
                }
            }
            assert!(failed, "the broken connection is reported");
        })
    }

    #[test]
    fn a_puts_windows_are_filled_from_what_the_instance_writes() {
        bounded(async {
            let regions = regions();
            let (hub, mut events, address) = start(regions.clone(), None);
            let (transfer, token) = hub.open(Kind::Put).unwrap();
            let mut conn = connect(&address, &token).await;
            hub.push_grant(
                transfer,
                GrantOut {
                    shape: shape(0, 2, true),
                    region: Region::Staging,
                    offset: 4096,
                },
            );
            let ToInstance::Grant(granted) = ToInstance::read(&mut conn, 64).await.unwrap() else {
                panic!("expected a grant");
            };
            assert_eq!(granted, shape(0, 2, true));
            let bytes: Vec<u8> = (0..3 * PAGE).map(|i| (i % 13) as u8).collect();
            ToManager::Filled {
                index: 0,
                row_xxh3: Some(3),
                page_xxh3: vec![1, 2],
            }
            .write(&mut conn)
            .await
            .unwrap();
            conn.write_all(&bytes).await.unwrap();
            let Event::Filled {
                index,
                row_xxh3,
                page_xxh3,
                ..
            } = next(&mut events).await
            else {
                panic!("expected Filled");
            };
            assert_eq!((index, row_xxh3, page_xxh3), (0, Some(3), vec![1, 2]));
            // SAFETY: the test owns this window for the borrow.
            unsafe {
                let src = regions
                    .bytes(Region::Staging, 4096, bytes.len() as u64)
                    .unwrap();
                assert_eq!(src, &bytes[..]);
            }
            // A second grant is served the same way, after the first.
            hub.push_grant(
                transfer,
                GrantOut {
                    shape: shape(1, 1, false),
                    region: Region::Staging,
                    offset: 65536,
                },
            );
            assert!(matches!(
                ToInstance::read(&mut conn, 64).await.unwrap(),
                ToInstance::Grant(_)
            ));
            hub.stop(transfer);
            assert!(matches!(next(&mut events).await, Event::Closed { .. }));
        })
    }

    fn rank_layout(world: usize) -> Arc<SliceLayout> {
        use flare_selene::shared::{Geometry, RankPlanes};
        let rank = RankPlanes {
            plane_bytes: vec![4096, 4096],
            row_bytes: 4096,
        };
        Arc::new(
            SliceLayout::new(&Geometry {
                page_tokens: 64,
                ranks: vec![rank; world],
                page_record_bytes: world as u64 * 8192,
                row_record_bytes: world as u64 * 4096,
            })
            .unwrap(),
        )
    }

    fn rank_shape(layout: &SliceLayout, rank: usize, index: u32, pages: u32, row: bool) -> Shape {
        Shape {
            length: layout.slice_bytes(rank, pages, row),
            ..shape(index, pages, row)
        }
    }

    #[test]
    fn a_ranks_leg_carries_only_its_slices_of_each_window_and_other_ranks_see_theirs() {
        bounded(async {
            let regions = regions();
            let (hub, mut events, address) = start(regions.clone(), None);
            let layout = rank_layout(3);
            let window: Vec<u8> = (0..layout.window_bytes(2, true) as usize)
                .map(|i| (i % 241) as u8)
                .collect();
            fill(&regions, Region::Staging, 8192, &window);
            let mut transfers = Vec::new();
            for rank in 0..3 {
                let (transfer, token) = hub.open_rank(Kind::Get, layout.clone(), rank).unwrap();
                let conn = connect(&address, &token).await;
                hub.push_window(
                    transfer,
                    WindowOut {
                        shape: rank_shape(&layout, rank, 0, 2, true),
                        page_xxh3: vec![1, 2],
                        row_xxh3: 3,
                        region: Region::Staging,
                        offset: 8192,
                    },
                );
                hub.finish(transfer);
                transfers.push((transfer, conn));
            }
            for (rank, (_, conn)) in transfers.iter_mut().enumerate() {
                let ToInstance::Window(got) = ToInstance::read(conn, 64).await.unwrap() else {
                    panic!("expected a window");
                };
                assert_eq!(got.shape, rank_shape(&layout, rank, 0, 2, true));
                let mut payload = vec![0u8; got.shape.length as usize];
                conn.read_exact(&mut payload).await.unwrap();
                let mut expect = vec![0u8; payload.len()];
                layout.gather(rank, &window, 2, true, &mut expect).unwrap();
                assert_eq!(payload, expect, "rank {rank}");
                assert_eq!(ToInstance::read(conn, 64).await.unwrap(), ToInstance::End);
            }
            let mut sent = 0;
            let mut closed = 0;
            while sent < 3 || closed < 3 {
                match next(&mut events).await {
                    Event::Sent { .. } => sent += 1,
                    Event::Closed { .. } => closed += 1,
                    other => panic!("unexpected {other:?}"),
                }
            }
        })
    }

    #[test]
    fn the_ranks_legs_of_a_put_assemble_one_window_and_a_wrong_length_ends_the_leg() {
        bounded(async {
            let regions = regions();
            let (hub, mut events, address) = start(regions.clone(), None);
            let layout = rank_layout(2);
            let truth: Vec<u8> = (0..layout.window_bytes(2, true) as usize)
                .map(|i| (i % 233) as u8)
                .collect();
            for rank in [1usize, 0] {
                let (transfer, token) = hub.open_rank(Kind::Put, layout.clone(), rank).unwrap();
                let mut conn = connect(&address, &token).await;
                hub.push_grant(
                    transfer,
                    GrantOut {
                        shape: rank_shape(&layout, rank, 0, 2, true),
                        region: Region::Staging,
                        offset: 16384,
                    },
                );
                let ToInstance::Grant(granted) = ToInstance::read(&mut conn, 64).await.unwrap()
                else {
                    panic!("expected a grant");
                };
                assert_eq!(granted, rank_shape(&layout, rank, 0, 2, true));
                let mut packed = vec![0u8; granted.length as usize];
                layout.gather(rank, &truth, 2, true, &mut packed).unwrap();
                ToManager::Filled {
                    index: 0,
                    row_xxh3: Some(3),
                    page_xxh3: vec![1, 2],
                }
                .write(&mut conn)
                .await
                .unwrap();
                conn.write_all(&packed).await.unwrap();
                let Event::Filled { index, .. } = next(&mut events).await else {
                    panic!("expected Filled");
                };
                assert_eq!(index, 0);
                hub.stop(transfer);
                assert!(matches!(next(&mut events).await, Event::Closed { .. }));
            }
            // SAFETY: the test owns this window for the borrow.
            unsafe {
                let window = regions
                    .bytes(Region::Staging, 16384, truth.len() as u64)
                    .unwrap();
                assert_eq!(window, &truth[..], "both ranks' slices are in place");
            }
            // A grant whose shape is not the rank's slice ends the leg.
            let (transfer, token) = hub.open_rank(Kind::Put, layout.clone(), 0).unwrap();
            let mut conn = connect(&address, &token).await;
            hub.push_grant(
                transfer,
                GrantOut {
                    shape: shape(0, 2, true),
                    region: Region::Staging,
                    offset: 16384,
                },
            );
            let ToInstance::Grant(_) = ToInstance::read(&mut conn, 64).await.unwrap() else {
                panic!("expected a grant");
            };
            ToManager::Filled {
                index: 0,
                row_xxh3: Some(3),
                page_xxh3: vec![1, 2],
            }
            .write(&mut conn)
            .await
            .unwrap();
            conn.write_all(&vec![0u8; 3 * PAGE]).await.unwrap();
            let mut failed = false;
            loop {
                match next(&mut events).await {
                    Event::Failed { .. } => failed = true,
                    Event::Closed { .. } => break,
                    other => panic!("unexpected {other:?}"),
                }
            }
            assert!(failed);
        })
    }

    #[test]
    fn a_filled_window_that_disagrees_with_its_grant_fails_the_put_without_writing() {
        bounded(async {
            let regions = regions();
            let (hub, mut events, address) = start(regions.clone(), None);
            let (transfer, token) = hub.open(Kind::Put).unwrap();
            let mut conn = connect(&address, &token).await;
            hub.push_grant(
                transfer,
                GrantOut {
                    shape: shape(0, 2, false),
                    region: Region::Staging,
                    offset: 0,
                },
            );
            let _ = ToInstance::read(&mut conn, 64).await.unwrap();
            // One checksum for a two-page window.
            ToManager::Filled {
                index: 0,
                row_xxh3: None,
                page_xxh3: vec![1],
            }
            .write(&mut conn)
            .await
            .unwrap();
            conn.write_all(&vec![0xee; 2 * PAGE]).await.unwrap();
            loop {
                match next(&mut events).await {
                    Event::Failed { .. } => {}
                    Event::Closed { .. } => break,
                    other => panic!("unexpected {other:?}"),
                }
            }
            // SAFETY: the test owns this window for the borrow.
            unsafe {
                let src = regions
                    .bytes(Region::Staging, 0, (2 * PAGE) as u64)
                    .unwrap();
                assert!(src.iter().all(|byte| *byte == 0), "nothing was written");
            }
        })
    }

    fn certificate() -> (Arc<rustls::ServerConfig>, rustls::RootCertStore) {
        use rustls::pki_types::{CertificateDer, pem::PemObject};
        let generated = crate::testing::certificate();
        let pem = generated.cert.pem();
        let config = server_config(
            pem.as_bytes(),
            generated.signing_key.serialize_pem().as_bytes(),
        )
        .unwrap();
        let mut roots = rustls::RootCertStore::empty();
        for der in CertificateDer::pem_slice_iter(pem.as_bytes()) {
            roots.add(der.unwrap()).unwrap();
        }
        (config, roots)
    }

    #[test]
    fn a_tls_hub_serves_a_client_that_trusts_it_and_refuses_one_that_does_not() {
        bounded(async {
            use rustls::pki_types::ServerName;
            let regions = regions();
            let (config, roots) = certificate();
            let (hub, mut events, address) = start(regions.clone(), Some(config));
            assert!(hub.encrypted());
            let (transfer, token) = hub.open(Kind::Get).unwrap();
            fill(&regions, Region::Arena, 0, &vec![5u8; PAGE]);
            hub.push_window(
                transfer,
                WindowOut {
                    shape: shape(0, 1, false),
                    page_xxh3: vec![1],
                    row_xxh3: 0,
                    region: Region::Arena,
                    offset: 0,
                },
            );
            hub.finish(transfer);
            let provider = Arc::new(rustls::crypto::ring::default_provider());
            let client = |roots: rustls::RootCertStore| {
                let config = rustls::ClientConfig::builder_with_provider(provider.clone())
                    .with_safe_default_protocol_versions()
                    .unwrap()
                    .with_root_certificates(roots)
                    .with_no_client_auth();
                tokio_rustls::TlsConnector::from(Arc::new(config))
            };
            // Plaintext cannot consume the token on a TLS listener.
            let mut plain = TcpStream::connect(&address).await.unwrap();
            plain.write_all(&token).await.unwrap();
            ToManager::Hello(Carrier::Stream).write(&mut plain).await.unwrap();
            assert!(ToInstance::read(&mut plain, 64).await.is_err());
            // A client with some other root fails the handshake; the token stays good.
            let tcp = TcpStream::connect(&address).await.unwrap();
            let name = ServerName::try_from("localhost").unwrap();
            let (_, other_roots) = certificate();
            assert!(
                client(other_roots)
                    .connect(name.clone(), tcp)
                    .await
                    .is_err()
            );
            let tcp = TcpStream::connect(&address).await.unwrap();
            let mut conn = client(roots).connect(name, tcp).await.unwrap();
            conn.write_all(&token).await.unwrap();
            ToManager::Hello(Carrier::Stream)
                .write(&mut conn)
                .await
                .unwrap();
            let ToInstance::Window(window) = ToInstance::read(&mut conn, 64).await.unwrap() else {
                panic!("expected a window");
            };
            assert_eq!(window.page_xxh3, vec![1]);
            let mut payload = vec![0u8; PAGE];
            conn.read_exact(&mut payload).await.unwrap();
            assert!(payload.iter().all(|byte| *byte == 5));
            assert_eq!(
                ToInstance::read(&mut conn, 64).await.unwrap(),
                ToInstance::End
            );
            assert!(matches!(next(&mut events).await, Event::Sent { .. }));
        })
    }

    #[test]
    fn a_certificate_and_key_that_do_not_belong_together_are_refused() {
        let one = rcgen::generate_simple_self_signed(vec!["a".to_string()]).unwrap();
        let two = rcgen::generate_simple_self_signed(vec!["b".to_string()]).unwrap();
        assert!(
            server_config(
                one.cert.pem().as_bytes(),
                two.signing_key.serialize_pem().as_bytes()
            )
            .is_err()
        );
        assert!(server_config(b"", one.signing_key.serialize_pem().as_bytes()).is_err());
    }

    /// An RDMA WRITE device backed by ordinary memory for CPU tests.
    mod fake_rdma {
        use super::*;
        use crate::data::channel::{Pair, Rdma};

        pub struct Device {
            pub remote: Arc<Mutex<Vec<u8>>>,
            pub regions: Arc<LocalRegions>,
            pub fail_close: bool,
        }

        struct Queue {
            remote: Arc<Mutex<Vec<u8>>>,
            regions: Arc<LocalRegions>,
            done: bool,
            fail_close: bool,
        }

        impl Rdma for Device {
            fn pair(&self) -> io::Result<Box<dyn Pair>> {
                Ok(Box::new(Queue {
                    remote: self.remote.clone(),
                    regions: self.regions.clone(),
                    done: false,
                    fail_close: self.fail_close,
                }))
            }
        }

        impl Pair for Queue {
            fn handshake(&self) -> Vec<u8> {
                vec![7]
            }
            fn connect(&mut self, _: &[u8]) -> io::Result<()> {
                Ok(())
            }
            fn write(
                &mut self,
                region: Region,
                offset: u64,
                len: u64,
                remote_offset: u64,
            ) -> io::Result<()> {
                let remote = self.remote.clone();
                // SAFETY: the test owns this window for the borrow.
                unsafe {
                    let bytes = self.regions.bytes(region, offset, len)?;
                    let at = remote_offset as usize;
                    remote.lock().unwrap()[at..at + bytes.len()].copy_from_slice(bytes);
                }
                self.done = true;
                Ok(())
            }
            fn poll_done(&mut self) -> io::Result<bool> {
                Ok(std::mem::take(&mut self.done))
            }
            fn close(&mut self) -> io::Result<()> {
                if self.fail_close {
                    return Err(io::Error::other("destroy failed"));
                }
                Ok(())
            }
        }
    }

    /// A hub with the fake WRITE device.
    fn start_fake_rdma() -> (
        Hub,
        mpsc::UnboundedReceiver<Event>,
        String,
        Arc<LocalRegions>,
        Arena,
        Arc<Mutex<Vec<u8>>>,
    ) {
        start_fake_rdma_with_close_failure(false)
    }

    fn start_fake_rdma_with_close_failure(
        fail_close: bool,
    ) -> (
        Hub,
        mpsc::UnboundedReceiver<Event>,
        String,
        Arc<LocalRegions>,
        Arena,
        Arc<Mutex<Vec<u8>>>,
    ) {
        let staging = Arena::new(1 << 20).unwrap();
        let regions = Arc::new(LocalRegions::new(
            Arena::new(1 << 20).unwrap(),
            staging.clone(),
        ));
        let remote = Arc::new(Mutex::new(vec![0u8; 1 << 20]));
        let device = fake_rdma::Device {
            fail_close,
            remote: remote.clone(),
            regions: regions.clone(),
        };
        let listener = StdListener::bind("127.0.0.1:0").unwrap();
        let (hub, events) = Hub::start(
            &Handle::current(),
            listener,
            regions.clone(),
            None,
            64,
            Some(Arc::new(device)),
        )
        .unwrap();
        let address = hub.address().to_string();
        (hub, events, address, regions, staging, remote)
    }

    async fn frame(conn: &mut (impl AsyncRead + Unpin)) -> ToInstance {
        timeout(Duration::from_secs(5), ToInstance::read(conn, 64))
            .await
            .expect("a frame in time")
            .unwrap()
    }

    async fn dial_rdma(address: &str, token: &[u8; TOKEN_BYTES]) -> Box<dyn Conn> {
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream.write_all(token).await.unwrap();
        ToManager::Hello(Carrier::Rdma(vec![1]))
            .write(&mut stream)
            .await
            .unwrap();
        let ToInstance::Accept(_) = frame(&mut stream).await else {
            panic!("expected the manager's end of the pair");
        };
        Box::new(stream)
    }

    #[test]
    fn failed_rdma_teardown_withholds_closed_and_keeps_transfer_attached() {
        bounded(async {
            let (hub, mut events, address, regions, _, _) =
                start_fake_rdma_with_close_failure(true);
            let (transfer, token) = hub.open(Kind::Get).unwrap();
            let _conn = dial_rdma(&address, &token).await;
            hub.finish(transfer);
            assert!(
                matches!(next(&mut events).await, Event::Failed { transfer: id, .. } if id == transfer)
            );
            assert!(!hub.stop(transfer));
            assert!(events.try_recv().is_err());
            assert!(Arc::strong_count(&regions) > 2);
        });
    }

    #[test]
    fn writes_land_before_the_window_is_named_and_only_scalar_claims_offer_rdma() {
        bounded(async {
            let (hub, mut events, address, regions, _, remote) = start_fake_rdma();
            let (transfer, token) = hub.open(Kind::Get).unwrap();
            assert!(hub.rdma_transfer(&token));
            let (_, put_token) = hub.open(Kind::Put).unwrap();
            assert!(!hub.rdma_transfer(&put_token));
            let (_, rank_token) = hub.open_rank(Kind::Get, rank_layout(2), 1).unwrap();
            assert!(!hub.rdma_transfer(&rank_token));
            let mut conn = dial_rdma(&address, &token).await;
            ToManager::Credit {
                offset: 4096,
                length: PAGE as u64,
            }
            .write(&mut conn)
            .await
            .unwrap();
            fill(&regions, Region::Arena, 0, &vec![9; PAGE]);
            hub.push_window(
                transfer,
                WindowOut {
                    shape: shape(0, 1, false),
                    page_xxh3: vec![1],
                    row_xxh3: 0,
                    region: Region::Arena,
                    offset: 0,
                },
            );
            hub.finish(transfer);
            let ToInstance::Window(window) = frame(&mut conn).await else {
                panic!("window missing")
            };
            assert_eq!(window.slot, 4096);
            assert_eq!(&remote.lock().unwrap()[4096..4096 + PAGE], &vec![9; PAGE]);
            assert_eq!(frame(&mut conn).await, ToInstance::End);
            assert!(matches!(next(&mut events).await, Event::Sent { .. }));
            assert!(matches!(next(&mut events).await, Event::Closed { .. }));
        });
    }

    #[cfg(feature = "rdma")]
    mod over_rdma {
        use super::*;
        use flare_selene::pool::rdma;

        /// The instance's end, over the real device: staging, a registration and a pair.
        struct Instance {
            staging: Arena,
            device: Arc<rdma::Device>,
        }

        impl Instance {
            fn new() -> Self {
                let staging = Arena::new(1 << 20).unwrap();
                let (base, len) = staging.mapping();
                // SAFETY: the arena stays mapped and in place for the life of the instance.
                let device = unsafe { rdma::Device::open(base, len) }.unwrap();
                Self { staging, device }
            }

            #[allow(clippy::mut_from_ref)]
            fn bytes(&self, offset: usize, length: usize) -> &mut [u8] {
                let (base, len) = self.staging.mapping();
                assert!(offset + length <= len);
                // SAFETY: inside the mapping; the test drives it from one task.
                unsafe { std::slice::from_raw_parts_mut(base.add(offset), length) }
            }

            /// Connect, say hello for RDMA and bring the pair up.
            async fn dial(
                &self,
                address: &str,
                token: &[u8; TOKEN_BYTES],
            ) -> (TcpStream, rdma::Pair) {
                let mut stream = TcpStream::connect(address).await.unwrap();
                stream.write_all(token).await.unwrap();
                let pair = rdma::Pair::new(&self.device).unwrap();
                ToManager::Hello(Carrier::Rdma(pair.handshake().encode().to_vec()))
                    .write(&mut stream)
                    .await
                    .unwrap();
                let ToInstance::Accept(theirs) = ToInstance::read(&mut stream, 64).await.unwrap()
                else {
                    panic!("expected the manager's end of the pair");
                };
                pair.connect(&rdma::Handshake::decode(&theirs).unwrap())
                    .unwrap();
                (stream, pair)
            }
        }

        fn start_rdma(regions: Arc<LocalRegions>) -> (Hub, mpsc::UnboundedReceiver<Event>, String) {
            let device = crate::data::ibv::Device::open(regions.clone()).unwrap();
            let listener = StdListener::bind("127.0.0.1:0").unwrap();
            let (hub, events) = Hub::start(
                &Handle::current(),
                listener,
                regions,
                None,
                64,
                Some(Arc::new(device)),
            )
            .unwrap();
            let address = hub.address().to_string();
            (hub, events, address)
        }

        #[test]
        fn a_claims_windows_are_written_into_lent_slots_and_named() {
            bounded(async {
                let regions = regions();
                let (hub, mut events, address) = start_rdma(regions.clone());
                assert!(hub.offers_rdma());
                let instance = Instance::new();
                let (transfer, token) = hub.open(Kind::Get).unwrap();
                let (mut stream, pair) = instance.dial(&address, &token).await;
                // Two slots lent, and two windows: each lands in the slot lent for it.
                for offset in [0u64, 4 * PAGE as u64] {
                    ToManager::Credit {
                        offset,
                        length: 4 * PAGE as u64,
                    }
                    .write(&mut stream)
                    .await
                    .unwrap();
                }
                let first: Vec<u8> = (0..2 * PAGE).map(|i| (i % 253) as u8).collect();
                let second: Vec<u8> = (0..PAGE).map(|i| (i % 7 + 1) as u8).collect();
                fill(&regions, Region::Arena, 0, &first);
                fill(&regions, Region::Staging, 8192, &second);
                hub.push_window(
                    transfer,
                    WindowOut {
                        shape: shape(0, 2, false),
                        page_xxh3: vec![1, 2],
                        row_xxh3: 0,
                        region: Region::Arena,
                        offset: 0,
                    },
                );
                hub.push_window(
                    transfer,
                    WindowOut {
                        shape: shape(1, 1, false),
                        page_xxh3: vec![3],
                        row_xxh3: 0,
                        region: Region::Staging,
                        offset: 8192,
                    },
                );
                hub.finish(transfer);
                let ToInstance::Window(one) = ToInstance::read(&mut stream, 64).await.unwrap()
                else {
                    panic!("expected a window");
                };
                assert_eq!(
                    (one.shape.index, one.slot, one.page_xxh3.clone()),
                    (0, 0, vec![1, 2])
                );
                assert_eq!(instance.bytes(0, 2 * PAGE), &first[..]);
                let ToInstance::Window(two) = ToInstance::read(&mut stream, 64).await.unwrap()
                else {
                    panic!("expected a second window");
                };
                assert_eq!((two.shape.index, two.slot), (1, 4 * PAGE as u64));
                assert_eq!(instance.bytes(4 * PAGE, PAGE), &second[..]);
                assert_eq!(
                    ToInstance::read(&mut stream, 64).await.unwrap(),
                    ToInstance::End
                );
                assert!(matches!(
                    next(&mut events).await,
                    Event::Sent { index: 0, .. }
                ));
                assert!(matches!(
                    next(&mut events).await,
                    Event::Sent { index: 1, .. }
                ));
                assert!(matches!(next(&mut events).await, Event::Closed { .. }));
                let moved = hub.moved();
                assert_eq!(moved.windows, [0, 2]);
                assert_eq!(moved.bytes, [0, 3 * PAGE as u64]);
                drop(pair);
            })
        }

        #[test]
        fn a_window_that_does_not_fit_the_lent_slot_fails_the_transfer() {
            bounded(async {
                let regions = regions();
                let (hub, mut events, address) = start_rdma(regions.clone());
                let instance = Instance::new();
                let (transfer, token) = hub.open(Kind::Get).unwrap();
                let (mut stream, _pair) = instance.dial(&address, &token).await;
                ToManager::Credit {
                    offset: 0,
                    length: PAGE as u64,
                }
                .write(&mut stream)
                .await
                .unwrap();
                hub.push_window(
                    transfer,
                    WindowOut {
                        shape: shape(0, 2, false),
                        page_xxh3: vec![1, 2],
                        row_xxh3: 0,
                        region: Region::Arena,
                        offset: 0,
                    },
                );
                assert!(matches!(next(&mut events).await, Event::Failed { .. }));
                assert!(matches!(next(&mut events).await, Event::Closed { .. }));
                assert!(instance.bytes(0, PAGE).iter().all(|byte| *byte == 0));
            })
        }

        #[test]
        fn stopping_an_rdma_transfer_fences_its_pair_before_it_reports_closed() {
            bounded(async {
                let regions = regions();
                let (hub, mut events, address) = start_rdma(regions.clone());
                let instance = Instance::new();
                let (transfer, token) = hub.open(Kind::Get).unwrap();
                let (mut stream, pair) = instance.dial(&address, &token).await;
                ToManager::Credit {
                    offset: 0,
                    length: 4 * PAGE as u64,
                }
                .write(&mut stream)
                .await
                .unwrap();
                tokio::time::sleep(Duration::from_millis(100)).await;
                assert_eq!(hub.handlers(), 1);
                assert!(!hub.stop(transfer));
                assert!(matches!(next(&mut events).await, Event::Closed { .. }));
                assert_eq!(hub.handlers(), 0);
                drop(pair);
            })
        }

        #[test]
        fn a_manager_without_rdma_or_with_tls_refuses_an_rdma_hello() {
            bounded(async {
                let regions = regions();
                let (hub, mut events, address) = start(regions.clone(), None);
                let instance = Instance::new();
                let (_transfer, token) = hub.open(Kind::Get).unwrap();
                let mut stream = TcpStream::connect(&address).await.unwrap();
                stream.write_all(&token).await.unwrap();
                let pair = rdma::Pair::new(&instance.device).unwrap();
                ToManager::Hello(Carrier::Rdma(pair.handshake().encode().to_vec()))
                    .write(&mut stream)
                    .await
                    .unwrap();
                let ToInstance::Abort(why) = ToInstance::read(&mut stream, 64).await.unwrap()
                else {
                    panic!("expected an abort");
                };
                assert!(why.contains("does not offer RDMA"), "{why}");
                assert!(matches!(next(&mut events).await, Event::Failed { .. }));
                // TLS and RDMA do not mix at the hub either.
                let generated = crate::testing::certificate();
                let tls = server_config(
                    generated.cert.pem().as_bytes(),
                    generated.signing_key.serialize_pem().as_bytes(),
                )
                .unwrap();
                let device = crate::data::ibv::Device::open(regions.clone()).unwrap();
                let listener = StdListener::bind("127.0.0.1:0").unwrap();
                assert!(
                    Hub::start(
                        &Handle::current(),
                        listener,
                        regions,
                        Some(tls),
                        64,
                        Some(Arc::new(device)),
                    )
                    .is_err()
                );
            })
        }
    }
}
