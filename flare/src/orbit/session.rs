//! An encrypted, mutually authenticated session between two identities over TCP.
//!
//! Handshake (three messages):
//!
//! 1. Initiator: `FLAREFLT` magic, protocol version, ephemeral X25519 public key.
//! 2. Responder: its ephemeral X25519 public key, then one record encrypted under the
//!    responder-to-initiator key carrying its Ed25519 public key and a signature over the
//!    transcript so far.
//! 3. Initiator: one record under the initiator-to-responder key carrying its Ed25519 public key
//!    and a signature over the transcript including message 2.
//!
//! Session keys come from HKDF-SHA256 over the X25519 shared secret, salted with the hash of the
//! two ephemeral messages; each direction has its own ChaCha20-Poly1305 key and a counter nonce.
//! Each party signs the transcript, so a party that relays the connection cannot present either
//! identity. Identities travel only encrypted. The handshake hash, the SHA-256 of every
//! handshake byte, is the channel binding that pairing proofs include.
//!
//! Records are `u16` big-endian length plus ciphertext of at most 65535 bytes, so a record
//! carries at most 65519 plaintext bytes. A frame is a tag byte, a little-endian `u64` body
//! length, and the body; its header starts a record and the body spans as many records as it
//! needs, so a frame of many gigabytes streams without buffering.

use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream};
use std::time::Duration;

use ring::aead::{self, Aad, LessSafeKey, Nonce, UnboundKey};
use ring::agreement::{self, EphemeralPrivateKey, X25519};
use ring::digest::{Context, SHA256};
use ring::hkdf;

use super::identity::{Key, PublicKey};
use super::{Failure, PROTOCOL_VERSION, frame};

const MAGIC: &[u8; 8] = b"FLAREFLT";
const RECORD_LIMIT: usize = 65535;
const TAG_LEN: usize = 16;
/// Plaintext a record carries.
pub const RECORD_PLAINTEXT: usize = RECORD_LIMIT - TAG_LEN;
const HEADER_LEN: usize = 9;
/// Bodies read whole with [`Session::read_frame`] may not exceed this.
pub const FRAME_LIMIT: u64 = 1 << 20;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

struct Len(usize);
impl hkdf::KeyType for Len {
    fn len(&self) -> usize {
        self.0
    }
}

struct Direction {
    key: LessSafeKey,
    counter: u64,
}

impl Direction {
    fn new(key: &[u8]) -> Result<Self, Failure> {
        Ok(Self {
            key: LessSafeKey::new(
                UnboundKey::new(&aead::CHACHA20_POLY1305, key).map_err(|_| "bad session key")?,
            ),
            counter: 0,
        })
    }

    fn nonce(&mut self) -> Result<Nonce, Failure> {
        let mut bytes = [0u8; aead::NONCE_LEN];
        bytes[4..].copy_from_slice(&self.counter.to_be_bytes());
        self.counter = self
            .counter
            .checked_add(1)
            .ok_or("session nonce space exhausted")?;
        Ok(Nonce::assume_unique_for_key(bytes))
    }

    fn seal(&mut self, plaintext: &[u8]) -> Result<Vec<u8>, Failure> {
        let mut buffer = plaintext.to_vec();
        let nonce = self.nonce()?;
        self.key
            .seal_in_place_append_tag(nonce, Aad::empty(), &mut buffer)
            .map_err(|_| "record encryption failed")?;
        Ok(buffer)
    }

    fn open(&mut self, mut ciphertext: Vec<u8>) -> Result<Vec<u8>, Failure> {
        let nonce = self.nonce()?;
        let length = self
            .key
            .open_in_place(nonce, Aad::empty(), &mut ciphertext)
            .map_err(|_| "record authentication failed")?
            .len();
        ciphertext.truncate(length);
        Ok(ciphertext)
    }
}

/// Which side of the handshake this session took.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Initiator,
    Responder,
}

pub struct Session {
    stream: TcpStream,
    send: Direction,
    receive: Direction,
    remote: PublicKey,
    handshake_hash: [u8; 32],
    role: Role,
    /// Decrypted bytes of the record that carried the current frame header, not yet consumed.
    pending: Vec<u8>,
    /// Body bytes of the current frame not yet consumed.
    remaining: u64,
    /// A failed streaming write must close the connection, never append another frame.
    incomplete_write: bool,
}

fn io_error(context: &str, error: io::Error) -> Failure {
    format!("{context}: {error}").into()
}

fn write_record(stream: &mut TcpStream, ciphertext: &[u8]) -> Result<(), Failure> {
    debug_assert!(ciphertext.len() <= RECORD_LIMIT);
    stream
        .write_all(&(ciphertext.len() as u16).to_be_bytes())
        .and_then(|_| stream.write_all(ciphertext))
        .map_err(|e| io_error("session write failed", e))
}

fn read_record(stream: &mut TcpStream) -> Result<Vec<u8>, Failure> {
    let mut length = [0u8; 2];
    stream
        .read_exact(&mut length)
        .map_err(|e| io_error("session read failed", e))?;
    let mut ciphertext = vec![0u8; u16::from_be_bytes(length) as usize];
    stream
        .read_exact(&mut ciphertext)
        .map_err(|e| io_error("session read failed", e))?;
    Ok(ciphertext)
}

fn read_exact(stream: &mut TcpStream, count: usize) -> Result<Vec<u8>, Failure> {
    let mut bytes = vec![0u8; count];
    stream
        .read_exact(&mut bytes)
        .map_err(|e| io_error("handshake read failed", e))?;
    Ok(bytes)
}

/// Both directions' keys from the shared secret and the transcript of the ephemeral messages.
fn derive(shared: &[u8], transcript: &[u8]) -> Result<([u8; 32], [u8; 32]), Failure> {
    let prk = hkdf::Salt::new(hkdf::HKDF_SHA256, transcript).extract(shared);
    let mut i2r = [0u8; 32];
    let mut r2i = [0u8; 32];
    prk.expand(&[b"flare-fleet initiator"], Len(32))
        .and_then(|okm| okm.fill(&mut i2r))
        .map_err(|_| "key derivation failed")?;
    prk.expand(&[b"flare-fleet responder"], Len(32))
        .and_then(|okm| okm.fill(&mut r2i))
        .map_err(|_| "key derivation failed")?;
    Ok((i2r, r2i))
}

fn transcript_hash(parts: &[&[u8]]) -> [u8; 32] {
    let mut context = Context::new(&SHA256);
    for part in parts {
        context.update(part);
    }
    context.finish().as_ref().try_into().unwrap()
}

fn authenticate(identity: &[u8], transcript: &[u8; 32], label: &[u8]) -> Vec<u8> {
    [label, transcript, identity].concat()
}

/// What a party proves in its authentication record: its public key and a signature over the
/// labelled transcript.
fn proof(key: &Key, transcript: &[u8; 32], label: &[u8]) -> Vec<u8> {
    let public = key.public();
    let message = authenticate(public.as_bytes(), transcript, label);
    [public.as_bytes().as_slice(), &key.sign(&message)].concat()
}

fn verify_proof(record: &[u8], transcript: &[u8; 32], label: &[u8]) -> Result<PublicKey, Failure> {
    if record.len() != 96 {
        return Err("malformed authentication record".into());
    }
    let public = PublicKey::from_bytes(&record[..32])?;
    public
        .verify(
            &authenticate(public.as_bytes(), transcript, label),
            &record[32..],
        )
        .map_err(|_| "peer authentication failed")?;
    Ok(public)
}

impl Session {
    /// Connect only through the selected fabric address; never retry on management Ethernet.
    pub fn connect_from(local: Ipv4Addr, remote: SocketAddrV4, key: &Key) -> Result<Self, Failure> {
        let socket = socket2::Socket::new(
            socket2::Domain::IPV4,
            socket2::Type::STREAM,
            Some(socket2::Protocol::TCP),
        )?;
        socket.bind(&SocketAddr::V4(SocketAddrV4::new(local, 0)).into())?;
        socket
            .connect_timeout(&SocketAddr::V4(remote).into(), Duration::from_secs(10))
            .map_err(|e| format!("cannot reach fabric agent {remote} from {local}: {e}"))?;
        let stream: TcpStream = socket.into();
        stream.set_write_timeout(Some(Duration::from_secs(120)))?;
        let mut session = Self::initiate(stream, key)?;
        session.transfer_timeout()?;
        Ok(session)
    }

    pub fn transfer_timeout(&mut self) -> Result<(), Failure> {
        self.stream
            .set_read_timeout(Some(Duration::from_secs(120)))?;
        self.stream
            .set_write_timeout(Some(Duration::from_secs(120)))?;
        Ok(())
    }

    pub fn addresses(&self) -> Result<(SocketAddr, SocketAddr), Failure> {
        Ok((self.stream.local_addr()?, self.stream.peer_addr()?))
    }

    /// Open a session to `address` as the initiator.
    pub fn connect(address: &str, key: &Key) -> Result<Self, Failure> {
        let stream = TcpStream::connect_timeout(
            &address
                .to_socket_addrs()
                .map_err(|e| format!("cannot resolve {address}: {e}"))?
                .next()
                .ok_or_else(|| format!("cannot resolve {address}"))?,
            Duration::from_secs(10),
        )
        .map_err(|e| format!("cannot reach flare agent at {address}: {e}"))?;
        Self::initiate(stream, key)
    }

    pub fn initiate(mut stream: TcpStream, key: &Key) -> Result<Self, Failure> {
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
        let rng = ring::rand::SystemRandom::new();
        let ephemeral = EphemeralPrivateKey::generate(&X25519, &rng).map_err(|_| "no entropy")?;
        let mine = ephemeral
            .compute_public_key()
            .map_err(|_| "bad ephemeral key")?;
        let message1 = [
            MAGIC.as_slice(),
            &PROTOCOL_VERSION.to_le_bytes(),
            mine.as_ref(),
        ]
        .concat();
        stream
            .write_all(&message1)
            .map_err(|e| io_error("handshake write failed", e))?;
        let theirs = read_exact(&mut stream, 32)?;
        let shared = agreement::agree_ephemeral(
            ephemeral,
            &agreement::UnparsedPublicKey::new(&X25519, &theirs),
            |secret| secret.to_vec(),
        )
        .map_err(|_| "key agreement failed")?;
        let ephemerals = transcript_hash(&[&message1, &theirs]);
        let (i2r, r2i) = derive(&shared, &ephemerals)?;
        let mut send = Direction::new(&i2r)?;
        let mut receive = Direction::new(&r2i)?;
        let record2 = read_record(&mut stream)?;
        let remote = verify_proof(&receive.open(record2.clone())?, &ephemerals, b"responder")?;
        let transcript = transcript_hash(&[&message1, &theirs, &record2]);
        let record3 = send.seal(&proof(key, &transcript, b"initiator"))?;
        write_record(&mut stream, &record3)?;
        stream.set_read_timeout(None)?;
        Ok(Self {
            stream,
            send,
            receive,
            remote,
            handshake_hash: transcript_hash(&[&message1, &theirs, &record2, &record3]),
            role: Role::Initiator,
            pending: Vec::new(),
            remaining: 0,
            incomplete_write: false,
        })
    }

    /// Complete the handshake on an accepted connection as the responder.
    pub fn accept(mut stream: TcpStream, key: &Key) -> Result<Self, Failure> {
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
        let message1 = read_exact(&mut stream, 8 + 4 + 32)?;
        if &message1[..8] != MAGIC {
            return Err("not a flare orbit connection".into());
        }
        let version = u32::from_le_bytes(message1[8..12].try_into().unwrap());
        if version != PROTOCOL_VERSION {
            return Err(format!(
                "peer speaks agent protocol {version}; this agent speaks {PROTOCOL_VERSION}"
            )
            .into());
        }
        let rng = ring::rand::SystemRandom::new();
        let ephemeral = EphemeralPrivateKey::generate(&X25519, &rng).map_err(|_| "no entropy")?;
        let mine = ephemeral
            .compute_public_key()
            .map_err(|_| "bad ephemeral key")?;
        let shared = agreement::agree_ephemeral(
            ephemeral,
            &agreement::UnparsedPublicKey::new(&X25519, &message1[12..]),
            |secret| secret.to_vec(),
        )
        .map_err(|_| "key agreement failed")?;
        let ephemerals = transcript_hash(&[&message1, mine.as_ref()]);
        let (i2r, r2i) = derive(&shared, &ephemerals)?;
        let mut send = Direction::new(&r2i)?;
        let mut receive = Direction::new(&i2r)?;
        let record2 = send.seal(&proof(key, &ephemerals, b"responder"))?;
        stream
            .write_all(mine.as_ref())
            .map_err(|e| io_error("handshake write failed", e))?;
        write_record(&mut stream, &record2)?;
        let transcript = transcript_hash(&[&message1, mine.as_ref(), &record2]);
        let record3 = read_record(&mut stream)?;
        let remote = verify_proof(&receive.open(record3.clone())?, &transcript, b"initiator")?;
        stream.set_read_timeout(None)?;
        Ok(Self {
            stream,
            send,
            receive,
            remote,
            handshake_hash: transcript_hash(&[&message1, mine.as_ref(), &record2, &record3]),
            role: Role::Responder,
            pending: Vec::new(),
            remaining: 0,
            incomplete_write: false,
        })
    }

    /// The authenticated identity on the other side.
    pub fn remote(&self) -> PublicKey {
        self.remote
    }

    /// SHA-256 of every handshake byte: the channel binding for proofs sent inside the session.
    pub fn handshake_hash(&self) -> &[u8; 32] {
        &self.handshake_hash
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn set_read_timeout(&mut self, timeout: Option<Duration>) -> Result<(), Failure> {
        self.stream.set_read_timeout(timeout)?;
        Ok(())
    }

    /// A shutdown handle for the connection; framed reads and writes stay with the session.
    pub fn try_clone_stream(&self) -> Result<TcpStream, Failure> {
        Ok(self.stream.try_clone()?)
    }

    fn write_plain(&mut self, plaintext: &[u8]) -> Result<(), Failure> {
        let record = self.send.seal(plaintext)?;
        write_record(&mut self.stream, &record)
    }

    /// Write a complete frame.
    pub fn write_frame(&mut self, tag: u8, body: &[u8]) -> Result<(), Failure> {
        let mut first = Vec::with_capacity(HEADER_LEN + body.len().min(RECORD_PLAINTEXT));
        first.push(tag);
        first.extend((body.len() as u64).to_le_bytes());
        let split = body.len().min(RECORD_PLAINTEXT - HEADER_LEN);
        first.extend(&body[..split]);
        self.write_plain(&first)?;
        for chunk in body[split..].chunks(RECORD_PLAINTEXT) {
            self.write_plain(chunk)?;
        }
        self.stream.flush()?;
        Ok(())
    }

    /// Write a frame whose `length` body bytes come from `source`.
    pub fn write_frame_from(
        &mut self,
        tag: u8,
        length: u64,
        source: &mut dyn Read,
    ) -> Result<(), Failure> {
        let mut buffer = vec![0u8; RECORD_PLAINTEXT];
        buffer[0] = tag;
        buffer[1..HEADER_LEN].copy_from_slice(&length.to_le_bytes());
        let mut remaining = length;
        let mut offset = HEADER_LEN;
        loop {
            let want = (buffer.len() - offset).min(remaining as usize);
            if want > 0 {
                source
                    .read_exact(&mut buffer[offset..offset + want])
                    .map_err(|e| io_error("frame source read failed", e))?;
                remaining -= want as u64;
            }
            self.incomplete_write = true;
            self.write_plain(&buffer[..offset + want])?;
            offset = 0;
            if remaining == 0 {
                break;
            }
        }
        self.stream.flush()?;
        self.incomplete_write = false;
        Ok(())
    }

    /// Read the next frame's header. Its body is then read with [`Self::body`] or
    /// [`Self::read_body`]; a body left unread is skipped by the next header read.
    pub fn read_header(&mut self) -> Result<(u8, u64), Failure> {
        self.skip_body()?;
        let record = self.receive.open(read_record(&mut self.stream)?)?;
        if record.len() < HEADER_LEN {
            return Err("truncated frame header".into());
        }
        let tag = record[0];
        let length = u64::from_le_bytes(record[1..HEADER_LEN].try_into().unwrap());
        if (record.len() - HEADER_LEN) as u64 > length {
            return Err("frame body exceeds its declared length".into());
        }
        self.pending = record[HEADER_LEN..].to_vec();
        self.remaining = length - self.pending.len() as u64;
        Ok((tag, length))
    }

    fn skip_body(&mut self) -> Result<(), Failure> {
        self.pending.clear();
        while self.remaining > 0 {
            let record = self.receive.open(read_record(&mut self.stream)?)?;
            self.remaining = self
                .remaining
                .checked_sub(record.len() as u64)
                .ok_or("frame body exceeds its declared length")?;
        }
        Ok(())
    }

    /// A reader over the current frame's body.
    pub fn body(&mut self) -> Body<'_> {
        Body { session: self }
    }

    /// Consume a fetch connection as a reader; dropping it closes the transfer.
    pub fn into_body(self) -> impl Read {
        OwnedBody(self)
    }

    /// The current frame's body, which must not exceed `limit` bytes.
    pub fn read_body(&mut self, length: u64, limit: u64) -> Result<Vec<u8>, Failure> {
        if length > limit {
            return Err(format!("frame body of {length} bytes exceeds {limit}").into());
        }
        let mut body = Vec::with_capacity(length as usize);
        self.body().read_to_end(&mut body)?;
        Ok(body)
    }

    /// Read a complete frame whose body fits [`FRAME_LIMIT`].
    pub fn read_frame(&mut self) -> Result<(u8, Vec<u8>), Failure> {
        let (tag, length) = self.read_header()?;
        Ok((tag, self.read_body(length, FRAME_LIMIT)?))
    }

    /// Read a frame that must carry `tag`; an ERROR frame becomes the error it carries.
    pub fn expect(&mut self, tag: u8) -> Result<Vec<u8>, Failure> {
        match self.read_frame()? {
            (found, body) if found == tag => Ok(body),
            (frame::ERROR, body) => Err(String::from_utf8_lossy(&body).into_owned().into()),
            (found, _) => Err(format!("unexpected frame {found}, expected {tag}").into()),
        }
    }

    /// Send `request` as `tag` with a JSON body and return the OK body.
    pub fn request(&mut self, tag: u8, request: &serde_json::Value) -> Result<Vec<u8>, Failure> {
        self.write_frame(tag, &serde_json::to_vec(request)?)?;
        self.expect(frame::OK)
    }

    pub fn ok(&mut self, body: &[u8]) -> Result<(), Failure> {
        self.write_frame(frame::OK, body)
    }

    pub fn error(&mut self, message: &str) -> Result<(), Failure> {
        if self.incomplete_write {
            self.stream.shutdown(std::net::Shutdown::Both)?;
            return Ok(());
        }
        self.write_frame(frame::ERROR, message.as_bytes())
    }
}

use std::net::ToSocketAddrs;

/// Reads the current frame's body.
pub struct Body<'a> {
    session: &'a mut Session,
}

struct OwnedBody(Session);

impl Read for OwnedBody {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.0.body().read(buffer)
    }
}

impl Read for Body<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let session = &mut *self.session;
        if session.pending.is_empty() {
            if session.remaining == 0 {
                return Ok(0);
            }
            let record = read_record(&mut session.stream)
                .and_then(|record| session.receive.open(record))
                .map_err(|e| io::Error::other(e.to_string()))?;
            if record.len() as u64 > session.remaining {
                return Err(io::Error::other("frame body exceeds its declared length"));
            }
            session.remaining -= record.len() as u64;
            session.pending = record;
        }
        let count = buffer.len().min(session.pending.len());
        buffer[..count].copy_from_slice(&session.pending[..count]);
        session.pending.drain(..count);
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    fn pair() -> (Session, Session) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let server_key = Key::from_seed(&[1; 32]).unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            Session::accept(stream, &server_key).unwrap()
        });
        let client = Session::connect(&address, &Key::from_seed(&[2; 32]).unwrap()).unwrap();
        (client, server.join().unwrap())
    }

    #[test]
    fn a_failed_relay_stream_closes_instead_of_inserting_an_error_into_payload() {
        let (mut client, mut server) = pair();
        let length = (RECORD_PLAINTEXT * 3) as u64;
        let writer = thread::spawn(move || {
            let bytes = vec![42u8; RECORD_PLAINTEXT * 2];
            let error = server
                .write_frame_from(frame::OK, length, &mut bytes.as_slice())
                .unwrap_err();
            server.error(&error.to_string()).unwrap();
        });
        assert_eq!(client.read_header().unwrap(), (frame::OK, length));
        let mut received = Vec::new();
        assert!(client.body().read_to_end(&mut received).is_err());
        assert!(!received.is_empty());
        assert!(received.iter().all(|b| *b == 42));
        writer.join().unwrap();
    }

    #[test]
    fn both_sides_learn_the_other_identity_and_the_same_binding() {
        let (client, server) = pair();
        assert_eq!(client.remote(), Key::from_seed(&[1; 32]).unwrap().public());
        assert_eq!(server.remote(), Key::from_seed(&[2; 32]).unwrap().public());
        assert_eq!(client.handshake_hash(), server.handshake_hash());
        assert_eq!(client.role(), Role::Initiator);
        assert_eq!(server.role(), Role::Responder);
    }

    #[test]
    fn frames_round_trip_across_record_boundaries() {
        let (mut client, mut server) = pair();
        let body: Vec<u8> = (0..(3 * RECORD_PLAINTEXT + 17)).map(|i| i as u8).collect();
        let sent = body.clone();
        let writer = thread::spawn(move || {
            client.write_frame(frame::PUT, &sent).unwrap();
            client.write_frame(frame::OK, b"").unwrap();
            client
                .write_frame_from(frame::LOG, sent.len() as u64, &mut sent.as_slice())
                .unwrap();
            client
        });
        let (tag, length) = server.read_header().unwrap();
        assert_eq!((tag, length), (frame::PUT, body.len() as u64));
        let mut received = Vec::new();
        server.body().read_to_end(&mut received).unwrap();
        assert_eq!(received, body);
        assert_eq!(server.read_frame().unwrap(), (frame::OK, Vec::new()));
        // An unread body is skipped by the next header read.
        let (tag, _) = server.read_header().unwrap();
        assert_eq!(tag, frame::LOG);
        let mut client = writer.join().unwrap();
        client.write_frame(frame::EXIT, &[1, 2, 3]).unwrap();
        assert_eq!(server.read_frame().unwrap(), (frame::EXIT, vec![1, 2, 3]));
        assert!(server.read_body(FRAME_LIMIT + 1, FRAME_LIMIT).is_err());
    }

    #[test]
    fn tampered_records_and_wrong_versions_are_refused() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            Session::accept(stream, &Key::from_seed(&[1; 32]).unwrap())
                .err()
                .unwrap()
                .to_string()
        });
        let mut stream = TcpStream::connect(address).unwrap();
        stream.write_all(MAGIC).unwrap();
        stream
            .write_all(&(PROTOCOL_VERSION + 1).to_le_bytes())
            .unwrap();
        stream.write_all(&[0; 32]).unwrap();
        assert!(server.join().unwrap().contains("protocol"));

        let (mut client, mut server) = pair();
        client.write_frame(frame::OK, b"fine").unwrap();
        assert_eq!(server.read_frame().unwrap(), (frame::OK, b"fine".to_vec()));
        // A replayed or corrupted record fails authentication.
        let mut raw = client.try_clone_stream().unwrap();
        let record = client.send.seal(b"x").unwrap();
        let mut corrupted = record.clone();
        corrupted[0] ^= 1;
        write_record(&mut raw, &corrupted).unwrap();
        assert!(server.read_frame().is_err());
    }
}
