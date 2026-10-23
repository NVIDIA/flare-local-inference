//! Orbit: resident agents on every host of a tensor-parallel group, and the controller
//! commands that pair with them, move executables and packages, and run, attach to and stop
//! groups from any machine. See docs/orbit.md.
//!
//! Every connection, controller to node and node to node, is an encrypted session between two
//! Ed25519 identities ([`identity`], [`session`]). A node runs ranks only under a launch grant
//! signed by one of its authorized controllers ([`grant`]), keeps received files in its cache
//! ([`cache`]), and supervises its rank for the process's life ([`agent`]). The controller side
//! ([`controller`]) keeps the orbit file ([`store`]) and drives launches.

pub mod agent;
pub mod banner;
pub(crate) use crate::package_cache as cache;
pub mod controller;
pub mod dashboard;
pub mod fabric;
pub mod gpu;
pub mod grant;
pub mod identity;
pub mod plan;
pub mod session;
pub mod store;

use std::path::PathBuf;

pub type Failure = Box<dyn std::error::Error + Send + Sync>;

/// The agent's default port on every host.
pub const PORT: u16 = 29900;
/// The agent protocol every node and controller of a orbit must share.
pub const PROTOCOL_VERSION: u32 = 6;

/// Frame tags. A request receives OK with a body or ERROR with a message.
pub mod frame {
    pub const OK: u8 = 1;
    pub const ERROR: u8 = 2;
    pub const HELLO: u8 = 3;
    pub const PAIR: u8 = 4;
    pub const HAVE: u8 = 5;
    pub const PUT: u8 = 6;
    pub const TRANSFER: u8 = 7;
    pub const FETCH: u8 = 8;
    pub const RUN: u8 = 9;
    pub const ATTACH: u8 = 10;
    pub const STOP: u8 = 11;
    pub const STATUS: u8 = 12;
    pub const LOG: u8 = 13;
    pub const EXIT: u8 = 14;
    pub const GET: u8 = 15;
    pub const STAT: u8 = 16;
    pub const CONTROLLERS: u8 = 17;
    pub const METRICS: u8 = 18;
    pub const REMOVE: u8 = 19;
    pub const DOWNLOAD: u8 = 20;
    pub const FABRIC: u8 = 21;
    pub const FETCH_FABRIC: u8 = 22;
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(text: &str) -> Result<Vec<u8>, Failure> {
    if !text.len().is_multiple_of(2) || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("invalid hex {text:?}").into());
    }
    Ok((0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect())
}

pub fn unhex32(text: &str) -> Result<[u8; 32], Failure> {
    unhex(text)?
        .try_into()
        .map_err(|_| "expected 32 hex-encoded bytes".into())
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/".into())
}

/// Random bytes from the operating system.
pub fn random(count: usize) -> Result<Vec<u8>, Failure> {
    use ring::rand::SecureRandom;
    let mut bytes = vec![0u8; count];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| "random number generation failed")?;
    Ok(bytes)
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    ring::digest::digest(&ring::digest::SHA256, bytes)
        .as_ref()
        .try_into()
        .unwrap()
}

pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, key);
    ring::hmac::sign(&key, message).as_ref().try_into().unwrap()
}

/// Seconds since the Unix epoch.
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The host's name, as agents name themselves by default.
pub fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|name| name.trim().to_owned())
        .ok()
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "node".into())
}

/// `HOST[:PORT]` with the agent's default port applied.
pub fn with_port(host: &str) -> String {
    match host.rsplit_once(':') {
        Some((_, port)) if port.parse::<u16>().is_ok() => host.to_owned(),
        _ => format!("{host}:{PORT}"),
    }
}

/// The local address this host uses to reach `target`: the source address of a route toward it.
pub fn address_toward(target: &str) -> Result<std::net::IpAddr, Failure> {
    use std::net::{ToSocketAddrs, UdpSocket};
    let target = with_port(target)
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| format!("cannot resolve {target}"))?;
    let socket = UdpSocket::bind(if target.is_ipv4() {
        "0.0.0.0:0"
    } else {
        "[::]:0"
    })?;
    socket.connect(target)?;
    Ok(socket.local_addr()?.ip())
}

/// Encode bytes as RFC 4648 base32 without padding, as join codes are shown.
pub fn base32(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let mut out = String::new();
    let (mut buffer, mut bits) = (0u32, 0u32);
    for &byte in bytes {
        buffer = (buffer << 8) | byte as u32;
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            out.push(ALPHABET[((buffer >> bits) & 31) as usize] as char);
        }
    }
    if bits > 0 {
        out.push(ALPHABET[((buffer << (5 - bits)) & 31) as usize] as char);
    }
    out
}

/// A join code as typed: groups of four separated by hyphens or spaces, case-insensitive.
pub fn normalize_code(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect()
}
