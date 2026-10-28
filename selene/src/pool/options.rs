use crate::{CacheGeometry, geometry};

/// The manager an instance shares prefix checkpoints with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolOptions {
    /// `unix:PATH`, plaintext `HOST:PORT`, or TLS `tls:HOST:PORT`.
    pub address: String,
    /// The CA certificate the manager's TLS certificate must chain to. Without
    /// one, TLS uses the host's trust store. Only a `tls:HOST:PORT` manager can use one.
    pub ca: Option<std::path::PathBuf>,
}

/// Accept a Unix socket, plain TCP address, or explicit TLS address.
pub fn parse_pool_address(text: &str) -> std::result::Result<String, String> {
    pool_address(text).map(|_| text.to_owned())
}

pub fn pool_address(text: &str) -> std::result::Result<crate::pool::PoolAddress, String> {
    use crate::pool::PoolAddress;
    let invalid =
        || format!("invalid pool address '{text}': use unix:PATH, HOST:PORT or tls:HOST:PORT");
    if let Some(path) = text.strip_prefix("unix:") {
        return if path.is_empty() || path.chars().any(char::is_control) {
            Err(invalid())
        } else {
            Ok(PoolAddress::Unix(path.into()))
        };
    }
    let address = text.strip_prefix("tls:").unwrap_or(text);
    let host_valid = |host: &str| {
        if let Some(ip) = host.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            return ip.parse::<std::net::Ipv6Addr>().is_ok();
        }
        !host.is_empty()
            && host
                .strip_suffix('.')
                .unwrap_or(host)
                .split('.')
                .all(|label| {
                    !label.is_empty()
                        && !label.starts_with('-')
                        && !label.ends_with('-')
                        && label
                            .bytes()
                            .all(|c| c.is_ascii_alphanumeric() || c == b'-')
                })
    };
    match address.rsplit_once(':') {
        Some((host, port))
            if host_valid(host)
                && port.bytes().all(|c| c.is_ascii_digit())
                && port.parse::<u16>().is_ok_and(|port| port != 0) =>
        {
            Ok(if text.starts_with("tls:") {
                PoolAddress::Tcp(address.to_owned())
            } else {
                PoolAddress::TcpPlaintext(address.to_owned())
            })
        }
        _ => Err(invalid()),
    }
}

/// What a pool shares: the model and record layout.
pub(super) fn pool_scope(
    identity: [u8; 32],
    geometry: &[CacheGeometry],
) -> crate::Result<crate::shared::Scope> {
    let mut scope = crate::shared::Scope::new(identity, geometry::fingerprint(geometry), geometry)?;
    for rank in &mut scope.geometry.ranks {
        for bytes in rank.plane_bytes.iter_mut().chain([&mut rank.row_bytes]) {
            *bytes = bytes.div_ceil(geometry::BLOCK_BYTES) * geometry::BLOCK_BYTES;
        }
    }
    Ok(scope)
}

#[cfg(test)]
mod tests;
