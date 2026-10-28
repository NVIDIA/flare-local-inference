//! A scripted instance on the data channel: it dials the grant's address, presents the
//! token, and reads a claim's windows or fills a put's, byte for byte as a real client
//! would. It shares the framing with the client crate and nothing else.

use super::pool::{Damage, PutResult, Session};
use flare_selene::shared::frame::{Carrier, ToInstance, ToManager, Window};
use selene_manager::{Key, proto::v2};
use std::{sync::Arc, time::Duration};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::TcpStream,
    time::timeout,
};

pub trait Stream: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Stream for T {}

pub type Conn = Box<dyn Stream>;

/// What the instance requires of the connection.
#[derive(Clone)]
pub enum Security {
    Plain,
    /// TLS, trusting these roots, with the manager's certificate named `localhost`.
    Tls(rustls::RootCertStore),
}

/// Pages in a window never exceed this.
pub const MAX_PAGES: u32 = 64;

pub fn connector(roots: rustls::RootCertStore) -> tokio_rustls::TlsConnector {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(roots)
        .with_no_client_auth();
    tokio_rustls::TlsConnector::from(Arc::new(config))
}

/// Connect to the grant's address and present its token and hello.
pub async fn dial(reference: &v2::DataChannelRef, security: &Security) -> Conn {
    let tcp = TcpStream::connect(&reference.address)
        .await
        .expect("connect to the data listener");
    let mut conn: Conn = match security {
        Security::Plain => Box::new(tcp),
        Security::Tls(roots) => Box::new(
            connector(roots.clone())
                .connect(
                    rustls::pki_types::ServerName::try_from("localhost").unwrap(),
                    tcp,
                )
                .await
                .expect("the TLS handshake"),
        ),
    };
    conn.write_all(&reference.token).await.unwrap();
    ToManager::Hello(Carrier::Stream)
        .write(&mut conn)
        .await
        .unwrap();
    conn
}

pub struct Received {
    pub window: Window,
    pub payload: Vec<u8>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Ended {
    /// The manager said the claim is complete.
    End,
    /// The manager said why it stopped.
    Abort(String),
    /// The connection ended or broke first.
    Broken,
}

/// Read windows and their payloads until the manager ends the transfer.
pub async fn read_windows(conn: &mut Conn) -> (Vec<Received>, Ended) {
    let mut received = Vec::new();
    loop {
        let frame = match timeout(Duration::from_secs(10), ToInstance::read(conn, MAX_PAGES)).await
        {
            Ok(Ok(frame)) => frame,
            _ => return (received, Ended::Broken),
        };
        match frame {
            ToInstance::Window(window) => {
                let mut payload = vec![0u8; window.shape.length as usize];
                match timeout(Duration::from_secs(10), conn.read_exact(&mut payload)).await {
                    Ok(Ok(_)) => received.push(Received { window, payload }),
                    _ => return (received, Ended::Broken),
                }
            }
            ToInstance::End => return (received, Ended::End),
            ToInstance::Abort(text) => return (received, Ended::Abort(text)),
            _ => return (received, Ended::Broken),
        }
    }
}

pub async fn put_record(
    session: &mut Session,
    security: &Security,
    put_id: u64,
    key: Key,
    parent: Option<Key>,
    damage: Damage,
) -> PutResult {
    super::put::put_record(session, Some(security), put_id, key, parent, damage).await
}
