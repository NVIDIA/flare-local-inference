//! The byte framing of the pool's data channel.
//!
//! The wire contract fixes only what a data channel must guarantee (see
//! `DataChannelRef` in the proto); this module is the one framing both ends use.
//! A transfer is one connection: the instance connects to the manager's data
//! listener, presents the grant's 32-byte token, and says which transport it
//! carries payload on. After that, frames flow in the direction of the data.
//! Integers are little-endian.
//!
//! ```text
//! instance -> manager, once, after the token:
//!   HELLO   0x10  carrier u8 (1 stream, 2 RDMA); RDMA adds hs_len u16, hs
//! manager -> instance, once, to an RDMA hello:
//!   ACCEPT  0x07  hs_len u16, hs: the manager's end of the queue pair
//! manager -> instance, a claim:
//!   WINDOW  0x01  index u32, record u32, first_page u32, page_count u32,
//!                 row u8, length u64, row_xxh3 u64, slot u64,
//!                 page_xxh3 u64 * page_count
//!   END     0x02  no window follows; the transfer is complete
//!   ABORT   0x03  text_len u16, text: the manager ended the transfer
//! manager -> instance, a put (the instance answers each GRANT with FILLED):
//!   GRANT   0x04  index u32, record u32, first_page u32, page_count u32,
//!                 row u8, length u64
//! instance -> manager, a put:
//!   FILLED  0x05  index u32, row u8, row_xxh3 u64, page_count u32,
//!                 page_xxh3 u64 * page_count
//! instance -> manager, an RDMA claim:
//!   CREDIT  0x06  offset u64, length u64: a slot the manager may write a window into
//! ```
//!
//! A window's payload is `length` bytes: its pages, then its row if it has one.
//! On the stream carrier a WINDOW's payload follows it on the connection (`slot`
//! is `NO_SLOT`) and so does a FILLED's. On the RDMA carrier the payload does not
//! travel on the connection: the manager writes a claim's window into a slot the
//! instance credited and then sends WINDOW naming that slot's offset. Puts use
//! the stream carrier. Windows arrive in index order and only the transfer's own windows are
//! ever named, so a reader needs no state beyond the grant's plan to check them.

use std::io;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Bound an I/O operation and retain its transport error or the caller's timeout message.
pub async fn within<T>(
    duration: std::time::Duration,
    what: &str,
    future: impl std::future::Future<Output = io::Result<T>>,
) -> io::Result<T> {
    tokio::time::timeout(duration, future)
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, what))?
}

/// Bytes of the single-use token that names a transfer.
pub const TOKEN_BYTES: usize = 32;
/// The most text an ABORT frame may carry.
pub const MAX_ABORT_TEXT: usize = 1024;

const HELLO: u8 = 0x10;
const WINDOW: u8 = 0x01;
const END: u8 = 0x02;
const ABORT: u8 = 0x03;
const GRANT: u8 = 0x04;
const FILLED: u8 = 0x05;
const CREDIT: u8 = 0x06;
const ACCEPT: u8 = 0x07;

/// The `slot` of a window whose payload follows it on the connection.
pub const NO_SLOT: u64 = u64::MAX;
/// The longest queue-pair description either side sends.
pub const MAX_HANDSHAKE: usize = 256;

/// What payload travels on. The stream carries it on the connection itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Carrier {
    Stream,
    /// RDMA: the sender's queue-pair description (opaque here).
    Rdma(Vec<u8>),
}

/// What a window is: where it lies in the claim's records and how long it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub index: u32,
    pub record: u32,
    pub first_page: u32,
    pub page_count: u32,
    pub row: bool,
    pub length: u64,
}

/// A window of a claim, with the checksums stored with its record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub shape: Shape,
    pub page_xxh3: Vec<u64>,
    pub row_xxh3: u64,
    /// Where the payload is in the instance's staging, for the RDMA carrier;
    /// `NO_SLOT` when it follows on the connection.
    pub slot: u64,
}

/// A frame the manager sends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToInstance {
    /// A window of a claim; its payload follows.
    Window(Window),
    End,
    Abort(String),
    /// A window of a put the instance is to fill.
    Grant(Shape),
    /// The manager's end of the queue pair, in answer to an RDMA hello.
    Accept(Vec<u8>),
}

/// A frame the instance sends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToManager {
    Hello(Carrier),
    /// A granted window, filled; its payload follows.
    Filled {
        index: u32,
        row_xxh3: Option<u64>,
        page_xxh3: Vec<u64>,
    },
    /// A slot the manager may write a claim's next window into (RDMA).
    Credit {
        offset: u64,
        length: u64,
    },
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn put_shape(out: &mut Vec<u8>, shape: &Shape) {
    out.extend_from_slice(&shape.index.to_le_bytes());
    out.extend_from_slice(&shape.record.to_le_bytes());
    out.extend_from_slice(&shape.first_page.to_le_bytes());
    out.extend_from_slice(&shape.page_count.to_le_bytes());
    out.push(u8::from(shape.row));
    out.extend_from_slice(&shape.length.to_le_bytes());
}

const SHAPE_BYTES: usize = 4 + 4 + 4 + 4 + 1 + 8;

fn parse_shape(bytes: &[u8; SHAPE_BYTES]) -> io::Result<Shape> {
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let row = match bytes[16] {
        0 => false,
        1 => true,
        _ => return Err(invalid("a window's row flag is neither 0 nor 1")),
    };
    Ok(Shape {
        index: word(0),
        record: word(4),
        first_page: word(8),
        page_count: word(12),
        row,
        length: u64::from_le_bytes(bytes[17..25].try_into().unwrap()),
    })
}

fn put_handshake(out: &mut Vec<u8>, handshake: &[u8]) {
    let handshake = &handshake[..handshake.len().min(MAX_HANDSHAKE)];
    out.extend_from_slice(&(handshake.len() as u16).to_le_bytes());
    out.extend_from_slice(handshake);
}

async fn read_handshake<R: AsyncRead + Unpin>(reader: &mut R) -> io::Result<Vec<u8>> {
    let length = reader.read_u16_le().await? as usize;
    if length == 0 || length > MAX_HANDSHAKE {
        return Err(invalid("a queue-pair description has an unusable length"));
    }
    let mut bytes = vec![0u8; length];
    reader.read_exact(&mut bytes).await?;
    Ok(bytes)
}

fn put_sums(out: &mut Vec<u8>, sums: &[u64]) {
    for sum in sums {
        out.extend_from_slice(&sum.to_le_bytes());
    }
}

async fn read_sums<R: AsyncRead + Unpin>(
    reader: &mut R,
    count: u32,
    max_pages: u32,
) -> io::Result<Vec<u64>> {
    if count > max_pages {
        return Err(invalid(format!(
            "a window names {count} pages, more than the {max_pages} allowed"
        )));
    }
    let mut bytes = vec![0u8; count as usize * 8];
    reader.read_exact(&mut bytes).await?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

impl ToInstance {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            ToInstance::Window(window) => {
                out.push(WINDOW);
                put_shape(&mut out, &window.shape);
                out.extend_from_slice(&window.row_xxh3.to_le_bytes());
                out.extend_from_slice(&window.slot.to_le_bytes());
                put_sums(&mut out, &window.page_xxh3);
            }
            ToInstance::End => out.push(END),
            ToInstance::Abort(text) => {
                let text = &text.as_bytes()[..text.len().min(MAX_ABORT_TEXT)];
                out.push(ABORT);
                out.extend_from_slice(&(text.len() as u16).to_le_bytes());
                out.extend_from_slice(text);
            }
            ToInstance::Grant(shape) => {
                out.push(GRANT);
                put_shape(&mut out, shape);
            }
            ToInstance::Accept(handshake) => {
                out.push(ACCEPT);
                put_handshake(&mut out, handshake);
            }
        }
        out
    }

    /// Read one frame. A window's payload is left unread on the reader.
    /// `max_pages` bounds the checksum table a window may announce.
    pub async fn read<R: AsyncRead + Unpin>(reader: &mut R, max_pages: u32) -> io::Result<Self> {
        let kind = reader.read_u8().await?;
        Self::read_body(kind, reader, max_pages).await
    }

    /// As [`ToInstance::read`], but `None` when the peer closed the connection between
    /// frames, which is how a manager ends a transfer it has no more to say about.
    pub async fn read_or_end<R: AsyncRead + Unpin>(
        reader: &mut R,
        max_pages: u32,
    ) -> io::Result<Option<Self>> {
        let mut kind = [0u8; 1];
        if reader.read(&mut kind).await? == 0 {
            return Ok(None);
        }
        Self::read_body(kind[0], reader, max_pages).await.map(Some)
    }

    async fn read_body<R: AsyncRead + Unpin>(
        kind: u8,
        reader: &mut R,
        max_pages: u32,
    ) -> io::Result<Self> {
        match kind {
            WINDOW => {
                let mut shape = [0u8; SHAPE_BYTES];
                reader.read_exact(&mut shape).await?;
                let shape = parse_shape(&shape)?;
                let row_xxh3 = reader.read_u64_le().await?;
                let slot = reader.read_u64_le().await?;
                let page_xxh3 = read_sums(reader, shape.page_count, max_pages).await?;
                Ok(ToInstance::Window(Window {
                    shape,
                    page_xxh3,
                    row_xxh3,
                    slot,
                }))
            }
            END => Ok(ToInstance::End),
            ABORT => {
                let length = reader.read_u16_le().await? as usize;
                if length > MAX_ABORT_TEXT {
                    return Err(invalid("an abort text is too long"));
                }
                let mut text = vec![0u8; length];
                reader.read_exact(&mut text).await?;
                Ok(ToInstance::Abort(
                    String::from_utf8_lossy(&text).into_owned(),
                ))
            }
            GRANT => {
                let mut shape = [0u8; SHAPE_BYTES];
                reader.read_exact(&mut shape).await?;
                Ok(ToInstance::Grant(parse_shape(&shape)?))
            }
            ACCEPT => Ok(ToInstance::Accept(read_handshake(reader).await?)),
            other => Err(invalid(format!("unknown manager frame {other:#x}"))),
        }
    }

    pub async fn write<W: AsyncWrite + Unpin>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.encode()).await
    }
}

impl ToManager {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            ToManager::Hello(carrier) => {
                out.push(HELLO);
                match carrier {
                    Carrier::Stream => out.push(1),
                    Carrier::Rdma(handshake) => {
                        out.push(2);
                        put_handshake(&mut out, handshake);
                    }
                }
            }
            ToManager::Filled {
                index,
                row_xxh3,
                page_xxh3,
            } => {
                out.push(FILLED);
                out.extend_from_slice(&index.to_le_bytes());
                // A row's checksum is carried with a presence byte.
                out.push(u8::from(row_xxh3.is_some()));
                out.extend_from_slice(&row_xxh3.unwrap_or(0).to_le_bytes());
                out.extend_from_slice(&(page_xxh3.len() as u32).to_le_bytes());
                put_sums(&mut out, page_xxh3);
            }
            ToManager::Credit { offset, length } => {
                out.push(CREDIT);
                out.extend_from_slice(&offset.to_le_bytes());
                out.extend_from_slice(&length.to_le_bytes());
            }
        }
        out
    }

    /// Read one frame. A filled window's payload is left unread on the reader.
    pub async fn read<R: AsyncRead + Unpin>(reader: &mut R, max_pages: u32) -> io::Result<Self> {
        let kind = reader.read_u8().await?;
        Self::read_body(kind, reader, max_pages).await
    }

    /// As [`ToManager::read`], but `None` when the instance closed the connection
    /// between frames.
    pub async fn read_or_end<R: AsyncRead + Unpin>(
        reader: &mut R,
        max_pages: u32,
    ) -> io::Result<Option<Self>> {
        let mut kind = [0u8; 1];
        if reader.read(&mut kind).await? == 0 {
            return Ok(None);
        }
        Self::read_body(kind[0], reader, max_pages).await.map(Some)
    }

    async fn read_body<R: AsyncRead + Unpin>(
        kind: u8,
        reader: &mut R,
        max_pages: u32,
    ) -> io::Result<Self> {
        match kind {
            HELLO => match reader.read_u8().await? {
                1 => Ok(ToManager::Hello(Carrier::Stream)),
                2 => Ok(ToManager::Hello(Carrier::Rdma(
                    read_handshake(reader).await?,
                ))),
                _ => Err(invalid("an unknown payload carrier")),
            },
            CREDIT => Ok(ToManager::Credit {
                offset: reader.read_u64_le().await?,
                length: reader.read_u64_le().await?,
            }),
            FILLED => {
                let index = reader.read_u32_le().await?;
                let has_row = match reader.read_u8().await? {
                    0 => false,
                    1 => true,
                    _ => return Err(invalid("a filled window's row flag is neither 0 nor 1")),
                };
                let row = reader.read_u64_le().await?;
                let count = reader.read_u32_le().await?;
                let page_xxh3 = read_sums(reader, count, max_pages).await?;
                Ok(ToManager::Filled {
                    index,
                    row_xxh3: has_row.then_some(row),
                    page_xxh3,
                })
            }
            other => Err(invalid(format!("unknown instance frame {other:#x}"))),
        }
    }

    pub async fn write<W: AsyncWrite + Unpin>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.encode()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> Shape {
        Shape {
            index: 3,
            record: 1,
            first_page: 2,
            page_count: 3,
            row: true,
            length: 4 * 4096,
        }
    }

    async fn round_trip_to_instance(frame: ToInstance) -> ToInstance {
        let bytes = frame.encode();
        let mut reader = &bytes[..];
        let read = ToInstance::read(&mut reader, 64).await.unwrap();
        assert!(reader.is_empty(), "the frame is read exactly");
        read
    }

    #[tokio::test]
    async fn every_manager_frame_survives_a_round_trip() {
        for frame in [
            ToInstance::Window(Window {
                shape: shape(),
                page_xxh3: vec![1, 2, u64::MAX],
                row_xxh3: 9,
                slot: NO_SLOT,
            }),
            ToInstance::Window(Window {
                shape: shape(),
                page_xxh3: vec![1, 2, 3],
                row_xxh3: 0,
                slot: 8192,
            }),
            ToInstance::Accept(vec![1, 2, 3]),
            ToInstance::End,
            ToInstance::Abort("the claim ended".into()),
            ToInstance::Grant(shape()),
        ] {
            assert_eq!(round_trip_to_instance(frame.clone()).await, frame);
        }
    }

    #[tokio::test]
    async fn every_instance_frame_survives_a_round_trip() {
        for frame in [
            ToManager::Hello(Carrier::Stream),
            ToManager::Hello(Carrier::Rdma(vec![9; 54])),
            ToManager::Filled {
                index: 7,
                row_xxh3: Some(5),
                page_xxh3: vec![10, 11],
            },
            ToManager::Filled {
                index: 0,
                row_xxh3: None,
                page_xxh3: vec![],
            },
            ToManager::Credit {
                offset: 8192,
                length: 1 << 20,
            },
        ] {
            let bytes = frame.encode();
            let mut reader = &bytes[..];
            assert_eq!(ToManager::read(&mut reader, 64).await.unwrap(), frame);
            assert!(reader.is_empty());
        }
    }

    #[test]
    fn a_window_has_the_documented_bytes() {
        let bytes = ToInstance::Window(Window {
            shape: Shape {
                index: 1,
                record: 2,
                first_page: 3,
                page_count: 1,
                row: false,
                length: 4096,
            },
            page_xxh3: vec![0x0102030405060708],
            row_xxh3: 0,
            slot: NO_SLOT,
        })
        .encode();
        let mut want = vec![0x01];
        want.extend([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0]);
        want.extend(4096u64.to_le_bytes());
        want.extend(0u64.to_le_bytes());
        want.extend(u64::MAX.to_le_bytes());
        want.extend([8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(bytes, want);
    }

    #[tokio::test]
    async fn a_close_between_frames_is_an_end_and_a_close_inside_one_is_not() {
        let mut empty: &[u8] = &[];
        assert_eq!(ToInstance::read_or_end(&mut empty, 64).await.unwrap(), None);
        let bytes = ToInstance::End.encode();
        let mut reader = &bytes[..];
        assert_eq!(
            ToInstance::read_or_end(&mut reader, 64).await.unwrap(),
            Some(ToInstance::End)
        );
        let cut = ToInstance::Grant(shape()).encode();
        let mut reader = &cut[..5];
        assert!(ToInstance::read_or_end(&mut reader, 64).await.is_err());
    }

    #[tokio::test]
    async fn hostile_frames_are_refused_without_allocating_what_they_claim() {
        // More pages than the reader allows.
        let mut bytes = ToInstance::Window(Window {
            shape: Shape {
                page_count: 1 << 30,
                ..shape()
            },
            page_xxh3: vec![],
            row_xxh3: 0,
            slot: NO_SLOT,
        })
        .encode();
        let mut reader = &bytes[..];
        assert!(ToInstance::read(&mut reader, 64).await.is_err());
        // An unknown kind, a bad row flag, and a truncated frame.
        bytes[0] = 0x7f;
        let mut reader = &bytes[..];
        assert!(ToInstance::read(&mut reader, 64).await.is_err());
        let mut flagged = ToInstance::Grant(shape()).encode();
        flagged[17] = 2;
        let mut reader = &flagged[..];
        assert!(ToInstance::read(&mut reader, 64).await.is_err());
        let cut = ToInstance::Grant(shape()).encode();
        let mut reader = &cut[..cut.len() - 1];
        assert!(ToInstance::read(&mut reader, 64).await.is_err());
        let mut reader = &[0x01u8][..];
        assert!(ToManager::read(&mut reader, 64).await.is_err());
    }
}
