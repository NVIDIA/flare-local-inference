//! Rank protocol messages carried by the shared shard link.
pub use crate::shard::Link as RankLink;
use crate::{
    Error, Result,
    shard::{Link, Writer},
};
use prost::Message;

use super::wire;
use crate::{shard::Reader, shared::SliceSums};

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Open {
    pub xfer: u64,
    pub manager_incarnation: u64,
    pub generation: u64,
    pub out: bool,
    pub kind: i32,
    pub fence_id: u64,
    pub records: Vec<wire::Record>,
    pub pages: Vec<u32>,
    pub has_slot: bool,
    pub slot: u32,
    pub channel: Option<wire::DataChannelRef>,
    pub page_bytes: u64,
    pub row_bytes: u64,
    pub host: bool,
    pub chain_first: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Window {
    pub xfer: u64,
    pub spec: Option<wire::WindowSpec>,
    pub sums: Option<SliceSums>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Msg {
    Hello {
        rank: u32,
        world: u32,
        rank0_nonce: Vec<u8>,
        layout: Vec<u8>,
    },
    Bind {
        rank_secret: Vec<u8>,
        instance_id: Vec<u8>,
    },
    Open(Open),
    Window(Window),
    Cancel {
        xfer: u64,
    },
    Close,
    AgentHello {
        rank: u32,
        world: u32,
        agent_nonce: Vec<u8>,
        layout: Vec<u8>,
        gpu_uuid: Vec<u8>,
    },
    Attached {
        manager_incarnation: u64,
        generation: u64,
        mode: i32,
    },
    Detached {
        reason: String,
    },
    WindowDone {
        xfer: u64,
        index: u32,
        ok: bool,
        error: String,
        sums: Option<SliceSums>,
    },
    Done {
        xfer: u64,
        ok: bool,
        error: String,
        quarantined: bool,
    },
    Disabled {
        reason: String,
    },
}

/// Rank 0's links, in peer-rank order.
pub type RankSet = Vec<Link>;

fn encode(message: Msg) -> Vec<u8> {
    let tag = match &message {
        Msg::Hello { .. } => 12,
        Msg::Bind { .. } => 13,
        Msg::Open(_) => 14,
        Msg::Window(_) => 15,
        Msg::Cancel { .. } => 16,
        Msg::Close => 17,
        Msg::AgentHello { .. } => 18,
        Msg::Attached { .. } => 19,
        Msg::Detached { .. } => 20,
        Msg::WindowDone { .. } => 21,
        Msg::Done { .. } => 22,
        Msg::Disabled { .. } => 23,
    };
    let mut w = Writer::new(tag);
    match message {
        Msg::Hello {
            rank,
            world,
            rank0_nonce,
            layout,
        } => {
            w.u32(rank).u32(world).blob(&rank0_nonce).blob(&layout);
        }
        Msg::Bind {
            rank_secret,
            instance_id,
        } => {
            w.blob(&rank_secret).blob(&instance_id);
        }
        Msg::AgentHello {
            rank,
            world,
            agent_nonce,
            layout,
            gpu_uuid,
        } => {
            w.u32(rank)
                .u32(world)
                .blob(&agent_nonce)
                .blob(&layout)
                .blob(&gpu_uuid);
        }
        Msg::Attached {
            manager_incarnation,
            generation,
            mode,
        } => {
            w.u64(manager_incarnation).u64(generation).u32(mode as u32);
        }
        Msg::Detached { reason } | Msg::Disabled { reason } => {
            w.text(&reason);
        }
        Msg::Cancel { xfer } => {
            w.u64(xfer);
        }
        Msg::Close => {}
        Msg::Open(o) => {
            w.u64(o.xfer)
                .u64(o.manager_incarnation)
                .u64(o.generation)
                .u8(o.out as u8)
                .u32(o.kind as u32)
                .u64(o.fence_id);
            w.u32(o.records.len() as u32);
            for record in o.records {
                w.blob(&record.encode_to_vec());
            }
            w.u32(o.pages.len() as u32);
            for page in o.pages {
                w.u32(page);
            }
            w.u8(o.has_slot as u8)
                .u32(o.slot)
                .u8(o.channel.is_some() as u8);
            if let Some(channel) = o.channel {
                w.blob(&channel.encode_to_vec());
            }
            w.u64(o.page_bytes)
                .u64(o.row_bytes)
                .u8(o.host as u8)
                .u64(o.chain_first);
        }
        Msg::Window(o) => {
            w.u64(o.xfer).u8(o.spec.is_some() as u8);
            if let Some(spec) = o.spec {
                w.blob(&spec.encode_to_vec());
            }
            write_sums(&mut w, o.sums);
        }
        Msg::WindowDone {
            xfer,
            index,
            ok,
            error,
            sums,
        } => {
            w.u64(xfer).u32(index).u8(ok as u8).text(&error);
            write_sums(&mut w, sums);
        }
        Msg::Done {
            xfer,
            ok,
            error,
            quarantined,
        } => {
            w.u64(xfer).u8(ok as u8).text(&error).u8(quarantined as u8);
        }
    }
    w.finish()
}

fn write_sums(w: &mut Writer, sums: Option<SliceSums>) {
    w.u8(sums.is_some() as u8);
    if let Some(sums) = sums {
        w.u32(sums.pages.len() as u32);
        for sum in sums.pages {
            w.u64(sum);
        }
        w.u8(sums.row.is_some() as u8);
        if let Some(row) = sums.row {
            w.u64(row);
        }
    }
}

fn read_sums(r: &mut Reader<'_>) -> Result<Option<SliceSums>> {
    if !r.boolean()? {
        return Ok(None);
    }
    let count = r.count(8)?;
    let pages = (0..count).map(|_| r.u64()).collect::<Result<_>>()?;
    let row = if r.boolean()? { Some(r.u64()?) } else { None };
    Ok(Some(SliceSums { pages, row }))
}

fn decode(tag: u8, bytes: &[u8]) -> Result<Msg> {
    if bytes.len() > 8 << 20 {
        return Err(Error::new("a rank link frame is too large"));
    }
    let mut r = Reader(bytes);
    let message = match tag {
        12 => Msg::Hello {
            rank: r.u32()?,
            world: r.u32()?,
            rank0_nonce: r.blob()?.to_vec(),
            layout: r.blob()?.to_vec(),
        },
        13 => Msg::Bind {
            rank_secret: r.blob()?.to_vec(),
            instance_id: r.blob()?.to_vec(),
        },
        14 => {
            let mut o = Open {
                xfer: r.u64()?,
                manager_incarnation: r.u64()?,
                generation: r.u64()?,
                out: r.boolean()?,
                kind: r.u32()? as i32,
                fence_id: r.u64()?,
                ..Default::default()
            };
            let count = r.count(4)?;
            for _ in 0..count {
                o.records
                    .push(wire::Record::decode(r.blob()?).map_err(|e| Error::new(e.to_string()))?);
            }
            let count = r.count(4)?;
            o.pages = (0..count).map(|_| r.u32()).collect::<Result<_>>()?;
            o.has_slot = r.boolean()?;
            o.slot = r.u32()?;
            if r.boolean()? {
                o.channel = Some(
                    wire::DataChannelRef::decode(r.blob()?)
                        .map_err(|e| Error::new(e.to_string()))?,
                );
            }
            o.page_bytes = r.u64()?;
            o.row_bytes = r.u64()?;
            o.host = r.boolean()?;
            o.chain_first = r.u64()?;
            Msg::Open(o)
        }
        15 => Msg::Window(Window {
            xfer: r.u64()?,
            spec: if r.boolean()? {
                Some(wire::WindowSpec::decode(r.blob()?).map_err(|e| Error::new(e.to_string()))?)
            } else {
                None
            },
            sums: read_sums(&mut r)?,
        }),
        16 => Msg::Cancel { xfer: r.u64()? },
        17 => Msg::Close,
        18 => Msg::AgentHello {
            rank: r.u32()?,
            world: r.u32()?,
            agent_nonce: r.blob()?.to_vec(),
            layout: r.blob()?.to_vec(),
            gpu_uuid: r.blob()?.to_vec(),
        },
        19 => Msg::Attached {
            manager_incarnation: r.u64()?,
            generation: r.u64()?,
            mode: r.u32()? as i32,
        },
        20 => Msg::Detached {
            reason: r.text()?.into(),
        },
        21 => Msg::WindowDone {
            xfer: r.u64()?,
            index: r.u32()?,
            ok: r.boolean()?,
            error: r.text()?.into(),
            sums: read_sums(&mut r)?,
        },
        22 => Msg::Done {
            xfer: r.u64()?,
            ok: r.boolean()?,
            error: r.text()?.into(),
            quarantined: r.boolean()?,
        },
        23 => Msg::Disabled {
            reason: r.text()?.into(),
        },
        _ => return Err(Error::new("unexpected pool link tag")),
    };
    r.end()?;
    Ok(message)
}

#[derive(Debug)]
pub(crate) enum Incoming {
    Message(Msg),
    Closed(String),
}

impl Link {
    pub(crate) fn send_rank(&mut self, message: Msg) -> bool {
        self.enqueue(&encode(message)).is_ok()
    }

    pub(crate) fn poll_rank(&mut self, out: &mut Vec<Incoming>) {
        loop {
            let result = match self.receive() {
                Ok(Some((tag, bytes))) => decode(tag, &bytes),
                Ok(None) => return,
                Err(error) => Err(error),
            };
            match result {
                Ok(message) => out.push(Incoming::Message(message)),
                Err(error) => {
                    self.fail(error.clone());
                    out.push(Incoming::Closed(error.to_string()));
                    return;
                }
            }
        }
    }
}

/// Connected production links on loopback, for tests.
#[cfg(any(test, feature = "test-support"))]
pub fn memory_links() -> (Link, Link) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let a = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (b, _) = listener.accept().unwrap();
    (Link::new(a).unwrap(), Link::new(b).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn wait(end: &mut Link, want: usize) -> Vec<Incoming> {
        let mut got = Vec::new();
        for _ in 0..200 {
            end.poll_rank(&mut got);
            if got.len() >= want {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        got
    }

    #[test]
    fn every_message_preserves_its_fields_and_rejects_truncation_and_trailing_bytes() {
        let sums = SliceSums {
            pages: vec![17, 19],
            row: Some(23),
        };
        let messages = vec![
            Msg::Hello {
                rank: 1,
                world: 2,
                rank0_nonce: vec![3; 16],
                layout: vec![4; 16],
            },
            Msg::Bind {
                rank_secret: vec![5; 32],
                instance_id: vec![6; 16],
            },
            Msg::AgentHello {
                rank: 1,
                world: 2,
                agent_nonce: vec![7; 16],
                layout: vec![4; 16],
                gpu_uuid: vec![8; 16],
            },
            Msg::Open(Open {
                xfer: 9,
                manager_incarnation: 10,
                generation: 11,
                out: true,
                kind: wire::rank_fence::Kind::Put as i32,
                fence_id: 12,
                records: vec![wire::Record {
                    key: Some(wire::PrefixKey {
                        consumed_tokens: 64,
                        digest: vec![13; 16],
                    }),
                    first_page: 2,
                    pages: 3,
                    row: true,
                }],
                pages: vec![14, 15],
                has_slot: true,
                slot: 16,
                channel: Some(wire::DataChannelRef {
                    transport: wire::DataTransport::Tcp as i32,
                    address: "localhost:17".into(),
                    token: vec![18; 32],
                    ..Default::default()
                }),
                page_bytes: 4096,
                row_bytes: 8192,
                host: true,
                chain_first: 18,
            }),
            Msg::Open(Open::default()),
            Msg::Window(Window {
                xfer: 19,
                spec: Some(wire::WindowSpec {
                    index: 20,
                    region: wire::Region::Staging as i32,
                    offset: 21,
                    length: 22,
                    record: 23,
                    first_page: 24,
                    page_count: 2,
                    row: true,
                }),
                sums: Some(sums.clone()),
            }),
            Msg::Window(Window {
                xfer: 25,
                spec: None,
                sums: None,
            }),
            Msg::Attached {
                manager_incarnation: 26,
                generation: 27,
                mode: wire::RankMode::RankChannel as i32,
            },
            Msg::Detached {
                reason: "detached".into(),
            },
            Msg::WindowDone {
                xfer: 28,
                index: 29,
                ok: true,
                error: String::new(),
                sums: Some(sums),
            },
            Msg::WindowDone {
                xfer: 30,
                index: 31,
                ok: false,
                error: "failed".into(),
                sums: Some(SliceSums {
                    pages: vec![32],
                    row: None,
                }),
            },
            Msg::Done {
                xfer: 33,
                ok: false,
                error: "undrained".into(),
                quarantined: true,
            },
            Msg::Disabled {
                reason: "disabled".into(),
            },
            Msg::Cancel { xfer: 34 },
            Msg::Close,
        ];
        for message in messages {
            let frame = encode(message.clone());
            let tag = frame[0];
            let payload = &frame[5..];
            let decoded = decode(tag, payload).unwrap();
            assert_eq!(decoded, message);
            for end in 0..payload.len() {
                assert!(decode(tag, &payload[..end]).is_err());
            }
            let mut trailing = payload.to_vec();
            trailing.push(0);
            assert!(decode(tag, &trailing).is_err());
        }
        assert!(decode(255, &[]).is_err());
        let mut bad = encode(Msg::WindowDone {
            xfer: 1,
            index: 2,
            ok: false,
            error: String::new(),
            sums: None,
        });
        bad[5 + 8 + 4] = 2;
        assert!(decode(bad[0], &bad[5..]).is_err());
        let mut bad = encode(Msg::Hello {
            rank: 1,
            world: 2,
            rank0_nonce: vec![],
            layout: vec![],
        });
        bad[5 + 8..5 + 12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(bad[0], &bad[5..]).is_err());
    }

    #[test]
    fn frames_round_trip_in_order_and_bad_frames_fail_the_link() {
        let (mut a, mut b) = memory_links();
        for xfer in 0..5 {
            assert!(a.send_rank(Msg::Cancel { xfer }));
        }
        let got = wait(&mut b, 5);
        let xfers: Vec<u64> = got
            .iter()
            .map(|m| match m {
                Incoming::Message(Msg::Cancel { xfer }) => *xfer,
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(xfers, [0, 1, 2, 3, 4]);
        let mut bad = Writer::new(12);
        bad.bytes(&[0xFF, 0xFF, 0xFF]);
        a.enqueue(&bad.finish()).unwrap();
        assert!(matches!(wait(&mut b, 1).as_slice(), [Incoming::Closed(_)]));
        assert!(b.failed());
        assert!(!b.send_rank(Msg::Close));
    }

    #[test]
    fn a_closed_peer_fails_the_link_and_sends() {
        let (mut a, b) = memory_links();
        drop(b);
        assert!(matches!(wait(&mut a, 1).as_slice(), [Incoming::Closed(_)]));
        assert!(a.failed());
        assert!(!a.send_rank(Msg::Close));
    }
}
