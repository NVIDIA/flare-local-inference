//! A scripted instance on the RDMA carrier: it registers staging, makes a queue pair for
//! each transfer and lends private slots to a claim, as the client does.

use flare_selene::pool::rdma;
use flare_selene::shared::frame::{Carrier, NO_SLOT, ToInstance, ToManager, Window};
use selene_manager::{arena::Arena, proto::v2};
use std::{sync::Arc, time::Duration};
use tokio::{net::TcpStream, time::timeout};

const MAX_PAGES: u32 = 64;
const SLOT: usize = 1 << 20;

pub struct Instance {
    staging: Arena,
    device: Arc<rdma::Device>,
}

pub struct Dialed {
    pub stream: TcpStream,
    pub pair: rdma::Pair,
}

pub struct Received {
    pub window: Window,
    pub payload: Vec<u8>,
}

impl Instance {
    pub fn new() -> Self {
        let staging = Arena::new(4 * SLOT).unwrap();
        let (base, len) = staging.mapping();
        // SAFETY: the arena stays mapped and in place for the life of the instance.
        let device = unsafe { rdma::Device::open(base, len) }.expect("an RDMA device");
        Self { staging, device }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn bytes(&self, offset: usize, length: usize) -> &mut [u8] {
        let (base, len) = self.staging.mapping();
        assert!(offset + length <= len);
        // SAFETY: inside the mapping; the test drives one transfer at a time.
        unsafe { std::slice::from_raw_parts_mut(base.add(offset), length) }
    }

    /// Connect, say hello for RDMA and bring the queue pair up.
    pub async fn dial(&self, reference: &v2::DataChannelRef) -> Dialed {
        use tokio::io::AsyncWriteExt;
        let mut stream = TcpStream::connect(&reference.address).await.unwrap();
        stream.write_all(&reference.token).await.unwrap();
        let pair = rdma::Pair::new(&self.device).unwrap();
        ToManager::Hello(Carrier::Rdma(pair.handshake().encode().to_vec()))
            .write(&mut stream)
            .await
            .unwrap();
        let ToInstance::Accept(theirs) = ToInstance::read(&mut stream, MAX_PAGES).await.unwrap()
        else {
            panic!("expected the manager's end of the pair");
        };
        pair.connect(&rdma::Handshake::decode(&theirs).unwrap())
            .unwrap();
        Dialed { stream, pair }
    }

    /// Lend two slots, read windows as they are named, and lend a slot back for each.
    pub async fn read_windows(&self, dialed: &mut Dialed) -> (Vec<Received>, bool) {
        let mut received = Vec::new();
        let mut next = 0usize;
        for _ in 0..2 {
            lend(&mut dialed.stream, next).await;
            next += 1;
        }
        loop {
            let frame = match timeout(
                Duration::from_secs(10),
                ToInstance::read(&mut dialed.stream, MAX_PAGES),
            )
            .await
            {
                Ok(Ok(frame)) => frame,
                _ => return (received, false),
            };
            match frame {
                ToInstance::Window(window) => {
                    assert_ne!(window.slot, NO_SLOT);
                    let payload = self
                        .bytes(window.slot as usize, window.shape.length as usize)
                        .to_vec();
                    received.push(Received { window, payload });
                    lend(&mut dialed.stream, next).await;
                    next += 1;
                }
                ToInstance::End => return (received, true),
                _ => return (received, false),
            }
        }
    }
}

async fn lend(stream: &mut TcpStream, n: usize) {
    ToManager::Credit {
        offset: ((n % 4) * SLOT) as u64,
        length: SLOT as u64,
    }
    .write(stream)
    .await
    .unwrap();
}
