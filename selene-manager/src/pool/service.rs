//! The gRPC `Pool` service: `Hello`, and one `Session` stream per instance process.

use super::core::{Event, OpenPeer, PoolHandle};
use crate::proto::v2::{
    self, ClientMessage, HelloReply, HelloRequest, ServerMessage, pool_server::Pool,
};
use std::pin::Pin;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tonic::{Request, Response, Status, Streaming, transport::server::UdsConnectInfo};

/// Outbound messages queued per session before it is considered stuck.
const OUTBOUND: usize = 4096;

type SessionStream = Pin<Box<dyn Stream<Item = Result<ServerMessage, Status>> + Send>>;
type RankStream = Pin<Box<dyn Stream<Item = Result<v2::RankServerMessage, Status>> + Send>>;

#[derive(Clone)]
pub struct PoolService {
    handle: PoolHandle,
}

impl PoolService {
    pub fn new(handle: PoolHandle) -> Self {
        Self { handle }
    }
    async fn forward<I: Send + 'static, O: Send + 'static>(
        &self,
        mut inbound: Streaming<I>,
        open: impl FnOnce(mpsc::Sender<O>, oneshot::Sender<Option<u64>>) -> Event + Send,
        message: fn(u64, I) -> Event,
        closed: fn(u64) -> Event,
    ) -> Result<Response<Pin<Box<dyn Stream<Item = Result<O, Status>> + Send>>>, Status> {
        let (tx, rx) = mpsc::channel(OUTBOUND);
        let (reply, answer) = oneshot::channel();
        let events = self.handle.events.clone();
        events
            .send(open(tx, reply))
            .await
            .map_err(|_| Status::unavailable("the pool is stopped"))?;
        if let Some(id) = answer
            .await
            .map_err(|_| Status::unavailable("the pool is stopped"))?
        {
            tokio::spawn(async move {
                while let Ok(Some(m)) = inbound.message().await {
                    if events.send(message(id, m)).await.is_err() {
                        return;
                    }
                }
                let _ = events.send(closed(id)).await;
            });
        }
        Ok(Response::new(Box::pin(
            ReceiverStream::new(rx).map(Ok::<_, Status>),
        )))
    }
}

#[tonic::async_trait]
impl Pool for PoolService {
    type SessionStream = SessionStream;
    type RankStream = RankStream;

    async fn rank(
        &self,
        request: Request<Streaming<v2::RankClientMessage>>,
    ) -> Result<Response<Self::RankStream>, Status> {
        let peer = peer(&request);
        let mut inbound = request.into_inner();
        let first = inbound
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("the stream ended before `attach`"))?;
        let Some(v2::rank_client_message::M::Attach(attach)) = first.m else {
            return Err(Status::invalid_argument(
                "the first message must be `attach`",
            ));
        };
        self.forward(
            inbound,
            |tx, reply| Event::RankOpen {
                peer,
                attach: Box::new(attach),
                tx,
                reply,
            },
            |aid, message| Event::RankMessage {
                aid,
                message: Box::new(message),
            },
            |aid| Event::RankClosed { aid },
        )
        .await
    }

    async fn hello(&self, request: Request<HelloRequest>) -> Result<Response<HelloReply>, Status> {
        let shared = &self.handle.shared;
        let peer = peer(&request);
        let message = request.into_inner();
        if message.protocol != 2 {
            return Ok(Response::new(HelloReply {
                status: v2::HelloStatus::VersionMismatch as i32,
                protocol: 2,
                ..Default::default()
            }));
        }
        let same_host = peer.unix && peer.uid == shared.uid && peer.gid == shared.gid;
        let readiness = *shared.readiness.lock().unwrap();
        Ok(Response::new(HelloReply {
            status: v2::HelloStatus::HelloOk as i32,
            pool_name: shared.name.clone(),
            protocol: 2,
            manager_incarnation: shared.incarnation,
            encryption: shared.encryption as i32,
            readiness: Some(readiness),
            data_transports: shared.transports.iter().map(|t| *t as i32).collect(),
            limits: Some(shared.limits.to_wire()),
            shared_memory: same_host.then(|| shared.shm.clone()),
        }))
    }

    async fn session(
        &self,
        request: Request<Streaming<ClientMessage>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let peer = peer(&request);
        let mut inbound = request.into_inner();
        let first = inbound
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("the stream ended before `open`"))?;
        let Some(v2::client_message::M::Open(open)) = first.m else {
            return Err(Status::invalid_argument("the first message must be `open`"));
        };
        self.forward(
            inbound,
            |tx, reply| Event::Open {
                peer,
                open: Box::new(open),
                tx,
                reply,
            },
            |sid, message| Event::Message {
                sid,
                message: Box::new(message),
            },
            |sid| Event::Closed { sid },
        )
        .await
    }
}

fn peer<T>(request: &Request<T>) -> OpenPeer {
    let credentials = request
        .extensions()
        .get::<UdsConnectInfo>()
        .and_then(|info| info.peer_cred);
    match credentials {
        Some(c) => OpenPeer {
            unix: true,
            pid: c.pid().unwrap_or(0) as u32,
            uid: c.uid(),
            gid: c.gid(),
        },
        None => OpenPeer {
            unix: false,
            pid: 0,
            uid: 0,
            gid: 0,
        },
    }
}
