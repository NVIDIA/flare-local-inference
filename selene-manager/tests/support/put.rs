use super::{
    channel::{self, Security},
    content_for, descriptor,
    pool::{Damage, PutResult, Session},
    store_client::{BUCKET, CancelAt, PutOptions},
};
use flare_selene::shared::frame::{ToInstance, ToManager};
use selene_manager::{
    Key,
    pool::key_to,
    proto::v2::{self, client_message as c, server_message::M},
    record::Descriptor,
};
use std::time::Duration;
use tokio::io::AsyncWriteExt;

pub enum Outcome {
    Stored { payload: Vec<u8>, bytes: u64 },
    Refused(v2::PutRefused),
    Cancelled,
    Abandoned,
}

pub async fn put_record(
    session: &mut Session,
    security: Option<&Security>,
    id: u64,
    key: Key,
    parent: Option<Key>,
    damage: Damage,
) -> PutResult {
    let options = PutOptions {
        corrupt_page: damage.corrupt_page,
        wrong_key: damage.wrong_key,
        cancel: damage.cancel_after_accept.then_some(CancelAt::Accept),
        keep_windows: true,
        ..Default::default()
    };
    match put(session, &descriptor(), id, key, parent, options, security).await {
        Outcome::Stored { bytes, .. } => PutResult::Stored(bytes),
        Outcome::Refused(refused) => PutResult::Refused(refused),
        Outcome::Cancelled => PutResult::Refused(v2::PutRefused {
            put_id: id,
            reason: 0,
            detail: "cancelled by the test".into(),
        }),
        Outcome::Abandoned => unreachable!(),
    }
}

async fn fence(session: &Session, id: u64) {
    session
        .send(c::M::FenceAck(v2::FenceAck {
            subject: Some(v2::fence_ack::Subject::PutId(id)),
            outcome: v2::FenceOutcome::Drained as i32,
        }))
        .await;
}

async fn cancel(session: &Session, id: u64, keep_windows: bool) -> Outcome {
    session
        .send(c::M::PutCancel(v2::PutCancel { put_id: id }))
        .await;
    if !keep_windows {
        fence(session, id).await;
    }
    Outcome::Cancelled
}

pub async fn put(
    session: &mut Session,
    descriptor: &Descriptor,
    id: u64,
    key: Key,
    parent: Option<Key>,
    options: PutOptions,
    security: Option<&Security>,
) -> Outcome {
    let chain_pages = key.consumed_tokens / descriptor.page_tokens;
    session
        .send(c::M::PutBegin(v2::PutBegin {
            put_id: id,
            key: Some(key_to(key)),
            chain_pages,
            row: true,
            has_published_parent: parent.is_some(),
            published_parent: parent.map(key_to),
            bucket: BUCKET.to_vec(),
            published_tail_pages: chain_pages
                - parent.map_or(0, |p| p.consumed_tokens / descriptor.page_tokens),
            mode: if security.is_some() {
                v2::CopyMode::DataChannel
            } else {
                v2::CopyMode::SharedWindows
            } as i32,
        }))
        .await;
    enum Next {
        Accept(v2::PutAccept),
        Grant(v2::WindowSpec),
        Stored(u64),
        Refused(v2::PutRefused),
    }
    let mut pick = |m: &M| match m {
        M::PutAccept(a) if a.put_id == id => Some(Next::Accept(a.clone())),
        M::PutGrant(g) if g.put_id == id => Some(Next::Grant(g.window.unwrap())),
        M::Stored(s) if s.put_id == id => Some(Next::Stored(s.bytes)),
        M::PutRefused(r) if r.put_id == id => Some(Next::Refused(r.clone())),
        _ => None,
    };
    let accept = match session
        .take_within(Duration::from_secs(30), &mut pick)
        .await
        .expect("put answer")
    {
        Next::Accept(a) => a,
        Next::Refused(r) => return Outcome::Refused(r),
        _ => panic!("expected admission verdict"),
    };
    if options.cancel == Some(CancelAt::Accept) {
        return cancel(session, id, options.keep_windows).await;
    }
    let mut conn = match security {
        Some(security) => Some(channel::dial(accept.channel.as_ref().unwrap(), security).await),
        None => None,
    };
    let first = accept.first_page as u32;
    let parent = accept.has_parent.then(|| {
        let p = accept.parent.as_ref().unwrap();
        Key {
            consumed_tokens: p.consumed_tokens,
            digest: p.digest.as_slice().try_into().unwrap(),
        }
    });
    let (header, mut bytes, _) = content_for(descriptor, key, parent, first);
    let payload = header.length_with_row() - header.page_segment;
    let page_bytes = header.page_record_bytes() as usize;
    let mut covered = 0;
    let mut grants = 0;
    let mut committed = false;
    loop {
        let next = if let Some(conn) = conn.as_mut().filter(|_| covered < payload) {
            match tokio::time::timeout(
                Duration::from_secs(30),
                ToInstance::read(conn, channel::MAX_PAGES),
            )
            .await
            {
                Ok(Ok(ToInstance::Grant(shape))) => Next::Grant(v2::WindowSpec {
                    index: shape.index,
                    first_page: shape.first_page,
                    page_count: shape.page_count,
                    row: shape.row,
                    length: shape.length,
                    ..Default::default()
                }),
                _ => session
                    .take_within(Duration::from_secs(30), &mut pick)
                    .await
                    .expect("put answer"),
            }
        } else {
            session
                .take_within(Duration::from_secs(30), &mut pick)
                .await
                .expect("put answer")
        };
        let window = match next {
            Next::Stored(stored) => {
                return Outcome::Stored {
                    payload: bytes[header.page_segment as usize..].to_vec(),
                    bytes: stored,
                };
            }
            Next::Refused(r) => {
                if !options.keep_windows {
                    fence(session, id).await;
                }
                return Outcome::Refused(r);
            }
            Next::Grant(window) => window,
            Next::Accept(_) => panic!("duplicate accept"),
        };
        grants += 1;
        if options.cancel == Some(CancelAt::Grant) {
            return cancel(session, id, options.keep_windows).await;
        }
        let start = if window.page_count > 0 {
            header.page_segment as usize + (window.first_page - first) as usize * page_bytes
        } else {
            header.row_segment as usize
        };
        let mut hashes = Vec::new();
        for position in 0..window.page_count as usize {
            let relative = (window.first_page - first) as usize + position;
            let page = header.page_segment as usize + relative * page_bytes;
            if options.corrupt_page == Some(relative) {
                bytes[page] ^= 1;
            }
            hashes.push(flare_selene::shared::record::checksum(
                &bytes[page..page + page_bytes],
            ));
        }
        let row = window.row.then(|| {
            if options.corrupt_row {
                bytes[header.row_segment as usize] ^= 1;
            }
            flare_selene::shared::record::checksum(&bytes[header.row_segment as usize..])
        });
        let data = &bytes[start..start + window.length as usize];
        if let Some(conn) = &mut conn {
            ToManager::Filled {
                index: window.index,
                page_xxh3: hashes,
                row_xxh3: row,
            }
            .write(conn)
            .await
            .unwrap();
            conn.write_all(data).await.unwrap();
        } else {
            session.shm.as_ref().unwrap().write(&window, data);
            if !(options.skip_filled && grants == 1) {
                session
                    .send(c::M::PutFilled(v2::PutFilled {
                        put_id: id,
                        index: window.index,
                        page_xxh3: hashes,
                        row_xxh3: row.unwrap_or(0),
                        has_row: window.row,
                        ..Default::default()
                    }))
                    .await;
            }
        }
        covered += window.length;
        if options
            .cancel_after_fills
            .is_some_and(|fills| grants >= fills)
        {
            return cancel(session, id, options.keep_windows).await;
        }
        if !committed && (covered >= payload || options.early_commit || options.skip_filled) {
            if options.cancel == Some(CancelAt::Filled) {
                return cancel(session, id, options.keep_windows).await;
            }
            let end = if options.early_commit {
                4096
            } else {
                header.page_segment as usize
            };
            let mut metadata = bytes[..end].to_vec();
            if options.wrong_key {
                metadata[40] ^= 0xff;
            }
            session
                .send(c::M::PutCommit(v2::PutCommit {
                    put_id: id,
                    metadata_offset: 0,
                    metadata,
                }))
                .await;
            committed = true;
            if options.abort_after_commit {
                tokio::time::sleep(Duration::from_millis(300)).await;
                return Outcome::Abandoned;
            }
            if options.cancel == Some(CancelAt::Commit) {
                session
                    .send(c::M::PutCancel(v2::PutCancel { put_id: id }))
                    .await;
            }
        }
    }
}
