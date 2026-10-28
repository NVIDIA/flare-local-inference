use super::*;
use crate::{
    CacheGeometry, CheckpointId, History, HostStore, InboundTicket, PageId, PublishedCheckpoint,
    StateSlotId, StoreEvent, StoreOutcome,
    offload::{self, CopierFactory, PrefixCacheOffload, Preparation},
    testing::{GpuMemory, IdentityRegistrar, SimulatedGpu, TestHistory, geometry},
};
use std::sync::{Arc, Mutex, mpsc};

#[test]
fn service_shutdown_does_not_join_a_thread_that_has_not_finished() {
    let (release, blocked) = mpsc::channel();
    let (finished, outcome) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        blocked.recv().unwrap();
        let _ = finished.send(Ok(()));
        Ok(())
    });
    let service = Service {
        thread: Some(thread),
        finished: Some(outcome),
        pool: None,
        summary: String::new(),
    };
    let (closed, result) = mpsc::channel();
    let shutdown = std::thread::spawn(move || {
        closed
            .send(service.join_timeout(Duration::from_millis(10)))
            .unwrap();
    });
    let result = result.recv_timeout(Duration::from_secs(1));
    release.send(()).unwrap();
    shutdown.join().unwrap();
    assert!(result.unwrap().unwrap_err().to_string().contains("in time"));
}

#[derive(Default)]
struct Copiers(Mutex<std::collections::HashMap<usize, GpuMemory>>);
impl CopierFactory for Copiers {
    fn mapped(
        &self,
        contexts: &[(usize, i32)],
        geometry: Vec<CacheGeometry>,
        _: &crate::slab::SlabPool,
    ) -> Result<crate::SharedCopier> {
        let (copier, memory) = SimulatedGpu::shared(&geometry)?;
        let rank = contexts[0].0;
        for page in 1..=2 {
            memory
                .pages
                .lock()
                .unwrap()
                .insert(page, vec![rank as u8 + page as u8; 4096]);
        }
        memory
            .rows
            .lock()
            .unwrap()
            .insert(1, vec![rank as u8 + 10; 4096]);
        self.0.lock().unwrap().insert(rank, memory);
        Ok(copier)
    }
    fn unmapped(
        &self,
        _: &[(usize, i32)],
        _: Vec<CacheGeometry>,
    ) -> Result<Box<dyn crate::PageCopier>> {
        panic!("this test has no manager pool")
    }
}

fn completed(store: &mut dyn HostStore) -> StoreEvent {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        if !events.is_empty() {
            assert_eq!(events.len(), 1);
            return events.remove(0);
        }
        assert!(Instant::now() < deadline, "record transfer stalled");
        std::thread::yield_now();
    }
}

#[test]
fn sharded_export_and_restore_use_each_ranks_real_record_file() {
    let directory = crate::testing::test_directory("sharded", "real-records").unwrap();
    let factory = Arc::new(Copiers::default());
    let local = geometry(64, &[("kv", 4096)], 4096);
    let prepare = |rank| {
        let options = PrefixCacheOffload {
            disk: Some(directory.join(format!("rank-{rank}"))),
            disk_limit: Some(1 << 20),
            ..Default::default()
        };
        let preparation = offload::prepare_with_inputs(
            &options,
            &directory,
            [3; 32],
            vec![local.clone()],
            vec![(rank, 0)],
            0,
            Some((rank, 2)),
            &IdentityRegistrar,
        )
        .unwrap();
        let Preparation::Store(_) = &preparation else {
            panic!("no disk store");
        };
        (
            preparation.into_shared().unwrap(),
            prepared_directory(&options, rank, &local),
        )
    };
    let (zero, zero_directory) = prepare(0);
    let (peer, peer_directory) = prepare(1);
    let (offer_tx, offer_rx) = mpsc::channel();
    let (reply_tx, reply_rx) = mpsc::channel();
    let peer_factory = factory.clone();
    let service = std::thread::spawn(move || {
        join_peer(Some(&peer), 1, peer_factory.as_ref(), |record| {
            reply_tx.send(record.to_vec()).unwrap();
            let offer = offer_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            Ok(vec![offer, record.to_vec()])
        })
        .unwrap()
        .unwrap()
    });
    let address = "127.0.0.1:1".parse().unwrap();
    join_rank_zero(Some(&zero), &[(address, address)], |mut offers| {
        offer_tx.send(offers.remove(1)).unwrap();
        Ok(vec![
            vec![],
            reply_rx.recv_timeout(Duration::from_secs(10)).unwrap(),
        ])
    })
    .unwrap();
    let service = service.join().unwrap();
    let mut store = offload::open(zero.take().unwrap(), factory.as_ref()).unwrap();
    let mut history = TestHistory::new([3; 32]);
    history.append(&[7; 129], &[]).unwrap();
    let key = history.key_at(128).unwrap();
    let checkpoint = PublishedCheckpoint {
        id: CheckpointId::new(1),
        key,
        parent: None,
        bucket: key.digest,
        chain_pages: vec![PageId(1), PageId(2)],
        tail_pages: 2,
        slot: StateSlotId(1),
    };
    assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
    assert!(matches!(
        completed(store.as_mut()),
        StoreEvent::Stored { result: Ok(_), .. }
    ));
    for directory in [&zero_directory, &peer_directory] {
        assert!(directory.join(record::file_name(&key)).is_file());
    }
    let hit = store.lookup(&history, 0, u64::MAX).unwrap();
    store
        .fetch(
            hit,
            InboundTicket {
                key,
                pages: vec![PageId(3), PageId(4)],
                slot: StateSlotId(2),
            },
        )
        .unwrap();
    assert!(matches!(
        completed(store.as_mut()),
        StoreEvent::Fetched { result: Ok(_), .. }
    ));
    for (rank, memory) in factory.0.lock().unwrap().iter() {
        let pages = memory.pages.lock().unwrap();
        assert_eq!(pages[&3], vec![*rank as u8 + 1; 4096]);
        assert_eq!(pages[&4], vec![*rank as u8 + 2; 4096]);
        assert_eq!(
            memory.rows.lock().unwrap()[&2],
            vec![*rank as u8 + 10; 4096]
        );
    }
    store.close().unwrap();
    drop(store);
    service.join().unwrap();
    drop(zero);
    std::fs::remove_dir_all(directory).unwrap();
}

fn prepared_directory(
    options: &PrefixCacheOffload,
    rank: usize,
    local: &CacheGeometry,
) -> std::path::PathBuf {
    options.disk.as_ref().unwrap().join(record::directory_name(
        &[3; 32],
        &crate::geometry::shard_fingerprint(local, rank, 2),
    ))
}
