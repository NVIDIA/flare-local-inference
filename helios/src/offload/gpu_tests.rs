//! Host-tier tests that need a CUDA device and an exported package.
//!
//! Selene's own tests cover the stores against copiers that never touch a
//! device; these run the stores over the real transfer threads. They are ignored
//! unless `FLARE_OFFLOAD_TEST_PACKAGE` names a package a free GPU 0 can load.

use flare_selene::{
    HostStore, StoreEvent,
    testing::{Ids, checkpoint, pattern as fill, test_directory},
};

fn seed(plane: usize, unit: u32) -> u64 {
    (plane as u64 + 1) * 1_000_003 + u64::from(unit)
}

fn load_test_model(package: &std::ffi::OsStr) -> crate::runtime_package::NativeModel {
    unsafe {
        crate::runtime_package::NativeModel::load(
            crate::runtime_package::test_support::authorize(std::path::Path::new(package)),
            &[0],
            &crate::runtime_package::LoadOptions::default(),
            &crate::startup::StartupProgress::silent(),
        )
    }
    .unwrap()
}

fn wait_event(store: &mut dyn HostStore, seconds: u64) -> StoreEvent {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    let mut events = Vec::new();
    while events.is_empty() {
        assert!(
            std::time::Instant::now() < deadline,
            "no store event within {seconds} s"
        );
        store.poll(&mut events).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(events.len(), 1, "{events:?}");
    events.pop().unwrap()
}

mod memory {
    use super::{fill, seed};
    use std::sync::{Arc, Mutex};

    use flare_selene::{
        CheckpointId, HostStore, InboundTicket, Layout, PageId, PrefixKey, PublishedCheckpoint,
        StateSlotId, StoreEvent, StoreOutcome, Tier, memory::MemoryStore, slab::SlabPool,
    };

    use crate::{
        cache::PrefixHistory,
        offload::{registrar::CudaRegistrar, transfer::Transfers},
    };

    const IDENTITY: [u8; 32] = [7; 32];

    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn memory_store_round_trips_a_multi_window_checkpoint() {
        use crate::cuda::Stream;
        use flare_selene::CacheGeometry;

        let Some(package) = std::env::var_os("FLARE_OFFLOAD_TEST_PACKAGE") else {
            eprintln!("skipping: FLARE_OFFLOAD_TEST_PACKAGE is unset");
            return;
        };
        let model = super::load_test_model(&package);
        let manifest = model.manifests().unwrap().remove(0);
        let geometry = vec![manifest.cache.clone().unwrap()];
        let rank: &CacheGeometry = &geometry[0];
        let (context, device) = (manifest.context, manifest.device);
        let layout = Layout::new(&geometry).unwrap();
        let pt = layout.page_tokens;
        let pages = 99u32;
        assert!(rank.page_capacity >= 2 * u64::from(pages) + 400 && rank.row_capacity >= 4);
        let sources: Vec<u32> = (1..=pages).collect();
        let targets: Vec<u32> = (300..300 + pages).collect();
        let (source_slot, target_slot) = (1u32, 2u32);

        let row_plane = rank.planes.len();
        let stream = Stream::open(context, device).unwrap();
        let largest = rank
            .planes
            .iter()
            .chain([&rank.row])
            .map(|plane| plane.unit_bytes as usize)
            .max()
            .unwrap();
        let mut staging = stream.host_buffer(largest).unwrap();
        let mut write_unit = |address: u64, bytes: usize, seed: u64| {
            fill(&mut staging.bytes_mut()[..bytes], seed);
            unsafe { stream.upload(address as usize, &staging, 0, bytes) }.unwrap();
            stream.synchronize().unwrap();
        };
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for &id in &sources {
                write_unit(
                    geometry.unit_address(u64::from(id)),
                    geometry.unit_bytes as usize,
                    seed(plane, id),
                );
            }
        }
        write_unit(
            rank.row.unit_address(u64::from(source_slot)),
            rank.row.unit_bytes as usize,
            seed(row_plane, source_slot),
        );

        let mut pool = SlabPool::sized(&layout, SlabPool::minimum_bytes(&layout)).unwrap();
        pool.register(&CudaRegistrar, context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
        ));
        let mut store = MemoryStore::new(IDENTITY, pool, transfers.clone()).unwrap();
        let consumed = u64::from(pages) * pt;
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..consumed as u32 + 40)
            .map(|t| t.wrapping_mul(2654435761))
            .collect();
        history.append(&tokens, &[]).unwrap();
        let checkpoint = PublishedCheckpoint {
            id: CheckpointId::new(1),
            key: history.key_at(consumed).unwrap(),
            parent: None,
            bucket: history.bucket_at(consumed).unwrap(),
            chain_pages: sources.iter().map(|id| PageId(*id)).collect(),
            tail_pages: pages as usize,
            slot: StateSlotId(source_slot),
        };

        assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
        match super::wait_event(&mut store, 60) {
            StoreEvent::Stored { result, .. } => {
                assert_eq!(
                    result.unwrap(),
                    u64::from(pages) * rank.page_bytes() + rank.row.unit_bytes
                );
            }
            other => panic!("unexpected event {other:?}"),
        }
        let hit = store.lookup(&history, 0, u64::MAX).unwrap();
        assert_eq!(hit.pages, pages as usize);
        let ticket = InboundTicket {
            key: checkpoint.key,
            pages: targets.iter().map(|id| PageId(*id)).collect(),
            slot: StateSlotId(target_slot),
        };
        store.fetch(hit, ticket).unwrap();
        match super::wait_event(&mut store, 60) {
            StoreEvent::Fetched { result, .. } => {
                assert_eq!(
                    result.unwrap(),
                    u64::from(pages) * rank.page_bytes() + rank.row.unit_bytes
                );
            }
            other => panic!("unexpected event {other:?}"),
        }
        let read_unit = |address: u64, bytes: usize| -> Vec<u8> {
            unsafe { stream.download(&staging, address as usize, bytes) }.unwrap();
            stream.synchronize().unwrap();
            staging.bytes()[..bytes].to_vec()
        };
        let mut mismatches = Vec::new();
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for (source, target) in sources.iter().zip(&targets) {
                let mut wanted = vec![0u8; geometry.unit_bytes as usize];
                fill(&mut wanted, seed(plane, *source));
                let actual = read_unit(
                    geometry.unit_address(u64::from(*target)),
                    geometry.unit_bytes as usize,
                );
                if actual != wanted {
                    mismatches.push((geometry.name.clone(), *source, *target));
                }
            }
        }
        let mut wanted = vec![0u8; rank.row.unit_bytes as usize];
        fill(&mut wanted, seed(row_plane, source_slot));
        let actual = read_unit(
            rank.row.unit_address(u64::from(target_slot)),
            rank.row.unit_bytes as usize,
        );
        assert!(
            actual == wanted,
            "row {target_slot} differs from row {source_slot}"
        );
        assert!(
            mismatches.is_empty(),
            "mismatched planes and pages: {mismatches:?}"
        );
        store.close().unwrap();
    }

    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn memory_store_round_trips_a_checkpoint_through_a_loaded_package() {
        use crate::cuda::Stream;
        use flare_selene::CacheGeometry;

        let Some(package) = std::env::var_os("FLARE_OFFLOAD_TEST_PACKAGE") else {
            eprintln!("skipping: FLARE_OFFLOAD_TEST_PACKAGE is unset");
            return;
        };
        let model = super::load_test_model(&package);
        let manifest = model.manifests().unwrap().remove(0);
        let geometry = vec![manifest.cache.clone().unwrap()];
        let rank: &CacheGeometry = &geometry[0];
        let (context, device) = (manifest.context, manifest.device);
        let layout = Layout::new(&geometry).unwrap();
        let pt = layout.page_tokens;
        assert!(rank.page_capacity >= 16 && rank.row_capacity >= 4);
        let capacity = rank.page_capacity as u32;
        let sources = [1u32, 7, capacity / 2];
        let targets = [2u32, 9, capacity / 2 + 1];
        let (source_slot, target_slot) = (1u32, 2u32);

        let row_plane = rank.planes.len();

        let stream = Stream::open(context, device).unwrap();
        let largest = rank
            .planes
            .iter()
            .chain([&rank.row])
            .map(|plane| plane.unit_bytes as usize)
            .max()
            .unwrap();
        let mut staging = stream.host_buffer(largest).unwrap();
        let mut write_unit = |address: u64, bytes: usize, seed: u64| {
            fill(&mut staging.bytes_mut()[..bytes], seed);
            unsafe { stream.upload(address as usize, &staging, 0, bytes) }.unwrap();
            stream.synchronize().unwrap();
        };
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for id in sources {
                write_unit(
                    geometry.unit_address(u64::from(id)),
                    geometry.unit_bytes as usize,
                    seed(plane, id),
                );
            }
        }
        write_unit(
            rank.row.unit_address(u64::from(source_slot)),
            rank.row.unit_bytes as usize,
            seed(row_plane, source_slot),
        );
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for id in targets {
                write_unit(
                    geometry.unit_address(u64::from(id)),
                    geometry.unit_bytes as usize,
                    0xDD + plane as u64,
                );
            }
        }
        write_unit(
            rank.row.unit_address(u64::from(target_slot)),
            rank.row.unit_bytes as usize,
            0xDD,
        );

        let mut pool = SlabPool::sized(&layout, SlabPool::minimum_bytes(&layout)).unwrap();
        pool.register(&CudaRegistrar, context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
        ));
        let mut store = MemoryStore::new(IDENTITY, pool, transfers.clone()).unwrap();

        let consumed = 3 * pt + 5;
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..consumed as u32 + 40)
            .map(|t| t.wrapping_mul(2654435761))
            .collect();
        history.append(&tokens, &[]).unwrap();
        let checkpoint = PublishedCheckpoint {
            id: CheckpointId::new(1),
            key: history.key_at(consumed).unwrap(),
            parent: None,
            bucket: history.bucket_at(consumed).unwrap(),
            chain_pages: sources.iter().map(|id| PageId(*id)).collect(),
            tail_pages: 3,
            slot: StateSlotId(source_slot),
        };

        assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().outbound_in_flight, 1);
        match super::wait_event(&mut store, 60) {
            StoreEvent::Stored {
                checkpoint: id,
                key,
                result,
            } => {
                assert_eq!(id, checkpoint.id);
                assert_eq!(key, checkpoint.key);
                assert_eq!(result.unwrap(), 3 * rank.page_bytes() + rank.row.unit_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        let stats = store.stats();
        assert_eq!(stats.stored, 1);
        assert_eq!(stats.outbound_in_flight, 0);
        assert_eq!(stats.memory_records, 1);
        assert_eq!(stats.memory_bytes, layout.record_bytes(3));
        assert!(stats.outbound_rate > 0.0);

        // The record's slabs hold the source pages' and row's patterns under the layout.
        let check_record = |store: &MemoryStore, key: &PrefixKey, label: &str| {
            let record = store.record(key).unwrap().clone();
            assert_eq!(record.pages.len(), 3, "{label}");
            let pool = store.pool();
            for (slab, id) in record.pages.iter().zip(sources) {
                let bytes = pool.page(*slab);
                for (plane, geometry) in rank.planes.iter().enumerate() {
                    let start = layout.ranks[0].plane_offsets[plane] as usize;
                    let mut expected = vec![0u8; geometry.unit_bytes as usize];
                    fill(&mut expected, seed(plane, id));
                    assert!(
                        bytes[start..start + expected.len()] == expected[..],
                        "{label}: slab bytes of plane {} ({plane}) page {id} differ from the device",
                        geometry.name
                    );
                }
            }
            let row_start = layout.ranks[0].row_offset as usize;
            let mut expected = vec![0u8; rank.row.unit_bytes as usize];
            fill(&mut expected, seed(row_plane, source_slot));
            assert!(
                pool.row(record.row.unwrap())[row_start..row_start + expected.len()]
                    == expected[..],
                "{label}: row slab bytes differ from the device"
            );
        };
        check_record(&store, &checkpoint.key, "first record");

        // Poison the device copies, then restore the chain into other ids.
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for id in sources {
                write_unit(
                    geometry.unit_address(u64::from(id)),
                    geometry.unit_bytes as usize,
                    0xEE + plane as u64,
                );
            }
        }
        write_unit(
            rank.row.unit_address(u64::from(source_slot)),
            rank.row.unit_bytes as usize,
            0xEE,
        );
        let hit = store.lookup(&history, 0, u64::MAX).unwrap();
        assert_eq!(hit.key, checkpoint.key);
        assert_eq!(hit.pages, 3);
        assert_eq!(hit.bytes, layout.record_bytes(3));
        assert_eq!(hit.tier, Tier::Dram);
        assert!(hit.estimated_seconds > 0.0);
        let ticket = InboundTicket {
            key: checkpoint.key,
            pages: targets.iter().map(|id| PageId(*id)).collect(),
            slot: StateSlotId(target_slot),
        };
        store.fetch(hit, ticket).unwrap();
        assert_eq!(store.stats().inbound_in_flight, 1);
        match super::wait_event(&mut store, 60) {
            StoreEvent::Fetched { key, result } => {
                assert_eq!(key, checkpoint.key);
                assert_eq!(result.unwrap(), 3 * rank.page_bytes() + rank.row.unit_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        let stats = store.stats();
        assert_eq!(stats.fetched, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.reused_tokens, consumed);
        assert_eq!(stats.inbound_in_flight, 0);
        assert!(!store.index().node(&checkpoint.key).unwrap().pinned());

        let read_unit = |address: u64, bytes: usize| -> Vec<u8> {
            unsafe { stream.download(&staging, address as usize, bytes) }.unwrap();
            stream.synchronize().unwrap();
            staging.bytes()[..bytes].to_vec()
        };
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for (source, target) in sources.iter().zip(targets) {
                let mut wanted = vec![0u8; geometry.unit_bytes as usize];
                fill(&mut wanted, seed(plane, *source));
                let actual = read_unit(
                    geometry.unit_address(u64::from(target)),
                    geometry.unit_bytes as usize,
                );
                assert!(
                    actual == wanted[..],
                    "plane {} ({plane}) page {target} differs from page {source}",
                    geometry.name
                );
            }
        }
        let mut wanted = vec![0u8; rank.row.unit_bytes as usize];
        fill(&mut wanted, seed(row_plane, source_slot));
        let actual = read_unit(
            rank.row.unit_address(u64::from(target_slot)),
            rank.row.unit_bytes as usize,
        );
        assert!(
            actual == wanted[..],
            "row {target_slot} differs from row {source_slot}"
        );

        // The same key is already resident: no second copy-out.
        assert_ne!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 1);
        store.close().unwrap();
        assert_eq!(store.stats().memory_records, 1);

        // A pool holding one record evicts it when a second key is stored.
        let mut small = SlabPool::new(
            layout.page_record_bytes,
            3,
            layout.row_record_bytes,
            1,
            true,
        )
        .unwrap();
        small.register(&CudaRegistrar, context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &small).unwrap(),
        ));
        let mut store = MemoryStore::new(IDENTITY, small, transfers).unwrap();
        assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
        assert!(matches!(
            super::wait_event(&mut store, 60),
            StoreEvent::Stored { result: Ok(_), .. }
        ));
        assert_eq!(store.pool().free_pages(), 0);
        assert_eq!(store.pool().free_rows(), 0);
        let mut other = PrefixHistory::new(IDENTITY);
        other.append(&tokens[..consumed as usize], &[]).unwrap();
        other.append(&[11, 12, 13, 14, 15], &[]).unwrap();
        let second_consumed = consumed + 5;
        let second = PublishedCheckpoint {
            id: CheckpointId::new(2),
            key: other.key_at(second_consumed).unwrap(),
            parent: None,
            bucket: other.bucket_at(second_consumed).unwrap(),
            chain_pages: targets.iter().map(|id| PageId(*id)).collect(),
            tail_pages: 3,
            slot: StateSlotId(target_slot),
        };
        assert_ne!(second.key, checkpoint.key);
        assert_eq!(store.store(&second).unwrap(), StoreOutcome::Queued);
        assert!(!store.index().contains(&checkpoint.key));
        assert!(store.index().contains(&second.key));
        assert_eq!(store.pool().free_pages(), 0);
        match super::wait_event(&mut store, 60) {
            StoreEvent::Stored { key, result, .. } => {
                assert_eq!(key, second.key);
                result.unwrap();
            }
            other => panic!("unexpected event {other:?}"),
        }
        assert_eq!(store.stats().memory_records, 1);
        assert!(store.lookup(&history, 0, u64::MAX).is_none());
        assert_eq!(store.lookup(&other, 0, u64::MAX).unwrap().key, second.key);
        check_record(&store, &second.key, "second record");

        // Closing waits for a copy-out in progress: the record is resident afterwards.
        let mut another = PrefixHistory::new(IDENTITY);
        another.append(&tokens[..consumed as usize], &[]).unwrap();
        another.append(&[21, 22, 23], &[]).unwrap();
        let third_consumed = consumed + 3;
        let third = PublishedCheckpoint {
            id: CheckpointId::new(3),
            key: another.key_at(third_consumed).unwrap(),
            parent: None,
            bucket: another.bucket_at(third_consumed).unwrap(),
            chain_pages: targets.iter().map(|id| PageId(*id)).collect(),
            tail_pages: 3,
            slot: StateSlotId(target_slot),
        };
        assert_eq!(store.store(&third).unwrap(), StoreOutcome::Queued);
        assert!(!store.index().contains(&second.key));
        assert_eq!(store.stats().outbound_in_flight, 1);
        store.close().unwrap();
        let stats = store.stats();
        assert_eq!(stats.stored, 3);
        assert_eq!(stats.store_failed, 0);
        assert_eq!(stats.outbound_in_flight, 0);
        assert_eq!(stats.memory_records, 1);
        assert!(store.index().node(&third.key).unwrap().ready);
        check_record(&store, &third.key, "record stored across close");
        drop(store);
        drop(staging);
        drop(stream);
        drop(model);
    }
}

mod composite {
    use super::{Ids, checkpoint, fill, seed, test_directory};
    use std::{
        fs,
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use flare_selene::{
        CacheGeometry, HostStore, INITIAL_MEMORY_RATE, InboundTicket, Layout, PageId,
        PublishedCheckpoint, StateSlotId, StoreEvent, StoreOutcome, Tier,
        composite::CompositeStore,
        disk::DiskThread,
        disk_store::{DiskStore, Staged},
        memory::MemoryStore,
        record,
        slab::SlabPool,
    };

    use crate::{
        cache::PrefixHistory,
        offload::{registrar::CudaRegistrar, transfer::Transfers},
    };

    const IDENTITY: [u8; 32] = [0x3c; 32];

    fn poll_until(
        store: &mut CompositeStore,
        events: &mut Vec<StoreEvent>,
        mut done: impl FnMut(&CompositeStore) -> bool,
    ) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !done(store) {
            assert!(
                Instant::now() < deadline,
                "condition not reached within 60 s"
            );
            store.poll(events).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn gpu_transport(context: usize, device: i32, geometry: &[CacheGeometry]) -> Staged {
        let mut pool = SlabPool::staging(&Layout::new(geometry).unwrap()).unwrap();
        pool.register(&CudaRegistrar, context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.to_vec(), &pool).unwrap(),
        ));
        Staged::new(
            Some(pool),
            transfers,
            DiskThread::start("flare-offload-disk").unwrap(),
        )
    }

    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn composite_writes_a_child_published_while_its_parent_is_in_flight() {
        use crate::cuda::Stream;
        use flare_selene::geometry::fingerprint;
        let Some(package) = std::env::var_os("FLARE_OFFLOAD_TEST_PACKAGE") else {
            eprintln!("skipping: FLARE_OFFLOAD_TEST_PACKAGE is unset");
            return;
        };
        let Some(directory) = test_directory("composite", "gpu-child") else {
            return;
        };
        let model = super::load_test_model(&package);
        let manifest = model.manifests().unwrap().remove(0);
        let geometry = vec![manifest.cache.clone().unwrap()];
        let rank = geometry[0].clone();
        let (context, device) = (manifest.context, manifest.device);
        let layout = Layout::new(&geometry).unwrap();
        let fingerprint = fingerprint(&geometry);
        let page_tokens = rank.page_tokens;
        let stream = Stream::open(context, device).unwrap();
        let mut staging = stream.host_buffer(rank.row.unit_bytes as usize).unwrap();
        let mut write_unit = |address: u64, bytes: usize, seed: u64| {
            for (index, byte) in staging.bytes_mut()[..bytes].iter_mut().enumerate() {
                *byte = (index as u64).wrapping_mul(31).wrapping_add(seed) as u8;
            }
            unsafe { stream.upload(address as usize, &staging, 0, bytes) }.unwrap();
            stream.synchronize().unwrap();
        };
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for id in 1..=8u64 {
                write_unit(
                    geometry.unit_address(id),
                    geometry.unit_bytes as usize,
                    plane as u64 * 100 + id,
                );
            }
        }
        for slot in 1..=4u64 {
            write_unit(
                rank.row.unit_address(slot),
                rank.row.unit_bytes as usize,
                900 + slot,
            );
        }
        let mut pool = SlabPool::new(
            layout.page_record_bytes,
            16,
            layout.row_record_bytes,
            4,
            true,
        )
        .unwrap();
        pool.register(&CudaRegistrar, context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
        ));
        let memory = MemoryStore::new_flat(IDENTITY, pool, transfers.clone()).unwrap();
        let disk = DiskStore::new(
            record::DirectoryLock::acquire(&directory).unwrap(),
            IDENTITY,
            fingerprint,
            1 << 40,
            geometry.clone(),
            Vec::new(),
            Box::new(gpu_transport(context, device, &geometry)),
        )
        .unwrap();
        let mut store = CompositeStore::new(memory, disk).unwrap();
        let mut ids = Ids::new();
        let drain = |store: &mut CompositeStore,
                     events: &mut Vec<StoreEvent>,
                     until: &dyn Fn(&CompositeStore) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(60);
            while !until(store) {
                assert!(
                    Instant::now() < deadline,
                    "composite did not settle: outbound {} jobs {} stats {:?} events {events:?}",
                    store.outbound_records(),
                    store.disk_jobs(),
                    store.stats()
                );
                store.poll(events).unwrap();
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        // Two root and child pairs: the child of the first pair is published
        // before the root's memory copy lands, the child of the second after
        // the root's memory copy landed but while its disk write may still run.
        for (pair, wait_for_root_copy) in [(0u32, false), (1u32, true)] {
            let root_consumed = 3 * page_tokens;
            let child_consumed = 4 * page_tokens + 5;
            let mut history = PrefixHistory::new(IDENTITY);
            let tokens: Vec<u32> = (0..child_consumed as u32 + 8)
                .map(|t| t.wrapping_mul(2654435761).wrapping_add(pair))
                .collect();
            history.append(&tokens, &[]).unwrap();
            let base = 1 + pair * 4;
            let root_pages: Vec<PageId> = (base..base + 3).map(PageId).collect();
            let child_page = PageId(base + 3);
            let root = PublishedCheckpoint {
                id: ids.next(),
                key: history.key_at(root_consumed).unwrap(),
                parent: None,
                bucket: history.bucket_at(root_consumed).unwrap(),
                chain_pages: root_pages.clone(),
                tail_pages: 3,
                slot: StateSlotId(1 + pair * 2),
            };
            let child = PublishedCheckpoint {
                id: ids.next(),
                key: history.key_at(child_consumed).unwrap(),
                parent: Some(root.key),
                bucket: history.bucket_at(child_consumed).unwrap(),
                chain_pages: root_pages.iter().copied().chain([child_page]).collect(),
                tail_pages: 1,
                slot: StateSlotId(2 + pair * 2),
            };
            let mut events = Vec::new();
            assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
            if wait_for_root_copy {
                drain(&mut store, &mut events, &|store| {
                    store.stats().bytes_out > 0 && store.memory().record(&root.key).is_some()
                });
            }
            assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
            drain(&mut store, &mut events, &|store| {
                store.outbound_records() == 0
            });
            let stored: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    StoreEvent::Stored { key, result, .. } => Some((*key, result.is_ok())),
                    _ => None,
                })
                .collect();
            assert_eq!(stored.len(), 2, "pair {pair}: {events:?}");
            assert!(stored.iter().all(|(_, ok)| *ok), "pair {pair}: {events:?}");
            let stats = store.stats();
            assert_eq!(
                (stats.stored, stats.store_failed, stats.store_skipped),
                (2 * (u64::from(pair) + 1), 0, 0),
                "pair {pair}"
            );
            for key in [root.key, child.key] {
                let node = store.disk().index().node(&key).unwrap();
                assert!(
                    node.ready && !node.pinned(),
                    "pair {pair} disk node {key:?}"
                );
                assert_eq!(
                    fs::metadata(&node.location.path).unwrap().len(),
                    node.location.header.length_with_row()
                );
                assert!(
                    store.memory().record(&key).unwrap().durable,
                    "pair {pair} memory {key:?}"
                );
            }
            assert!(
                store.disk_jobs() == 0,
                "pair {pair}: {} disk jobs",
                store.disk_jobs()
            );
        }
        // A grandchild under a page-less child: the child's endpoint shares its
        // parent's last page, so its record holds only a row. After a restart the
        // chain restores through disk with the child contributing no bytes.
        let (pair, base) = (2u32, 9u32);
        let root_consumed = 3 * page_tokens;
        let child_consumed = 3 * page_tokens + 5;
        let grandchild_consumed = 4 * page_tokens + 7;
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..grandchild_consumed as u32 + 8)
            .map(|t| t.wrapping_mul(2654435761).wrapping_add(pair))
            .collect();
        history.append(&tokens, &[]).unwrap();
        let root_pages: Vec<PageId> = (base..base + 3).map(PageId).collect();
        let grandchild_page = PageId(base + 3);
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for id in base..=base + 3 {
                write_unit(
                    geometry.unit_address(u64::from(id)),
                    geometry.unit_bytes as usize,
                    plane as u64 * 100 + u64::from(id),
                );
            }
        }
        for slot in 5..=6u64 {
            write_unit(
                rank.row.unit_address(slot),
                rank.row.unit_bytes as usize,
                900 + slot,
            );
        }
        let root = PublishedCheckpoint {
            id: ids.next(),
            key: history.key_at(root_consumed).unwrap(),
            parent: None,
            bucket: history.bucket_at(root_consumed).unwrap(),
            chain_pages: root_pages.clone(),
            tail_pages: 3,
            slot: StateSlotId(5),
        };
        let child = PublishedCheckpoint {
            id: ids.next(),
            key: history.key_at(child_consumed).unwrap(),
            parent: Some(root.key),
            bucket: history.bucket_at(child_consumed).unwrap(),
            chain_pages: root_pages.clone(),
            tail_pages: 0,
            slot: StateSlotId(6),
        };
        let grandchild = PublishedCheckpoint {
            id: ids.next(),
            key: history.key_at(grandchild_consumed).unwrap(),
            parent: Some(child.key),
            bucket: history.bucket_at(grandchild_consumed).unwrap(),
            chain_pages: root_pages
                .iter()
                .copied()
                .chain([grandchild_page])
                .collect(),
            tail_pages: 1,
            slot: StateSlotId(5),
        };
        let mut events = Vec::new();
        for checkpoint in [&root, &child, &grandchild] {
            assert_eq!(store.store(checkpoint).unwrap(), StoreOutcome::Queued);
            drain(&mut store, &mut events, &|store| {
                store.outbound_records() == 0
            });
        }
        assert_eq!(store.stats().stored, 7);
        store.close().unwrap();
        drop(store);

        // Restart: a fresh composite over the scanned directory and an empty memory pool.
        let directory_lock = record::DirectoryLock::acquire(&directory).unwrap();
        let scanned =
            flare_selene::record::scan(&directory_lock, &IDENTITY, &fingerprint, geometry[0].page_tokens, &flare_selene::record::rank_sizes(&geometry))
                .unwrap();
        assert_eq!(scanned.len(), 7);
        let mut pool = SlabPool::new(
            layout.page_record_bytes,
            16,
            layout.row_record_bytes,
            4,
            true,
        )
        .unwrap();
        pool.register(&CudaRegistrar, context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
        ));
        let memory = MemoryStore::new_flat(IDENTITY, pool, transfers.clone()).unwrap();
        let disk = DiskStore::new(
            directory_lock,
            IDENTITY,
            fingerprint,
            1 << 40,
            geometry.clone(),
            scanned,
            Box::new(gpu_transport(context, device, &geometry)),
        )
        .unwrap();
        let mut store = CompositeStore::new(memory, disk).unwrap();
        let hit = store.lookup(&history, 0, u64::MAX).unwrap();
        assert_eq!(
            (hit.key, hit.pages, hit.tier),
            (grandchild.key, 4, Tier::Nvme)
        );
        let targets: Vec<PageId> = (base + 10..base + 14).map(PageId).collect();
        let ticket = InboundTicket {
            key: grandchild.key,
            pages: targets.clone(),
            slot: StateSlotId(7),
        };
        store.fetch(hit, ticket).unwrap();
        let mut events = Vec::new();
        drain(&mut store, &mut events, &|store| {
            store.inbound_records() == 0
        });
        assert!(
            matches!(
                events.as_slice(),
                [StoreEvent::Fetched { result: Ok(_), .. }]
            ),
            "{events:?}"
        );
        let read_unit = |address: u64, bytes: usize| -> Vec<u8> {
            unsafe { stream.download(&staging, address as usize, bytes) }.unwrap();
            stream.synchronize().unwrap();
            staging.bytes()[..bytes].to_vec()
        };
        let pattern = |seed: u64, bytes: usize| -> Vec<u8> {
            (0..bytes)
                .map(|index| (index as u64).wrapping_mul(31).wrapping_add(seed) as u8)
                .collect()
        };
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for (source, target) in (base..=base + 3).zip(&targets) {
                let bytes = geometry.unit_bytes as usize;
                assert!(
                    read_unit(geometry.unit_address(u64::from(target.0)), bytes)
                        == pattern(plane as u64 * 100 + u64::from(source), bytes),
                    "plane {} page {} differs from page {source}",
                    geometry.name,
                    target.0
                );
            }
        }
        let bytes = rank.row.unit_bytes as usize;
        assert!(read_unit(rank.row.unit_address(7), bytes) == pattern(905, bytes));
        assert_eq!(store.stats().fetched, 1);
        assert!(store.memory().record(&child.key).is_some());
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn composite_round_trips_through_memory_and_disk_on_a_loaded_package() {
        use crate::cuda::Stream;
        use flare_selene::geometry::fingerprint;
        let Some(package) = std::env::var_os("FLARE_OFFLOAD_TEST_PACKAGE") else {
            eprintln!("skipping: FLARE_OFFLOAD_TEST_PACKAGE is unset");
            return;
        };
        let Some(directory) = test_directory("composite", "gpu") else {
            return;
        };
        let model = super::load_test_model(&package);
        let manifest = model.manifests().unwrap().remove(0);
        let geometry = vec![manifest.cache.clone().unwrap()];
        let rank = geometry[0].clone();
        let (context, device) = (manifest.context, manifest.device);
        let layout = Layout::new(&geometry).unwrap();
        let fingerprint = fingerprint(&geometry);
        let page_tokens = rank.page_tokens;
        assert!(rank.page_capacity >= 16 && rank.row_capacity >= 4);
        let capacity = rank.page_capacity as u32;
        let sources = pages_of([1, 7, capacity / 2]);
        let targets = pages_of([2, 9, capacity / 2 + 1]);
        let (source_slot, target_slot) = (StateSlotId(1), StateSlotId(2));
        fn pages_of(ids: [u32; 3]) -> Vec<PageId> {
            ids.into_iter().map(PageId).collect()
        }

        let row_plane = rank.planes.len();
        let stream = Stream::open(context, device).unwrap();
        let largest = rank
            .planes
            .iter()
            .chain([&rank.row])
            .map(|plane| plane.unit_bytes as usize)
            .max()
            .unwrap();
        let mut staging = stream.host_buffer(largest).unwrap();
        let mut write_unit = |address: u64, bytes: usize, seed: u64| {
            fill(&mut staging.bytes_mut()[..bytes], seed);
            unsafe { stream.upload(address as usize, &staging, 0, bytes) }.unwrap();
            stream.synchronize().unwrap();
        };
        let write_pages = |write_unit: &mut dyn FnMut(u64, usize, u64),
                           ids: &[PageId],
                           pattern: &dyn Fn(usize, u32) -> u64| {
            for (plane, geometry) in rank.planes.iter().enumerate() {
                for id in ids {
                    write_unit(
                        geometry.unit_address(u64::from(id.0)),
                        geometry.unit_bytes as usize,
                        pattern(plane, id.0),
                    );
                }
            }
        };
        let write_row =
            |write_unit: &mut dyn FnMut(u64, usize, u64), slot: StateSlotId, seed: u64| {
                write_unit(
                    rank.row.unit_address(u64::from(slot.0)),
                    rank.row.unit_bytes as usize,
                    seed,
                );
            };
        write_pages(&mut write_unit, &sources, &seed);
        write_row(&mut write_unit, source_slot, seed(row_plane, source_slot.0));
        write_pages(&mut write_unit, &targets, &|_, _| 0xDD);
        write_row(&mut write_unit, target_slot, 0xDD);

        // Seven page slabs and two row slabs hold two three-page records with
        // one page to spare; a third record evicts the least recently used.
        let mut pool = SlabPool::new(
            layout.page_record_bytes,
            7,
            layout.row_record_bytes,
            2,
            true,
        )
        .unwrap();
        pool.register(&CudaRegistrar, context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
        ));
        let memory = MemoryStore::new_flat(IDENTITY, pool, transfers.clone()).unwrap();
        let disk = DiskStore::new(
            record::DirectoryLock::acquire(&directory).unwrap(),
            IDENTITY,
            fingerprint,
            1 << 40,
            geometry.clone(),
            Vec::new(),
            Box::new(gpu_transport(context, device, &geometry)),
        )
        .unwrap();
        let mut store = CompositeStore::new(memory, disk).unwrap();

        let consumed = 3 * page_tokens + 5;
        let base: Vec<u32> = (0..consumed as u32 + 40)
            .map(|t| t.wrapping_mul(2654435761))
            .collect();
        let histories: Vec<PrefixHistory> = (0..4u32)
            .map(|salt| {
                let mut history = PrefixHistory::new(IDENTITY);
                let tokens: Vec<u32> = base.iter().map(|t| t.wrapping_add(salt)).collect();
                history.append(&tokens, &[]).unwrap();
                history
            })
            .collect();
        let mut ids = Ids::new();
        let checkpoints: Vec<PublishedCheckpoint> = histories
            .iter()
            .enumerate()
            .map(|(index, history)| PublishedCheckpoint {
                id: ids.next(),
                key: history.key_at(consumed).unwrap(),
                parent: None,
                bucket: history.bucket_at(consumed).unwrap(),
                chain_pages: if index == 0 {
                    sources.clone()
                } else {
                    targets.clone()
                },
                tail_pages: 3,
                slot: source_slot,
            })
            .collect();
        let gpu_bytes = 3 * rank.page_bytes() + rank.row.unit_bytes;
        let record_bytes = layout.record_bytes(3);

        fn wait_until(store: &mut CompositeStore, mut done: impl FnMut(&CompositeStore) -> bool) {
            let deadline = Instant::now() + Duration::from_secs(120);
            let mut events = Vec::new();
            while !done(store) {
                assert!(
                    Instant::now() < deadline,
                    "condition not reached within 120 s"
                );
                store.poll(&mut events).unwrap();
                assert!(events.is_empty(), "{events:?}");
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let store_and_land = |store: &mut CompositeStore, checkpoint: &PublishedCheckpoint| {
            assert_eq!(store.store(checkpoint).unwrap(), StoreOutcome::Queued);
            match super::wait_event(store, 120) {
                StoreEvent::Stored {
                    checkpoint: id,
                    key,
                    result,
                } => {
                    assert_eq!((id, key), (checkpoint.id, checkpoint.key));
                    assert_eq!(result.unwrap(), gpu_bytes);
                }
                other => panic!("unexpected event {other:?}"),
            }
            let key = checkpoint.key;
            wait_until(store, |store| !store.copying_out(&key));
            let node = store.disk().index().node(&key).unwrap();
            assert!(node.ready && !node.pinned());
            assert_eq!(
                fs::metadata(&node.location.path).unwrap().len(),
                node.location.header.length_with_row()
            );
            assert!(store.memory().record(&key).unwrap().durable);
        };

        // The first checkpoint lands in memory, is written through, and is
        // served from memory.
        let first = &checkpoints[0];
        store_and_land(&mut store, first);
        let stats = store.stats();
        assert_eq!(
            (
                stats.stored,
                stats.bytes_out,
                stats.memory_records,
                stats.disk_records
            ),
            (1, gpu_bytes, 1, 1)
        );
        assert_eq!(stats.memory_bytes, record_bytes);
        write_pages(&mut write_unit, &sources, &|plane, _| 0xEE + plane as u64);
        write_row(&mut write_unit, source_slot, 0xEE);
        let hit = store.lookup(&histories[0], 0, u64::MAX).unwrap();
        assert_eq!(
            (hit.key, hit.pages, hit.tier, hit.bytes),
            (first.key, 3, Tier::Dram, record_bytes)
        );
        let ticket = || InboundTicket {
            key: first.key,
            pages: targets.clone(),
            slot: target_slot,
        };
        store.fetch(hit, ticket()).unwrap();
        assert_eq!(store.stats().inbound_in_flight, 1);
        match super::wait_event(&mut store, 120) {
            StoreEvent::Fetched { key, result } => {
                assert_eq!(key, first.key);
                assert_eq!(result.unwrap(), gpu_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        let readback = stream.host_buffer(largest).unwrap();
        let read_unit = |address: u64, bytes: usize| -> Vec<u8> {
            unsafe { stream.download(&readback, address as usize, bytes) }.unwrap();
            stream.synchronize().unwrap();
            readback.bytes()[..bytes].to_vec()
        };
        let check = |read_unit: &dyn Fn(u64, usize) -> Vec<u8>| {
            for (plane, geometry) in rank.planes.iter().enumerate() {
                for (source, target) in sources.iter().zip(&targets) {
                    let mut wanted = vec![0u8; geometry.unit_bytes as usize];
                    fill(&mut wanted, seed(plane, source.0));
                    let actual = read_unit(
                        geometry.unit_address(u64::from(target.0)),
                        geometry.unit_bytes as usize,
                    );
                    assert!(
                        actual == wanted,
                        "plane {} ({plane}) page {} differs from page {}",
                        geometry.name,
                        target.0,
                        source.0
                    );
                }
            }
            let mut wanted = vec![0u8; rank.row.unit_bytes as usize];
            fill(&mut wanted, seed(row_plane, source_slot.0));
            let actual = read_unit(
                rank.row.unit_address(u64::from(target_slot.0)),
                rank.row.unit_bytes as usize,
            );
            assert!(
                actual == wanted,
                "row {} differs from row {}",
                target_slot.0,
                source_slot.0
            );
        };
        check(&read_unit);
        let stats = store.stats();
        assert_eq!(
            (
                stats.fetched,
                stats.hits,
                stats.reused_tokens,
                stats.bytes_in
            ),
            (1, 1, consumed, gpu_bytes)
        );

        // Two more records fill the memory pool; the third evicts the first,
        // which stays on disk.
        store_and_land(&mut store, &checkpoints[1]);
        assert!(store.memory().index().contains(&first.key));
        store_and_land(&mut store, &checkpoints[2]);
        assert!(!store.memory().index().contains(&first.key));
        assert!(store.disk().index().node(&first.key).unwrap().ready);
        let stats = store.stats();
        assert_eq!(
            (stats.stored, stats.memory_records, stats.disk_records),
            (3, 2, 3)
        );

        // The first record now comes from its file through memory slabs, and
        // is resident again afterwards.
        write_pages(&mut write_unit, &targets, &|_, _| 0xCC);
        write_row(&mut write_unit, target_slot, 0xCC);
        let hit = store.lookup(&histories[0], 0, u64::MAX).unwrap();
        assert_eq!((hit.key, hit.tier), (first.key, Tier::Nvme));
        assert!(hit.estimated_seconds > record_bytes as f64 / INITIAL_MEMORY_RATE);
        store.fetch(hit, ticket()).unwrap();
        match super::wait_event(&mut store, 120) {
            StoreEvent::Fetched { key, result } => {
                assert_eq!(key, first.key);
                assert_eq!(result.unwrap(), gpu_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        check(&read_unit);
        let node = store.memory().index().node(&first.key).unwrap();
        assert!(node.ready && !node.pinned() && node.location.durable);
        assert_eq!(node.location.pages.len(), 3);
        assert!(!store.memory().index().contains(&checkpoints[1].key));
        assert!(store.memory().index().nodes().all(|node| !node.pinned()));
        assert!(store.disk().index().nodes().all(|node| !node.pinned()));
        let stats = store.stats();
        assert_eq!(
            (stats.fetched, stats.fetch_failed, stats.bytes_in),
            (2, 0, 2 * gpu_bytes)
        );
        assert_eq!((stats.memory_records, stats.disk_records), (2, 3));

        // A record whose file cannot be created is dropped from memory and counted.
        let fourth = &checkpoints[3];
        let blocked = directory.join(record::temporary_name(&fourth.key));
        fs::create_dir(&blocked).unwrap();
        assert_eq!(store.store(fourth).unwrap(), StoreOutcome::Queued);
        match super::wait_event(&mut store, 120) {
            StoreEvent::Stored { key, result, .. } => {
                assert_eq!(key, fourth.key);
                assert_eq!(result.unwrap(), gpu_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        wait_until(&mut store, |store| store.outbound_records() == 0);
        fs::remove_dir(&blocked).unwrap();
        assert!(!store.memory().index().contains(&fourth.key));
        assert!(!store.disk().index().contains(&fourth.key));
        assert!(store.memory().index().contains(&first.key));
        let stats = store.stats();
        assert_eq!(
            (stats.stored, stats.store_failed, stats.store_skipped),
            (3, 1, 0)
        );
        assert_eq!((stats.memory_records, stats.disk_records), (1, 3));
        assert_eq!(
            record::scan(
                &store.disk().directory_lock,
                &IDENTITY,
                &fingerprint,
                geometry[0].page_tokens,
                &record::rank_sizes(&geometry)
            )
            .unwrap()
            .len(),
            3
        );
        store.close().unwrap();
        drop(store);
        // Retention is smaller than a record and staging holds one record's
        // pages. Exercise simultaneous memory copy-out, streamed copy-out,
        // and streamed restore, then reopen the files with an empty RAM tier.
        let open_small = || {
            let mut pool = SlabPool::new(
                layout.page_record_bytes,
                1,
                layout.row_record_bytes,
                1,
                true,
            )
            .unwrap();
            pool.register(&CudaRegistrar, context).unwrap();
            let transfers = Arc::new(Mutex::new(
                Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
            ));
            let memory = MemoryStore::new_flat(IDENTITY, pool, transfers).unwrap();
            let mut staging_pool = SlabPool::new(
                layout.page_record_bytes,
                3,
                layout.row_record_bytes,
                2,
                true,
            )
            .unwrap();
            staging_pool.register(&CudaRegistrar, context).unwrap();
            let transfers = Arc::new(Mutex::new(
                Transfers::start(&[(context, device)], geometry.clone(), &staging_pool).unwrap(),
            ));
            let lock = record::DirectoryLock::acquire(&directory).unwrap();
            let scanned = record::scan(&lock, &IDENTITY, &fingerprint, geometry[0].page_tokens, &record::rank_sizes(&geometry)).unwrap();
            let disk = DiskStore::new(
                lock,
                IDENTITY,
                fingerprint,
                1 << 40,
                geometry.clone(),
                scanned,
                Box::new(Staged::new(
                    Some(staging_pool),
                    transfers,
                    DiskThread::start("flare-offload-disk").unwrap(),
                )),
            )
            .unwrap();
            CompositeStore::new(memory, disk).unwrap()
        };
        let mut store = open_small();
        write_pages(&mut write_unit, &sources, &seed);
        write_row(&mut write_unit, source_slot, seed(row_plane, source_slot.0));
        let mut fresh = PrefixHistory::new(IDENTITY);
        fresh
            .append(
                &base
                    .iter()
                    .map(|token| token.wrapping_add(77))
                    .collect::<Vec<_>>(),
                &[],
            )
            .unwrap();
        let short = checkpoint(
            &mut ids,
            &fresh,
            page_tokens + 5,
            None,
            sources[..1].to_vec(),
            1,
            source_slot,
        );
        let large = checkpoint(
            &mut ids,
            &fresh,
            consumed,
            None,
            sources.clone(),
            3,
            source_slot,
        );
        assert_eq!(store.store(&short).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.store(&large).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.outbound_records(), 1);
        assert_eq!(store.disk().stats().outbound_in_flight, 1);
        assert_eq!(store.memory().pool().free_pages(), 0);
        let hit = store.lookup(&histories[0], 0, u64::MAX).unwrap();
        assert!(!store.retains_chain(&hit.chain));
        store.fetch(hit, ticket()).unwrap();
        let mut events = Vec::new();
        poll_until(&mut store, &mut events, |store| {
            let stats = store.stats();
            stats.outbound_in_flight == 0 && stats.inbound_in_flight == 0
        });
        assert_eq!(events.len(), 3, "{events:?}");
        let mut stored_keys = Vec::new();
        for event in events {
            match event {
                StoreEvent::Stored { key, result, .. } => {
                    result.unwrap();
                    stored_keys.push(key);
                }
                StoreEvent::Fetched { key, result } => {
                    assert_eq!(key, first.key);
                    assert_eq!(result.unwrap(), record_bytes);
                }
                _ => panic!("unexpected event"),
            }
        }
        assert!(stored_keys.contains(&short.key) && stored_keys.contains(&large.key));
        check(&read_unit);
        assert!(store.memory().record(&short.key).unwrap().durable);
        assert!(store.memory().record(&large.key).is_none());
        assert!(store.memory().index().nodes().all(|node| !node.pinned()));
        assert!(store.disk().index().nodes().all(|node| !node.pinned()));
        assert_eq!(
            (
                store.stats().stored,
                store.stats().fetched,
                store.stats().store_failed,
                store.stats().fetch_failed
            ),
            (2, 1, 0, 0)
        );
        assert_eq!(
            store.stats().inbound_rate,
            store.disk().stats().inbound_rate
        );
        store.close().unwrap();
        drop(store);

        let mut store = open_small();
        write_pages(&mut write_unit, &targets, &|_, _| 0xA5);
        write_row(&mut write_unit, target_slot, 0xA5);
        let hit = store.lookup(&fresh, 0, u64::MAX).unwrap();
        assert_eq!((hit.key, hit.pages, hit.tier), (large.key, 3, Tier::Nvme));
        store
            .fetch(
                hit,
                InboundTicket {
                    key: large.key,
                    pages: targets.clone(),
                    slot: target_slot,
                },
            )
            .unwrap();
        match super::wait_event(&mut store, 120) {
            StoreEvent::Fetched { key, result } => {
                assert_eq!(key, large.key);
                assert_eq!(result.unwrap(), record_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        check(&read_unit);
        assert_eq!(store.stats().memory_records, 0);
        assert!(store.disk().index().nodes().all(|node| !node.pinned()));
        store.close().unwrap();
        drop(store);
        drop(readback);
        drop(staging);
        drop(stream);
        drop(model);
        fs::remove_dir_all(&directory).unwrap();
    }
}

mod disk_store {
    use super::{Ids, checkpoint, fill, seed, test_directory};
    use std::{
        fs,
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use flare_selene::{
        HostStore, InboundTicket, Layout, PageId, PublishedCheckpoint, StateSlotId, StoreEvent,
        StoreOutcome, Tier,
        disk::DiskThread,
        disk_store::{DiskStore, Staged},
        record::{self, Scanned},
        slab::{STAGING_PAGE_SLABS, SlabPool},
    };

    use crate::{
        cache::PrefixHistory,
        offload::{registrar::CudaRegistrar, transfer::Transfers},
    };

    const IDENTITY: [u8; 32] = [0x5a; 32];

    fn history(tokens: u64) -> PrefixHistory {
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..tokens as u32)
            .map(|t| t.wrapping_mul(2654435761))
            .collect();
        history.append(&tokens, &[]).unwrap();
        history
    }

    fn pages(range: std::ops::Range<u32>) -> Vec<PageId> {
        range.map(PageId).collect()
    }

    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn staged_store_round_trips_records_through_files_and_a_restart() {
        use crate::cuda::Stream;
        use flare_selene::geometry::fingerprint;
        let Some(package) = std::env::var_os("FLARE_OFFLOAD_TEST_PACKAGE") else {
            eprintln!("skipping: FLARE_OFFLOAD_TEST_PACKAGE is unset");
            return;
        };
        let Some(directory) = test_directory("disk-store", "round-trip") else {
            return;
        };
        let model = super::load_test_model(&package);
        let manifest = model.manifests().unwrap().remove(0);
        let geometry = vec![manifest.cache.clone().unwrap()];
        let rank = geometry[0].clone();
        let (context, device) = (manifest.context, manifest.device);
        let layout = Layout::new(&geometry).unwrap();
        let fingerprint = fingerprint(&geometry);
        let page_tokens = rank.page_tokens;
        assert!(rank.page_capacity >= 400 && rank.row_capacity >= 4);

        const SOURCE: u32 = 100;
        const RESTORED: u32 = 200;
        const AGAIN: u32 = 300;
        const PAGES: u32 = 70;
        const EXTRA: u32 = 2;
        let source_pages = pages(SOURCE..SOURCE + PAGES);
        let extra_pages = pages(SOURCE + PAGES..SOURCE + PAGES + EXTRA);
        let (source_slot, child_slot, restored_slot, again_slot) = (
            StateSlotId(1),
            StateSlotId(0),
            StateSlotId(2),
            StateSlotId(3),
        );

        let row_plane = rank.planes.len();
        let stream = Stream::open(context, device).unwrap();
        let largest = rank
            .planes
            .iter()
            .chain([&rank.row])
            .map(|plane| plane.unit_bytes as usize)
            .max()
            .unwrap();
        let mut staging = stream.host_buffer(largest).unwrap();
        let mut write_unit = |address: u64, bytes: usize, seed: u64| {
            fill(&mut staging.bytes_mut()[..bytes], seed);
            unsafe { stream.upload(address as usize, &staging, 0, bytes) }.unwrap();
            stream.synchronize().unwrap();
        };
        let write_pages = |write_unit: &mut dyn FnMut(u64, usize, u64),
                           ids: &[PageId],
                           pattern: &dyn Fn(usize, u32) -> u64| {
            for (plane, geometry) in rank.planes.iter().enumerate() {
                for id in ids {
                    write_unit(
                        geometry.unit_address(u64::from(id.0)),
                        geometry.unit_bytes as usize,
                        pattern(plane, id.0),
                    );
                }
            }
        };
        write_pages(&mut write_unit, &source_pages, &seed);
        write_pages(&mut write_unit, &extra_pages, &seed);
        write_pages(
            &mut write_unit,
            &pages(RESTORED..RESTORED + PAGES),
            &|_, _| 0xDD,
        );
        write_pages(&mut write_unit, &pages(AGAIN..AGAIN + PAGES), &|_, _| 0xDD);
        for slot in [source_slot, child_slot] {
            write_unit(
                rank.row.unit_address(u64::from(slot.0)),
                rank.row.unit_bytes as usize,
                seed(row_plane, slot.0),
            );
        }
        for slot in [restored_slot, again_slot] {
            write_unit(
                rank.row.unit_address(u64::from(slot.0)),
                rank.row.unit_bytes as usize,
                0xDD,
            );
        }

        let consumed = u64::from(PAGES) * page_tokens + 3;
        let child_consumed = u64::from(PAGES + EXTRA) * page_tokens + 5;
        let full = history(child_consumed + 40);
        let mut ids = Ids::new();
        let root = checkpoint(
            &mut ids,
            &full,
            consumed,
            None,
            source_pages.clone(),
            PAGES as usize,
            source_slot,
        );

        let open = |scanned: Vec<Scanned>, limit: u64| -> DiskStore {
            let mut pool = SlabPool::staging(&layout).unwrap();
            pool.register(&CudaRegistrar, context).unwrap();
            let transfers = Arc::new(Mutex::new(
                Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
            ));
            let disk = DiskThread::start("flare-offload-disk").unwrap();
            DiskStore::new(
                record::DirectoryLock::acquire(&directory).unwrap(),
                IDENTITY,
                fingerprint,
                limit,
                geometry.clone(),
                scanned,
                Box::new(Staged::new(Some(pool), transfers, disk)),
            )
            .unwrap()
        };

        fn staging_free(store: &mut DiskStore) -> (usize, usize) {
            let pool = store.transport_mut().staged_mut().unwrap().pool().unwrap();
            (pool.free_pages(), pool.free_rows())
        }
        let scan = |lock: &record::DirectoryLock| {
            record::scan(lock, &IDENTITY, &fingerprint, geometry[0].page_tokens, &record::rank_sizes(&geometry)).unwrap()
        };

        assert!(scan(&record::DirectoryLock::acquire(&directory).unwrap()).is_empty());
        let mut store = open(Vec::new(), 1 << 40);
        assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().outbound_in_flight, 1);
        let StoreEvent::Stored {
            checkpoint: stored,
            key,
            result,
        } = super::wait_event(&mut store, 120)
        else {
            panic!("expected a Stored event");
        };
        assert_eq!((stored, key), (root.id, root.key));
        let bytes = result.unwrap();
        let header = store.record(&root.key).unwrap().header.clone();
        let path = directory.join(record::file_name(&root.key));
        assert_eq!(bytes, header.length_with_row());
        assert_eq!(fs::metadata(&path).unwrap().len(), header.length_with_row());
        assert!(!directory.join(record::temporary_name(&root.key)).exists());
        assert_eq!(header.tail_pages, PAGES);
        let scanned = scan(&store.directory_lock);
        assert_eq!(scanned.len(), 1);
        assert_eq!(scanned[0].header.key, root.key);
        assert!(scanned[0].row);
        let node = store.index().node(&root.key).unwrap();
        assert!(node.ready && !node.pinned());
        let stats = store.stats();
        assert_eq!(
            (stats.stored, stats.bytes_out, stats.disk_bytes),
            (1, bytes, bytes)
        );
        assert_eq!(staging_free(&mut store), (STAGING_PAGE_SLABS, 4));

        // Poison the source pages and row, then restore into fresh ids and another slot.
        write_pages(&mut write_unit, &source_pages, &|plane, _| {
            0xEE + plane as u64
        });
        write_unit(
            rank.row.unit_address(u64::from(source_slot.0)),
            rank.row.unit_bytes as usize,
            0xEE,
        );
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(
            (hit.key, hit.pages, hit.tier),
            (root.key, PAGES as usize, Tier::Nvme)
        );
        store
            .fetch(
                hit,
                InboundTicket {
                    key: root.key,
                    pages: pages(RESTORED..RESTORED + PAGES),
                    slot: restored_slot,
                },
            )
            .unwrap();
        let StoreEvent::Fetched { key, result } = super::wait_event(&mut store, 120) else {
            panic!("expected a Fetched event");
        };
        assert_eq!(key, root.key);
        assert_eq!(result.unwrap(), layout.record_bytes(u64::from(PAGES)));
        assert_eq!(staging_free(&mut store), (STAGING_PAGE_SLABS, 4));
        let stats = store.stats();
        assert_eq!((stats.fetched, stats.reused_tokens), (1, consumed));
        let read_unit = |address: u64, bytes: usize| -> Vec<u8> {
            unsafe { stream.download(&staging, address as usize, bytes) }.unwrap();
            stream.synchronize().unwrap();
            staging.bytes()[..bytes].to_vec()
        };
        let check = |read_unit: &dyn Fn(u64, usize) -> Vec<u8>,
                     sources: &[PageId],
                     targets: &[PageId],
                     row_source: StateSlotId,
                     slot: StateSlotId| {
            assert_eq!(sources.len(), targets.len());
            for (plane, geometry) in rank.planes.iter().enumerate() {
                for (source, target) in sources.iter().zip(targets) {
                    let mut wanted = vec![0u8; geometry.unit_bytes as usize];
                    fill(&mut wanted, seed(plane, source.0));
                    let actual = read_unit(
                        geometry.unit_address(u64::from(target.0)),
                        geometry.unit_bytes as usize,
                    );
                    assert!(
                        actual == wanted,
                        "plane {} ({plane}) page {} differs from page {}",
                        geometry.name,
                        target.0,
                        source.0
                    );
                }
            }
            let mut wanted = vec![0u8; rank.row.unit_bytes as usize];
            fill(&mut wanted, seed(row_plane, row_source.0));
            let actual = read_unit(
                rank.row.unit_address(u64::from(slot.0)),
                rank.row.unit_bytes as usize,
            );
            assert!(
                actual == wanted,
                "row {} differs from row {}",
                slot.0,
                row_source.0
            );
        };
        check(
            &read_unit,
            &source_pages,
            &pages(RESTORED..RESTORED + PAGES),
            source_slot,
            restored_slot,
        );
        store.close().unwrap();
        drop(store);

        // A store opened over a fresh scan serves the same record.
        let scanned = scan(&record::DirectoryLock::acquire(&directory).unwrap());
        assert_eq!(scanned.len(), 1);
        let mut store = open(scanned, 1 << 40);
        assert_eq!(store.used(), header.length_with_row());
        assert_eq!(store.stats().disk_records, 1);
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        store
            .fetch(
                hit,
                InboundTicket {
                    key: root.key,
                    pages: pages(AGAIN..AGAIN + PAGES),
                    slot: again_slot,
                },
            )
            .unwrap();
        let StoreEvent::Fetched { key, result } = super::wait_event(&mut store, 120) else {
            panic!("expected a Fetched event");
        };
        assert_eq!(key, root.key);
        result.unwrap();
        check(
            &read_unit,
            &source_pages,
            &pages(AGAIN..AGAIN + PAGES),
            source_slot,
            again_slot,
        );

        // A child record of two more pages makes the root a single-child ancestor.
        let mut chain = source_pages.clone();
        chain.extend_from_slice(&extra_pages);
        let child = checkpoint(
            &mut ids,
            &full,
            child_consumed,
            Some(consumed),
            chain,
            EXTRA as usize,
            child_slot,
        );
        assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
        let StoreEvent::Stored { key, result, .. } = super::wait_event(&mut store, 120) else {
            panic!("expected a Stored event");
        };
        assert_eq!(key, child.key);
        let child_bytes = result.unwrap();
        let child_header = store.record(&child.key).unwrap().header.clone();
        assert_eq!(
            (child_header.parent, child_header.tail_pages),
            (Some(root.key), EXTRA)
        );
        assert_eq!(child_bytes, child_header.length_with_row());
        assert_eq!(scan(&store.directory_lock).len(), 2);
        assert_eq!(staging_free(&mut store), (STAGING_PAGE_SLABS, 4));

        // Above a resident checkpoint five pages into the root, the chain
        // restores through both files: the root's remaining pages, then the
        // child's pages and row.
        const SKIP: u32 = 5;
        let chain_pages: Vec<PageId> = source_pages.iter().chain(&extra_pages).copied().collect();
        let targets = pages(RESTORED..RESTORED + PAGES + EXTRA - SKIP);
        let mut poison = stream.host_buffer(largest).unwrap();
        let mut poison_unit = |address: u64, bytes: usize, seed: u64| {
            fill(&mut poison.bytes_mut()[..bytes], seed);
            unsafe { stream.upload(address as usize, &poison, 0, bytes) }.unwrap();
            stream.synchronize().unwrap();
        };
        write_pages(&mut poison_unit, &targets, &|_, _| 0xCC);
        poison_unit(
            rank.row.unit_address(u64::from(restored_slot.0)),
            rank.row.unit_bytes as usize,
            0xCC,
        );
        let hit = store
            .lookup(&full, u64::from(SKIP) * page_tokens, u64::MAX)
            .unwrap();
        assert_eq!(hit.chain.nodes, [root.key, child.key]);
        assert_eq!(
            (hit.chain.skip_pages, hit.pages, hit.resident_tokens),
            (SKIP as usize, targets.len(), u64::from(SKIP) * page_tokens)
        );
        store
            .fetch(
                hit,
                InboundTicket {
                    key: child.key,
                    pages: targets.clone(),
                    slot: restored_slot,
                },
            )
            .unwrap();
        let StoreEvent::Fetched { key, result } = super::wait_event(&mut store, 120) else {
            panic!("expected a Fetched event");
        };
        assert_eq!(key, child.key);
        assert_eq!(
            result.unwrap(),
            layout.record_bytes(u64::from(PAGES + EXTRA - SKIP))
        );
        check(
            &read_unit,
            &chain_pages[SKIP as usize..],
            &targets,
            child_slot,
            restored_slot,
        );
        assert_eq!(staging_free(&mut store), (STAGING_PAGE_SLABS, 4));
        assert_eq!(store.stats().fetched, 2);
        store.close().unwrap();
        drop(store);

        // Under a limit the two records fill, with the child pinned, a third
        // record of only a row cannot be placed, and making room compacts the
        // root: its file is truncated at the row segment.
        let scanned = scan(&record::DirectoryLock::acquire(&directory).unwrap());
        assert_eq!(scanned.len(), 2);
        let mut store = open(scanned, bytes + child_bytes);
        assert_eq!(store.used(), store.limit());
        store.index_mut().pin(&child.key).unwrap();
        let mut chain = source_pages.clone();
        chain.extend_from_slice(&extra_pages);
        let grandchild = checkpoint(
            &mut ids,
            &full,
            child_consumed + 4,
            Some(child_consumed),
            chain,
            0,
            child_slot,
        );
        assert_ne!(store.store(&grandchild).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 1);
        assert!(store.record(&grandchild.key).is_none());
        let node = store.index().node(&root.key).unwrap();
        assert!(!node.row && node.ready);
        assert_eq!(node.location.bytes, header.row_segment);
        assert_eq!(store.used(), header.row_segment + child_bytes);
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let mut events = Vec::new();
            store.poll(&mut events).unwrap();
            assert!(events.is_empty());
            if fs::metadata(&path).unwrap().len() == header.row_segment {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the root record was not truncated within 60 s"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(!header.row_present(fs::metadata(&path).unwrap().len()));
        let scanned = scan(&store.directory_lock);
        assert_eq!(scanned.len(), 2);
        assert!(!scanned[0].row && scanned[0].header.key == root.key);
        assert_eq!(scanned[0].bytes, header.row_segment);
        assert!(scanned[1].row && scanned[1].header.key == child.key);
        assert!(store.lookup(&history(consumed), 0, u64::MAX).is_none());
        assert_eq!(store.lookup(&full, 0, u64::MAX).unwrap().key, child.key);
        store.index_mut().unpin(&child.key).unwrap();
        store.close().unwrap();
        drop(store);
        drop(staging);
        drop(stream);
        drop(model);
        fs::remove_dir_all(&directory).unwrap();
    }
}

mod geometry {
    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn package_geometry_lists_every_plane() {
        let Some(package) = std::env::var_os("FLARE_OFFLOAD_TEST_PACKAGE") else {
            return;
        };
        let model = super::load_test_model(&package);
        let manifest = model.manifests().unwrap().remove(0);
        let geometry = manifest.cache.unwrap();
        eprintln!(
            "page_tokens {} pages {} rows {} row {} {} bytes",
            geometry.page_tokens,
            geometry.page_capacity,
            geometry.row_capacity,
            geometry.row.name,
            geometry.row.unit_bytes
        );
        for plane in &geometry.planes {
            eprintln!(
                "plane {} {} bytes at {:#x}",
                plane.name, plane.unit_bytes, plane.address
            );
        }
        eprintln!(
            "page bytes {} record {}",
            geometry.page_bytes(),
            geometry.page_record_bytes()
        );
    }
}
