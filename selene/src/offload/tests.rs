use super::*;

fn prepare(
    options: &PrefixCacheOffload,
    package: &std::path::Path,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
) -> Result<Preparation> {
    super::prepare(
        options,
        package,
        identity,
        geometry,
        contexts,
        &crate::testing::IdentityRegistrar,
    )
}
fn prepare_with_inputs(
    options: &PrefixCacheOffload,
    package: &std::path::Path,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
    inputs: u64,
    shard: Option<(usize, usize)>,
) -> Result<Preparation> {
    super::prepare_with_inputs(
        options,
        package,
        identity,
        geometry,
        contexts,
        inputs,
        shard,
        &crate::testing::IdentityRegistrar,
    )
}

fn test_directory() -> PathBuf {
    static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "flare-offload-disabled-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

fn pool(address: &str) -> PoolOptions {
    PoolOptions {
        address: address.into(),
        ca: None,
    }
}

#[test]
fn a_pool_with_shards_keeps_one_local_rank_and_refuses_memory_retention() {
    let mut options = PrefixCacheOffload {
        pool: Some(pool("unix:/run/selene.sock")),
        ..Default::default()
    };
    let local = crate::testing::geometry(64, &[("kv", 4096)], 4096);
    let prepare = |options: &PrefixCacheOffload| {
        prepare_with_inputs(
            options,
            std::path::Path::new("model.safetensors"),
            [0; 32],
            vec![local.clone()],
            vec![(0, 0)],
            0,
            Some((1, 2)),
        )
    };
    let Preparation::Store(prepared) = prepare(&options).unwrap() else {
        panic!("no pool store")
    };
    assert_eq!(prepared.geometry.len(), 1);
    assert_eq!(
        prepared.fingerprint,
        geometry::shard_fingerprint(&local, 1, 2)
    );
    options.memory_bytes = Some(1);
    assert!(
        prepare(&options)
            .err()
            .unwrap()
            .to_string()
            .contains("disk only")
    );
}

#[test]
fn without_a_pool_nothing_for_it_is_budgeted_or_built() {
    let layout = Layout::new(&[crate::testing::geometry(64, &[("kv", 4096)], 4096)]).unwrap();
    assert_eq!(
        PrefixCacheOffload::default()
            .headroom_bytes(1, &layout)
            .unwrap(),
        (0, 0)
    );
    let tiers = PrefixCacheOffload {
        memory_bytes: Some(1 << 30),
        ..Default::default()
    };
    let pooled = PrefixCacheOffload {
        pool: Some(pool("host:7000")),
        ..tiers.clone()
    };
    let (device, host) = tiers.headroom_bytes(1, &layout).unwrap();
    // Static headroom covers the copier; Hello's regions are budgeted at startup.
    assert_eq!(
        pooled.headroom_bytes(1, &layout).unwrap(),
        (device + STORE_DEVICE_MARGIN_BYTES, host)
    );
    // A pool shares what the CPU reads, so its memory is mapped whole by the device
    // instead of split across ranks.
    let split = tiers.headroom_bytes(2, &layout).unwrap().0;
    let whole = pooled.headroom_bytes(2, &layout).unwrap().0;
    assert!(whole > split + STORE_DEVICE_MARGIN_BYTES, "{whole} {split}");
    let alone = PrefixCacheOffload {
        pool: Some(pool("host:7000")),
        ..Default::default()
    };
    assert!(alone.enabled());
    assert_eq!(
        alone.headroom_bytes(1, &layout).unwrap(),
        (STORE_DEVICE_MARGIN_BYTES, 0)
    );
}

#[test]
fn disabled_preparation_requires_no_geometry_context_or_worker_binding() {
    for options in [
        PrefixCacheOffload::default(),
        PrefixCacheOffload {
            memory_bytes: Some(0),
            disk_limit: Some(0),
            ..Default::default()
        },
    ] {
        let prepared = prepare(
            &options,
            std::path::Path::new("/missing/model.safetensors"),
            [0; 32],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert!(matches!(&prepared, Preparation::None(reason) if reason == "disabled"));
        assert!(prepared.into_shared().is_none());
    }
}

#[test]
fn zero_disk_limit_does_not_create_a_record_directory() {
    let directory = test_directory();
    let prepared = prepare(
        &PrefixCacheOffload {
            disk: Some(directory.clone()),
            disk_limit: Some(0),
            ..Default::default()
        },
        std::path::Path::new("/missing/model.safetensors"),
        [0; 32],
        vec![crate::testing::geometry(64, &[("kv", 4096)], 4096)],
        Vec::new(),
    )
    .unwrap();
    let created = directory.exists();
    if created {
        std::fs::remove_dir_all(&directory).unwrap();
    }
    assert!(
        !created,
        "disabled disk store created {}",
        directory.display()
    );
    assert!(prepared.into_shared().is_none());
}

#[test]
fn zero_disk_limit_avoids_disk_setup_when_memory_is_enabled() {
    let directory = test_directory();
    let geometry = vec![crate::testing::geometry(64, &[("kv", 4096)], 4096)];
    let memory_bytes = slab::SlabPool::minimum_bytes(&Layout::new(&geometry).unwrap());
    let error = prepare(
        &PrefixCacheOffload {
            memory_bytes: Some(memory_bytes),
            disk: Some(directory.clone()),
            disk_limit: Some(0),
            ..Default::default()
        },
        std::path::Path::new("/missing/model.safetensors"),
        [0; 32],
        geometry,
        Vec::new(),
    )
    .err()
    .expect("a memory store requires a CUDA context");
    let created = directory.exists();
    if created {
        std::fs::remove_dir_all(&directory).unwrap();
    }
    assert_eq!(
        error.to_string(),
        "the host prefix store requires at least one rank"
    );
    assert!(
        !created,
        "disabled disk store created {}",
        directory.display()
    );
}

#[test]
fn zero_disk_limit_does_not_lock_or_scan_existing_records() {
    let base = test_directory();
    let geometry = vec![crate::testing::geometry(64, &[("kv", 4096)], 4096)];
    let fingerprint = geometry::fingerprint(&geometry);
    let directory = base.join(record::directory_name(&[0; 32], &fingerprint));
    std::fs::create_dir_all(&directory).unwrap();
    let lock = record::DirectoryLock::acquire(&directory).unwrap();
    let active_write = directory.join("active-record.tmp");
    std::fs::write(&active_write, b"active writer payload").unwrap();
    let prepared = prepare(
        &PrefixCacheOffload {
            disk: Some(base.clone()),
            disk_limit: Some(0),
            ..Default::default()
        },
        std::path::Path::new("/missing/model.safetensors"),
        [0; 32],
        geometry,
        Vec::new(),
    )
    .unwrap();
    assert!(prepared.into_shared().is_none());
    assert_eq!(
        std::fs::read(active_write).unwrap(),
        b"active writer payload"
    );
    drop(lock);
    std::fs::remove_dir_all(base).unwrap();
}

#[test]
fn sizes_parse_binary_suffixes_like_the_ignition_configuration() {
    assert_eq!(parse_size("16G").unwrap(), 16 << 30);
    assert_eq!(parse_size("1.5 GiB").unwrap(), 3 << 29);
    assert_eq!(parse_size("512m").unwrap(), 512 << 20);
    assert_eq!(parse_size("2TB").unwrap(), 2 << 40);
    assert_eq!(parse_size("4096").unwrap(), 4096);
    assert_eq!(parse_size("0").unwrap(), 0);
    for text in ["", "G", "16X", "1.2.3G", "-1G", "16 GB extra"] {
        assert!(parse_size(text).is_err(), "{text:?}");
    }
}

#[test]
fn memory_headroom_accounts_for_the_actual_host_staging_geometry() {
    let layout = Layout {
        page_tokens: 64,
        ranks: Vec::new(),
        page_record_bytes: 32 << 20,
        row_record_bytes: 8 << 20,
    };
    let mut options = PrefixCacheOffload::default();
    assert_eq!(options.headroom_bytes(2, &layout).unwrap(), (0, 0));
    options.memory_bytes = Some(16 << 30);
    assert_eq!(
        options.headroom_bytes(2, &layout).unwrap(),
        ((8u64 << 30) / 512 + STORE_DEVICE_MARGIN_BYTES, 16 << 30)
    );
    options.disk = Some("records".into());
    let staging = 256 * (32u64 << 20) + 4 * (8u64 << 20);
    let host = (16u64 << 30) + staging;
    assert_eq!(
        options.headroom_bytes(2, &layout).unwrap(),
        (host.div_ceil(512) + STORE_DEVICE_MARGIN_BYTES, host)
    );
    options.disk_limit = Some(0);
    assert_eq!(options.headroom_bytes(2, &layout).unwrap().1, 16 << 30);
}

#[test]
fn ordinary_input_share_preserves_checkpoint_minimum_and_mapping() {
    assert_eq!(input_memory_bytes(0, 4096, true), 0);
    assert_eq!(input_memory_bytes(8192, 4096, true), 0);
    assert_eq!(input_memory_bytes(1 << 30, 128 << 20, true), 128 << 20);
    assert_eq!(input_memory_bytes(1 << 30, 1000 << 20, true), 24 << 20);
    assert_eq!(input_memory_bytes(1 << 30, 128 << 20, false), 0);
    let layout = Layout {
        page_tokens: 64,
        ranks: Vec::new(),
        page_record_bytes: 4096,
        row_record_bytes: 4096,
    };
    let options = PrefixCacheOffload {
        memory_bytes: Some(1 << 30),
        ..Default::default()
    };
    let ordinary = input_memory_bytes(1 << 30, slab::SlabPool::minimum_bytes(&layout), true);
    let (device, host) = options.headroom_with_inputs(2, &layout, true).unwrap();
    assert_eq!(host, 1 << 30);
    assert_eq!(
        device,
        ((1u64 << 30) - ordinary).div_ceil(2).div_ceil(512) + STORE_DEVICE_MARGIN_BYTES
    );
}

#[test]
fn pooled_row_warning_uses_the_allocated_tier_and_selected_concurrency() {
    let geometry = vec![crate::testing::geometry(64, &[("kv", 4096)], 4096)];
    let layout = Layout::new(&geometry).unwrap();
    let mut options = PrefixCacheOffload {
        memory_bytes: Some(slab::SlabPool::minimum_bytes(&layout)),
        pool: Some(pool("unix:/run/selene.sock")),
        ..Default::default()
    };
    let build = |options: &PrefixCacheOffload| {
        let Preparation::Store(prepared) = prepare(
            options,
            std::path::Path::new("model.safetensors"),
            [0; 32],
            geometry.clone(),
            vec![(0, 0)],
        )
        .unwrap() else {
            panic!("no store")
        };
        prepared
    };
    let prepared = build(&options);
    let rows = prepared.memory_pool.as_ref().unwrap().row_count();
    assert!(prepared.pool_row_warning(rows / 2).is_none());
    let warning = prepared.pool_row_warning(rows / 2 + 1).unwrap();
    assert!(warning.contains(&format!("{rows} rows")));
    assert!(warning.contains("publications may be skipped"));
    options.memory_bytes = Some(0);
    assert!(build(&options).pool_row_warning(32).is_none());
}
