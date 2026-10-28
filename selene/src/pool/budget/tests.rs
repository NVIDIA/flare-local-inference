use super::*;
fn plan(arena: u64, staging: u64) -> Vec<PlannedRegion> {
    vec![
        PlannedRegion {
            name: "arena",
            bytes: arena,
        },
        PlannedRegion {
            name: "staging",
            bytes: staging,
        },
    ]
}

#[test]
fn approval_charges_whole_regions_and_each_gpu_before_registration() {
    let plan = plan(1 << 30, 1 << 29);
    let (host, device) = Budget::costs(plan.iter().map(|region| region.bytes)).unwrap();
    assert_eq!((host, device), (3 << 29, 3 << 20));
    let budget = Budget {
        host: Ok(host),
        devices: vec![(device, false); 2],
    };
    assert!(budget.approve(&plan).is_ok());
    assert_eq!(budget.charges(host, device), vec![device; 2]);
    assert!(
        Budget {
            host: Ok(host - 1),
            ..budget
        }
        .approve(&plan)
        .unwrap_err()
        .contains("host budget")
    );
    let budget = Budget {
        host: Ok(host),
        devices: vec![(device, false), (device - 1, false)],
    };
    assert!(budget.approve(&plan).unwrap_err().contains("GPU 1"));
    assert!(Budget::costs(self::plan(u64::MAX, 1).iter().map(|region| region.bytes)).is_err());
}

#[test]
fn remote_staging_and_shared_memory_use_the_same_budget() {
    let plan = vec![PlannedRegion {
        name: "data-staging",
        bytes: 4097,
    }];
    assert_eq!(
        Budget::costs(plan.iter().map(|region| region.bytes)).unwrap(),
        (4097, 9)
    );
    let budget = Budget {
        host: Ok(4097),
        devices: vec![(2058, true), (2058, true), (9, false)],
    };
    assert_eq!(budget.charges(4097, 9), [2058, 2058, 9]);
    assert!(budget.approve(&plan).is_ok());
    assert!(
        Budget {
            host: Ok(4097),
            devices: vec![(2057, true), (2058, true)]
        }
        .approve(&plan)
        .is_err()
    );
    assert!(
        Budget {
            host: Err("no memory observation".into()),
            devices: Vec::new()
        }
        .approve(&plan)
        .unwrap_err()
        .contains("no memory observation")
    );
}
