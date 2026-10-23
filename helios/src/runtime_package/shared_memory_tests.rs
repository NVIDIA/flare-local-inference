use super::*;
use crate::ReplayWorker;
use std::time::Duration;

#[test]
#[ignore = "requires FLARE_SHARED_MEMORY_TEST_PACKAGE exported by shared_memory_package_fixture.py on the assigned GPU"]
fn captured_shared_memory_exceeding_function_allowance_replays() {
    let root =
        std::path::PathBuf::from(std::env::var_os("FLARE_SHARED_MEMORY_TEST_PACKAGE").unwrap());
    let owner =
        unsafe { NativeResources::load(test_support::authorize(root.join("package")), 0).unwrap() };
    let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1).unwrap() };
    for _ in 0..4 {
        for (index, name) in owner.graph_names().iter().enumerate() {
            owner.upload("output", &(-1i32).to_le_bytes()).unwrap();
            worker.submit(index, 2).unwrap();
            let result = worker.poll(Duration::from_secs(10)).unwrap().unwrap();
            assert!(result.error.is_none(), "{:?}", result.error);
            let value = i32::from_le_bytes(owner.download("output").unwrap().try_into().unwrap());
            let shared: i32 = name.parse().unwrap();
            assert_eq!(value, shared + (shared - 1) % 251, "{name}");
        }
    }
    assert!(worker.close().drained);
}
