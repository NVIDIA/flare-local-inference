use flare_helios::{ReplayWorker, runtime_package::NativeResources};
use serde::Deserialize;
use std::{collections::HashMap, path::PathBuf, time::Duration};

mod tp4_collective;

#[derive(Deserialize)]
struct Reference {
    allocation: String,
    offset: usize,
}
#[derive(Deserialize)]
struct Expected {
    reference: Reference,
    bytes: Vec<u8>,
    addresses: HashMap<String, usize>,
    zero_reference: Reference,
    zero_bytes: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("expected fixture directory")?,
    );
    if std::env::args().nth(2).as_deref() == Some("--tp4-collective") {
        return tp4_collective::run(root);
    }
    let expected: Expected = serde_json::from_slice(&std::fs::read(root.join("expected.json"))?)?;
    let owner = unsafe {
        NativeResources::load(
            flare_helios::runtime_package::test_support::authorize(root.join("package")),
            0,
        )?
    };
    for (name, address) in expected.addresses {
        assert_eq!(owner.allocation(&name)?.address, address);
    }
    owner.execution_manifest()?.validate()?;
    let transient = owner.download(&expected.zero_reference.allocation)?;
    assert!(
        transient
            [expected.zero_reference.offset..expected.zero_reference.offset + expected.zero_bytes]
            .iter()
            .all(|byte| *byte == 0)
    );
    let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1)? };
    let ticket = worker.submit(0, 7)?;
    let result = worker
        .poll(Duration::from_secs(30))?
        .ok_or("replay timed out")?;
    assert_eq!(result.ticket, ticket);
    if let Some(error) = result.error {
        return Err(error.into());
    }
    let bytes = owner.download(&expected.reference.allocation)?;
    assert_eq!(
        &bytes[expected.reference.offset..expected.reference.offset + expected.bytes.len()],
        &expected.bytes
    );
    assert!(worker.close().drained);
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    assert!(
        !maps.contains("libpython")
            && !maps.contains("libtorch")
            && !maps.contains("site-packages")
    );
    println!(
        "native fixed-address replay passed with mapped host input and reconstructed execution contract"
    );
    Ok(())
}
