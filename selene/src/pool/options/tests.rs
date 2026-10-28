use super::*;
use crate::Layout;
#[test]
fn pool_addresses_default_to_plaintext_and_allow_explicit_tls() {
    use crate::pool::PoolAddress;
    assert_eq!(
        pool_address("unix:/run/selene.sock").unwrap(),
        PoolAddress::Unix("/run/selene.sock".into())
    );
    for address in ["cache.example:7000", "[::1]:7000"] {
        assert_eq!(
            pool_address(address).unwrap(),
            PoolAddress::TcpPlaintext(address.into())
        );
        assert_eq!(
            pool_address(&format!("tls:{address}")).unwrap(),
            PoolAddress::Tcp(address.into())
        );
        assert!(pool_address(&format!("plain:{address}")).is_err());
    }
}

#[test]
fn the_pool_scope_follows_the_record_layout_and_matches_the_model() {
    let geometry = vec![
        crate::testing::geometry(64, &[("k", 4096), ("v", 5000)], 4096),
        crate::testing::geometry(64, &[("k", 4096), ("v", 5000)], 4096),
    ];
    let layout = Layout::new(&geometry).unwrap();
    let scope = pool_scope([9; 32], &geometry).unwrap();
    assert_eq!(scope.model_identity, [9; 32]);
    assert_eq!(scope.layout_fingerprint, geometry::fingerprint(&geometry));
    assert_eq!(scope.geometry.page_record_bytes, layout.page_record_bytes);
    assert_eq!(scope.geometry.row_record_bytes, layout.row_record_bytes);
    assert_eq!(scope.geometry.ranks.len(), 2);
    assert_eq!(
        scope.geometry.ranks[0].plane_bytes.iter().sum::<u64>() * 2,
        layout.page_record_bytes
    );
    assert_ne!(
        scope.layout_fingerprint,
        geometry::fingerprint(&geometry[..1])
    );
}
