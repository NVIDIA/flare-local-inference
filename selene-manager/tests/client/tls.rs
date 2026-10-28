use super::*;
use flare_selene::testing::{self as net, Roots};

fn roots(manager: &Manager) -> Roots {
    let (_, ca) = remote(&manager).unwrap();
    Roots::Fixed(net::trusted(ca.as_deref(), &Roots::System).unwrap())
}

fn other_roots() -> Roots {
    let cert = selene_manager::testing::certificate();
    Roots::Fixed(vec![cert.cert.der().clone()])
}

fn connect(config: PoolConfig, roots: Roots) -> PoolClient {
    let (copier, _) = SimulatedGpu::shared(&geo()).unwrap();
    PoolClient::start(
        config,
        None,
        copier,
        Arc::new(CountingRegistrar::default()),
        roots,
        Duration::from_secs(2),
    )
    .unwrap()
}

fn ready(client: &PoolClient) {
    wait("the client to be ready", || {
        client.status() == PoolStatus::Ready
    });
}

#[test]
fn a_client_with_a_ca_connects_to_a_tls_manager() {
    let manager = network_manager(true);
    let mut client = connect(config(&manager, 1), Roots::System);
    ready(&client);
    assert!(!client.registered_regions().is_empty());
    client.close().unwrap();
}

#[test]
fn a_client_without_a_ca_file_trusts_the_hosts_store() {
    let manager = network_manager(true);
    let mut config = config(&manager, 1);
    config.ca = None;
    let mut client = connect(config, roots(&manager));
    ready(&client);
    client.close().unwrap();
}

#[test]
fn an_untrusted_manager_disables_the_pool_without_registration() {
    let manager = network_manager(true);
    let mut config = config(&manager, 1);
    config.ca = None;
    let mut client = connect(config, other_roots());
    let PoolStatus::Disabled(reason) = client.status() else {
        panic!("{:?}", client.status());
    };
    assert!(
        reason.contains("cannot verify the pool manager over TLS"),
        "{reason}"
    );
    assert!(client.registered_regions().is_empty());
    client.close().unwrap();
}

#[test]
fn a_ca_file_overrides_the_hosts_store() {
    let manager = network_manager(true);
    let mut client = connect(config(&manager, 1), other_roots());
    ready(&client);
    client.close().unwrap();
}

#[test]
fn a_bare_address_uses_plaintext() {
    let manager = network_manager(false);
    let mut config = config(&manager, 1);
    let PoolAddress::TcpPlaintext(address) = &config.address else {
        panic!()
    };
    config.address = flare_selene::testing::pool_address(address).unwrap();
    assert!(matches!(config.address, PoolAddress::TcpPlaintext(_)));
    let mut client = connect(config, Roots::System);
    ready(&client);
    client.close().unwrap();
}

#[test]
fn a_client_with_a_ca_never_downgrades_to_plaintext() {
    let manager = network_manager(false);
    let mut config = config(&manager, 1);
    let PoolAddress::TcpPlaintext(address) = config.address else {
        panic!()
    };
    config.address = flare_selene::testing::pool_address(&format!("tls:{address}")).unwrap();
    let cert = selene_manager::testing::certificate();
    let ca = manager.base.join("client-ca.pem");
    std::fs::write(&ca, cert.cert.pem()).unwrap();
    config.ca = Some(ca);
    let mut client = connect(config, Roots::System);
    assert!(matches!(client.status(), PoolStatus::Disabled(_)));
    assert!(client.registered_regions().is_empty());
    client.close().unwrap();
}

#[test]
fn a_plaintext_client_refuses_a_tls_manager() {
    let manager = network_manager(true);
    let mut config = config(&manager, 1);
    let PoolAddress::Tcp(address) = &config.address else {
        panic!()
    };
    config.address = flare_selene::testing::pool_address(address).unwrap();
    config.ca = None;
    let mut client = connect(config, Roots::System);
    assert!(matches!(client.status(), PoolStatus::Disabled(_)));
    assert!(client.registered_regions().is_empty());
    client.close().unwrap();
}

#[test]
fn a_ca_is_refused_for_plaintext_and_unix_addresses() {
    let manager = network_manager(false);
    for address in [
        config(&manager, 1).address,
        PoolAddress::Unix(manager.config.socket.clone()),
    ] {
        let mut config = config(&manager, 1);
        config.address = address;
        config.ca = Some(manager.base.join("client-ca.pem"));
        let (copier, _) = SimulatedGpu::shared(&geo()).unwrap();
        let result = PoolClient::start(
            config,
            None,
            copier,
            Arc::new(CountingRegistrar::default()),
            Roots::System,
            Duration::from_secs(2),
        );
        let Err(error) = result else {
            panic!("a CA must require TLS")
        };
        assert!(
            error
                .to_string()
                .contains("a CA certificate applies to an encrypted network address")
        );
    }
}
