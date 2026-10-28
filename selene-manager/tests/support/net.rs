//! A manager with a network listener and a data listener, started the way the binary
//! starts it, for tests of sessions that reach it from another host.

use super::pool::Pool;
use selene_manager::{config::Config as NetConfig, store::Config, testing::Manager};
use std::{net::SocketAddr, path::PathBuf};
use tonic::transport::Endpoint;

pub struct Net {
    pub socket: PathBuf,
    pub address: SocketAddr,
    pub data: SocketAddr,
    pub base: PathBuf,
    pub name: String,
    _manager: Manager,
}

pub use selene_manager::testing::free_address;

impl Net {
    pub fn start(tweak: impl FnOnce(&mut NetConfig)) -> Self {
        Self::start_with(tweak, |_| {})
    }

    pub fn start_with(
        tweak: impl FnOnce(&mut NetConfig),
        store_tweak: impl FnOnce(&mut Config),
    ) -> Self {
        let base = std::env::temp_dir().join(format!(
            "selene-net-{}-{}",
            std::process::id(),
            super::instance_name("n")
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let socket = base.join("run/m.sock");
        let address = free_address();
        let data = free_address();
        let mut store = Config::dram_only(socket.clone(), 8 * 1024 * 1024);
        let net_name = super::instance_name("net");
        store.name = net_name.clone();
        store_tweak(&mut store);
        let mut net = store;
        net.listen = Some(address);
        net.listen_data = Some(data);
        tweak(&mut net);
        let data = net.data_address().unwrap().unwrap();
        let mut manager = Manager::open_config(net);
        manager.run();
        Self {
            socket,
            address,
            data,
            base,
            name: net_name,
            _manager: manager,
        }
    }

    pub async fn unix(&self) -> Pool {
        Pool::connect(&self.socket).await
    }

    pub async fn tcp(&self) -> Pool {
        let channel = Endpoint::from_shared(format!("http://{}", self.address))
            .unwrap()
            .connect()
            .await
            .expect("connect over TCP");
        Pool::over(channel, &self.socket).await
    }
}
