//! Attach a pool before the host allocates its final cache storage.
use super::{
    AgentConfig, Budget, DeferredRecords, PoolClient, PoolConfig, PoolOptions, PoolPeers,
    PooledStore, RankAgent, RegisteredRegion, pool_address, pool_scope,
};
use crate::{
    CacheGeometry, HostRegistrar, HostStore, Layout, Registration, Result, SplitMapping,
    plan::Identity,
};
use crate::{
    PageCopier,
    copier::{Job, Reply},
};
use std::sync::{Arc, Mutex};

/// Registration precedes cache sizing; copying starts with the final cache geometry.
struct Copier {
    layout: Layout,
    transfers: Option<Box<dyn crate::PageCopier>>,
}

impl PageCopier for Copier {
    fn layout(&self) -> &Layout {
        &self.layout
    }
    fn submit(&mut self, job: Job) -> crate::Result<()> {
        self.transfers
            .as_mut()
            .ok_or("pool copies are not initialized")?
            .submit(job)
    }
    fn poll(&mut self, replies: &mut Vec<Reply>) {
        if let Some(transfers) = &mut self.transfers {
            transfers.poll(replies);
        }
    }
    fn in_flight(&self) -> usize {
        self.transfers
            .as_ref()
            .map_or(0, |copier| copier.in_flight())
    }
    fn close(&mut self) -> crate::Result<()> {
        self.transfers
            .as_mut()
            .map_or(Ok(()), |copier| copier.close())
    }
}

pub struct PoolBuilder<'a> {
    pub options: &'a PoolOptions,
    pub identity: [u8; 32],
    pub geometry: &'a [CacheGeometry],
    pub gpu_uuids: Vec<[u8; 16]>,
    pub registration_context: usize,
    pub budget: Budget,
    pub peers: Option<PoolPeers>,
    pub disk: bool,
}

pub struct PreparedPool {
    pub client: Option<PoolClient>,
    pub agent: Option<RankAgent>,
    pub records: Option<DeferredRecords>,
    pub identity: Identity,
    listener: Option<(std::net::TcpListener, [u8; 16])>,
    layout: Layout,
    copier: Arc<Mutex<Copier>>,
}

impl PoolBuilder<'_> {
    pub fn start(
        self,
        registrar: Arc<dyn HostRegistrar + Send + Sync>,
    ) -> Result<(PreparedPool, Vec<u64>)> {
        let Self {
            options,
            identity,
            geometry,
            gpu_uuids,
            registration_context,
            budget,
            mut peers,
            disk,
        } = self;
        let copier = Arc::new(Mutex::new(Copier {
            layout: Layout::new(geometry)?,
            transfers: None,
        }));
        let listener = peers.as_mut().and_then(|peers| peers.listener.take());
        let registrar: Arc<dyn HostRegistrar + Send + Sync> = Arc::new(PoolRegistrar(registrar));
        let devices = budget.devices.clone();
        let all_ranks = if let Some(peers) = &peers {
            vec![geometry[0].clone(); peers.gpu_uuids.len()]
        } else {
            geometry.to_vec()
        };
        let scope = pool_scope(identity, &all_ranks)?;
        let identity = crate::plan::Identity {
            model: scope.model_identity,
            layout: scope.layout_fingerprint,
        };
        let config = PoolConfig {
            address: pool_address(&options.address)?,
            ca: options.ca.clone(),
            scope,
            gpu_uuids: peers
                .as_ref()
                .map_or(gpu_uuids, |peers| peers.gpu_uuids.clone()),
            export_slots: 2,
            claim_slots: 2,

            registration_context,
            approve: Some(Arc::new(move |plan| budget.approve(plan))),
        };
        let (client, agent, records) = match peers {
            Some(mut peers) if peers.rank != 0 => {
                let records = disk.then(DeferredRecords::default);
                let agent = RankAgent::start(AgentConfig {
                    address: config.address,
                    ca: config.ca,
                    scope: config.scope,
                    rank: peers.rank,
                    world: all_ranks.len(),
                    link: peers.links.remove(0),
                    copier: copier.clone(),
                    registrar,
                    registration_context,
                    gpu_uuid: config.gpu_uuids[peers.rank],
                    records: records
                        .clone()
                        .map(|records| Box::new(records) as Box<dyn crate::pool::RankRecords>),
                    approve: config.approve,
                })?;
                (None, Some(agent), records)
            }
            peers => {
                let client = PoolClient::start(
                    config,
                    peers.map(|peers| peers.links),
                    copier.clone(),
                    registrar,
                    super::Roots::System,
                    super::STARTUP_WAIT,
                )?;
                (Some(client), None, None)
            }
        };
        let mut prepared = PreparedPool {
            client,
            agent,
            records,
            listener,
            layout: copier.lock().unwrap().layout().clone(),
            copier: copier.clone(),
            identity,
        };
        if prepared.registered_regions().is_empty() {
            // Finish attachment before caches consume its startup allowance.
            prepared.close()?;
        }
        let (host, device) = prepared.registration_costs()?;
        let charges = Budget {
            host: Ok(0),
            devices,
        }
        .charges(host, device);
        Ok((prepared, charges))
    }
}

impl PreparedPool {
    pub fn prepare_peer(
        &mut self,
        prepared: &crate::offload::Prepared,
        factory: &dyn crate::offload::CopierFactory,
    ) -> Result<()> {
        if self.agent.is_some() {
            self.start_copies(&prepared.contexts, prepared.geometry.clone(), factory)?;
            if let Some(records) = &self.records
                && let Some(directory) = &prepared.directory_lock
            {
                records.bind(
                    prepared
                        .resolved
                        .directory
                        .clone()
                        .ok_or("a rank's records need a disk directory")?,
                    prepared.identity,
                    prepared.fingerprint,
                    prepared.geometry.clone(),
                    directory.clone(),
                )?;
            }
        }
        Ok(())
    }

    pub fn start_copies(
        &self,
        contexts: &[(usize, i32)],
        geometry: Vec<CacheGeometry>,
        factory: &dyn crate::offload::CopierFactory,
    ) -> Result<()> {
        let mut copier = self.copier.lock().unwrap();
        if Layout::new(&geometry)? != copier.layout {
            return Err("pool registration and final cache layouts disagree".into());
        }
        copier.transfers = Some(factory.unmapped(contexts, geometry)?);
        Ok(())
    }

    fn registration_costs(&self) -> Result<(u64, u64)> {
        Ok(Budget::costs(
            self.registered_regions()
                .iter()
                .map(|region| region.bytes as u64),
        )?)
    }
    pub fn summary(&self) -> Result<String> {
        let (host, device) = self.registration_costs()?;
        Ok(if self.approved() {
            format!("{host} host bytes · {device} device bytes per GPU")
        } else {
            "local tiers only".into()
        })
    }
    fn registered_regions(&self) -> &[RegisteredRegion] {
        match (&self.client, &self.agent) {
            (Some(client), _) => client.registered_regions(),
            (_, Some(agent)) => agent.registered_regions(),
            _ => &[],
        }
    }
    pub fn take_listener(&mut self) -> Option<(std::net::TcpListener, [u8; 16])> {
        self.listener.take()
    }
    pub fn approved(&self) -> bool {
        !self.registered_regions().is_empty()
    }
    pub fn close(&mut self) -> Result<()> {
        if let Some(client) = &mut self.client {
            client.close()?;
        }
        if let Some(agent) = &mut self.agent {
            agent.close()?;
        }
        Ok(())
    }
    pub fn attach(
        self,
        local: Option<Box<dyn crate::store::TierStore>>,
    ) -> Result<Box<dyn HostStore>> {
        Ok(Box::new(PooledStore::new(
            super::local::Tiers::new(local),
            self.client
                .ok_or("a pool rank agent cannot serve rank zero's store")?,
            self.identity,
            &self.layout,
        )))
    }
}

/// Registers the pool's shared-memory regions. The pool's copier names their
/// host addresses directly in copies, so a registration whose device address
/// differs from its host address is refused.
struct PoolRegistrar(Arc<dyn HostRegistrar + Send + Sync>);

impl HostRegistrar for PoolRegistrar {
    fn register(
        &self,
        context: usize,
        address: usize,
        bytes: usize,
    ) -> Result<Box<dyn Registration>> {
        let registration = self.0.register(context, address, bytes)?;
        if registration.device_address() != address as u64 {
            return Err(crate::Error::new(
                "pool memory needs identical host and device addresses",
            ));
        }
        Ok(registration)
    }

    fn map_split(&self, _requests: &[(usize, i32, usize)]) -> Result<Box<dyn SplitMapping>> {
        Err(crate::Error::new("the pool does not use split mappings"))
    }
}
