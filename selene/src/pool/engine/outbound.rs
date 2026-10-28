//! Window copying shared by exports and puts.
use std::collections::{HashMap, HashSet};

use super::{
    super::{
        exporter::Checksums,
        net::{Channel, Filled},
        wire,
    },
    Engine, Ranks,
    hostfill::{FillJob, FillOwner},
    ranks::{PeerSet, PeerWindow, XferOwner},
};
use crate::{Error, Result};

#[derive(Default)]
pub(super) struct Outbound {
    pub channel: Option<Channel>,
    pub channel_open: bool,
    pub slots: HashMap<u32, u32>,
    pub sends_outstanding: u32,
    pub peers: PeerSet,
    pub host_filling: HashSet<u32>,
    pub host_sums: HashMap<u32, Checksums>,
    pub jobs_outstanding: u32,
}

impl Outbound {
    pub fn release_slots(&mut self, staging: Option<&super::super::net::Staging>) {
        if self.channel_open || self.jobs_outstanding != 0 {
            return;
        }
        if let Some(staging) = staging {
            for (_, slot) in self.slots.drain() {
                staging.release(slot);
            }
        }
    }

    pub fn running(&self, index: u32) -> bool {
        self.host_filling.contains(&index)
            || self
                .peers
                .windows
                .get(&index)
                .is_some_and(|w| !w.left.is_empty())
    }

    pub fn fill_host(
        &mut self,
        owner: FillOwner,
        spec: wire::WindowSpec,
        address: usize,
        slot: Option<u32>,
        ranks: Option<&Ranks>,
        work: Box<dyn FnOnce(&mut [u8]) -> Result<Checksums> + Send>,
    ) -> FillJob {
        self.jobs_outstanding += 1;
        self.host_filling.insert(spec.index);
        let scatter = ranks.filter(|_| slot.is_none()).map(|r| r.layout.clone());
        FillJob {
            owner,
            window: spec.index,
            work: Box::new(move || {
                // The transfer keeps the mapped window until this completion is collected.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(address as *mut u8, spec.length as usize)
                };
                match scatter {
                    Some(layout) => {
                        let mut packed =
                            vec![0; layout.slice_bytes(0, spec.page_count, spec.row) as usize];
                        let sums = work(&mut packed)?;
                        layout.scatter(0, dst, spec.page_count, spec.row, &packed)?;
                        Ok(sums)
                    }
                    None => work(dst),
                }
            }),
        }
    }

    pub fn fan_out(&mut self, mut ranks: Option<&mut Ranks>, spec: wire::WindowSpec) -> Result<()> {
        let shared: Vec<_> = self
            .peers
            .parts
            .iter()
            .filter(|p| p.shared && p.done.is_none())
            .map(|p| (p.rank, p.xfer))
            .collect();
        if !shared.is_empty() {
            self.peers.windows.insert(
                spec.index,
                PeerWindow {
                    left: shared.iter().map(|(rank, _)| *rank).collect(),
                    sums: Vec::new(),
                },
            );
        }
        for (rank, xfer) in shared {
            if !ranks.as_mut().is_some_and(|ranks| {
                ranks.send_window(
                    rank,
                    super::super::rank::Window {
                        xfer,
                        spec: Some(spec),
                        sums: None,
                    },
                )
            }) {
                return Err(Error::new(format!("rank {rank}'s link is down")));
            }
        }
        Ok(())
    }

    pub fn filled(
        &mut self,
        spec: wire::WindowSpec,
        sums: Checksums,
        channel: bool,
        ranked: bool,
    ) -> Result<Option<(Checksums, Vec<wire::RankSums>)>> {
        if channel {
            let sent = self.slots.get(&spec.index).is_some_and(|slot| {
                self.channel.as_ref().is_some_and(|channel| {
                    channel.send(Filled {
                        index: spec.index,
                        row_xxh3: sums.row,
                        page_xxh3: sums.pages.clone(),
                        slot: *slot,
                        length: spec.length,
                    })
                })
            });
            if !sent {
                return Err(Error::new("the data channel is gone"));
            }
            self.sends_outstanding += 1;
            return Ok(None);
        }
        let mut rank_sums = Vec::new();
        if ranked {
            let peers = self.peers.windows.remove(&spec.index);
            for (rank, sums) in
                std::iter::once((0, sums)).chain(peers.into_iter().flat_map(|p| p.sums))
            {
                rank_sums.push(wire::RankSums {
                    rank: rank as u32,
                    page_xxh3: sums.pages,
                    row_xxh3: sums.row.unwrap_or(0),
                    has_row: sums.row.is_some(),
                });
            }
            return Ok(Some((
                Checksums {
                    pages: Vec::new(),
                    row: None,
                },
                rank_sums,
            )));
        }
        Ok(Some((sums, rank_sums)))
    }
}

impl Engine {
    pub(super) fn collect_outbound_fills(&mut self) {
        for done in self.collect_fills() {
            let owner = match done.owner {
                FillOwner::Export(id) => XferOwner::Export(id),
                FillOwner::Put(id) => XferOwner::Put(id),
            };
            let Some(out) = self.outbound(owner) else {
                continue;
            };
            out.jobs_outstanding = out.jobs_outstanding.saturating_sub(1);
            out.host_filling.remove(&done.window);
            match done.result {
                Ok(sums) => {
                    out.host_sums.insert(done.window, sums);
                }
                Err(error) => self.fail_outbound(owner, error),
            }
            self.drive_outbound(owner, Some(done.window));
        }
    }

    pub(super) fn window_shape(&self, channel: bool) -> (u64, u64) {
        match &self.ranks {
            Some(ranks) if channel => (ranks.layout.rank(0).page_len, ranks.layout.rank(0).row_len),
            _ => (self.st.page_record_bytes(), self.st.row_record_bytes()),
        }
    }

    pub(super) fn window_sums(&self, spec: wire::WindowSpec, channel: bool) -> Result<Checksums> {
        let memory = self.memory();
        let bytes = memory
            .as_ref()
            .and_then(|m| m.region(spec.region()))
            .and_then(|r| r.bytes(spec.offset, spec.length))
            .ok_or_else(|| Error::new("window outside its region"))?;
        let layout = super::super::slice::SliceLayout::new(&self.st.scope.geometry)?;
        let sums = match (&self.ranks, channel) {
            (Some(_), true) => layout.packed_sums(0, bytes, spec.page_count, spec.row)?,
            (Some(_), false) => layout.slice_sums(0, bytes, spec.page_count, spec.row)?,
            (None, _) => layout.whole_sums(bytes, spec.page_count, spec.row)?,
        };
        Ok(sums)
    }

    pub(super) fn outbound_sent(&mut self, owner: XferOwner, index: u32, slot: u32) {
        if let Some(out) = self.outbound(owner) {
            out.sends_outstanding = out.sends_outstanding.saturating_sub(1);
            if out.slots.get(&index) == Some(&slot) {
                out.slots.remove(&index);
                self.free_slot(slot);
            }
        }
        self.drive_outbound(owner, None);
    }

    pub(super) fn outbound(&mut self, owner: XferOwner) -> Option<&mut Outbound> {
        match owner {
            XferOwner::Export(id) => self.source.outbound(id),
            XferOwner::Put(id) => self.writer.outbound(id),
            XferOwner::Claim(_) => None,
        }
    }

    pub(super) fn cancel_outbound_peers(&mut self, owner: XferOwner) {
        let cancels = self
            .outbound(owner)
            .map(|out| out.peers.take_uncancelled())
            .unwrap_or_default();
        if let Some(ranks) = &mut self.ranks {
            for xfer in cancels {
                ranks.cancel(xfer);
            }
        }
    }

    pub(super) fn fail_outbound(&mut self, owner: XferOwner, error: Error) {
        match owner {
            XferOwner::Export(id) => {
                self.source_failed(id, error);
            }
            XferOwner::Put(id) => self.fail_put(id, error),
            XferOwner::Claim(_) => unreachable!(),
        }
    }

    pub(super) fn drive_outbound(&mut self, owner: XferOwner, index: Option<u32>) {
        match owner {
            XferOwner::Export(id) => {
                if let Some(index) = index {
                    self.export_after_peer_window(id, index);
                }
                self.drive_export(id);
            }
            XferOwner::Put(id) => {
                if let Some(index) = index {
                    self.put_after_peer_window(id, index);
                }
                self.drive_put(id);
            }
            XferOwner::Claim(_) => unreachable!(),
        }
    }
}
