//! A session without shared memory: staging for its windows, and the events of
//! the data-channel tasks that move them.

use std::sync::Arc;

use crate::{Error, Result};

use super::{
    super::{
        net::{self, Dial, Event, Id, Staging},
        shm::{Region, Regions},
        wire,
    },
    Engine,
};

/// Where the open session's windows are.
pub(crate) enum Memory {
    Shm(Arc<Regions>),
    Net(Arc<Staging>),
}

impl Memory {
    pub(crate) fn region(&self, which: wire::Region) -> Option<&Region> {
        match self {
            Memory::Shm(regions) => regions.get(which),
            Memory::Net(staging) => (which == wire::Region::Staging).then(|| staging.region()),
        }
    }
}

impl Engine {
    /// The registered slots, made on first use and kept across sessions. A
    /// window never exceeds `max_window_bytes`, except a lone page or row.
    pub(crate) fn net_staging(&mut self, limits: &wire::Limits) -> Result<Arc<Staging>> {
        if let Some(staging) = &self.net {
            return Ok(staging.clone());
        }
        let (slots, slot_bytes) = self.net_shape(limits);
        let staging = Arc::new(Staging::new(
            self.registrar(),
            self.st.context,
            slots,
            slot_bytes,
        )?);
        self.net = Some(staging.clone());
        Ok(staging)
    }

    /// Slots and bytes per slot of the data channel's staging.
    pub(crate) fn net_shape(&self, limits: &wire::Limits) -> (u32, u64) {
        let slots =
            super::super::limits::staging_slots(limits, self.st.claim_slots, self.st.export_slots);
        let slot_bytes = limits
            .max_window_bytes
            .max(self.st.page_record_bytes())
            .max(self.st.row_record_bytes());
        (slots, slot_bytes)
    }

    /// The open session reaches the manager over the network: its windows pass
    /// through staging and its transfers are opened by `dial`.
    pub(crate) fn set_dial(&mut self, dial: Arc<dyn Dial>) {
        self.dial = Some(dial);
    }

    /// The memory the open session's windows are in, if it can copy at all.
    pub(crate) fn memory(&self) -> Option<Memory> {
        let sess = self.sess.as_ref()?;
        match (&sess.regions, &self.dial, &self.net) {
            (Some(regions), _, _) => Some(Memory::Shm(regions.clone())),
            (None, Some(_), Some(staging)) => Some(Memory::Net(staging.clone())),
            _ => None,
        }
    }

    /// Open the data-channel transfer a grant names.
    pub(crate) fn open_channel(
        &self,
        id: Id,
        reference: &wire::DataChannelRef,
    ) -> Result<net::Channel> {
        let dial = self
            .dial
            .as_ref()
            .ok_or_else(|| Error::new("this session has no data channel"))?;
        dial.open(net::Open {
            id,
            reference: reference.clone(),
        })
        .map_err(|e| Error::caused_by("cannot open the data channel", e))
    }

    /// Give a slot back to the staging.
    pub(crate) fn free_slot(&self, slot: u32) {
        if let Some(staging) = &self.net {
            staging.release(slot);
        }
    }

    /// Apply what the data-channel tasks reported since the last call.
    pub(crate) fn collect_net(&mut self) {
        for event in self.net_events.take() {
            match event {
                Event::Window {
                    claim,
                    window,
                    slot,
                } => self.on_net_window(claim, window, slot),
                Event::End { claim } => self.on_net_end(claim),
                Event::Granted { id, shape, slot } => match id {
                    Id::Export(export) => self.on_export_grant(export, shape, slot),
                    Id::Put(put) => self.on_net_grant(put, shape, slot),
                    Id::Claim(_) => self.free_slot(slot),
                },
                Event::Sent { id, index, slot } => match id {
                    Id::Export(export) => {
                        self.outbound_sent(super::ranks::XferOwner::Export(export), index, slot)
                    }
                    Id::Put(put) => {
                        self.outbound_sent(super::ranks::XferOwner::Put(put), index, slot)
                    }
                    Id::Claim(_) => self.free_slot(slot),
                },
                Event::Failed { id, error } => self.on_net_failed(id, error),
                Event::Closed { id } => self.on_net_closed(id),
            }
        }
    }
}
