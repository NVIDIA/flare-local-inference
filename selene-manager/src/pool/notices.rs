//! Live index hints over one ordered control stream.
use super::{Checked, Notice, Violation, core::Core, entry_from, index::SourceId, notice_from};
use crate::proto::v2::{self, server_message::M};

impl Core {
    pub(super) fn notices(&mut self, sid: u64, batch: v2::NoticeBatch) -> Checked<()> {
        if batch.notices.len() > self.cfg.limits.max_notices_per_batch as usize {
            return Err(Violation::new("notice batch exceeds max_notices_per_batch"));
        }
        let Some(session) = self.sessions.get(&sid) else {
            return Ok(());
        };
        if session.need_resync {
            return Ok(());
        }
        let (scope, source) = (session.scope, SourceId::Instance(session.instance));
        for notice in &batch.notices {
            self.apply_notice(scope, source, notice_from(notice)?);
        }
        self.bound_hints(sid);
        Ok(())
    }
    pub(super) fn apply_notice(&mut self, scope: [u8; 32], source: SourceId, notice: Notice) {
        let Some(state) = self.scopes.get_mut(&scope) else {
            return;
        };
        let index = &mut state.index;
        match notice {
            Notice::Saved(entry) => {
                index.upsert(source, entry);
            }
            Notice::Spilled { to, from_tier } => {
                let key = to.key;
                index.upsert(source, to);
                index.remove(source, key, from_tier);
            }
            Notice::Evicted { key, tier, heir } => {
                index.remove(source, key, tier);
                if let Some(heir) = heir {
                    index.upsert(source, heir);
                }
            }
        }
    }

    fn bound_hints(&mut self, sid: u64) {
        let Some(session) = self.sessions.get(&sid) else {
            return;
        };
        if self.scopes.get(&session.scope).is_some_and(|s| {
            s.index.source_len(SourceId::Instance(session.instance))
                > self.cfg.limits.max_resync_entries as usize
        }) {
            self.demand_resync(sid, v2::ResyncReason::ResyncOverflow);
        }
    }

    pub(super) fn demand_resync(&mut self, sid: u64, reason: v2::ResyncReason) {
        let Some(session) = self.sessions.get_mut(&sid) else {
            return;
        };
        session.need_resync = true;
        session.syncing = false;
        if let Some(scope) = self.scopes.get_mut(&session.scope) {
            scope.index.clear(SourceId::Instance(session.instance));
        }
        self.send(
            sid,
            M::ResyncRequired(v2::ResyncRequired {
                reason: reason as i32,
            }),
        );
    }

    pub(super) fn resync_begin(&mut self, sid: u64, _: v2::ResyncBegin) -> Checked<()> {
        let Some(session) = self.sessions.get_mut(&sid) else {
            return Ok(());
        };
        session.need_resync = false;
        session.syncing = true;
        if let Some(scope) = self.scopes.get_mut(&session.scope) {
            scope.index.clear(SourceId::Instance(session.instance));
        }
        Ok(())
    }

    pub(super) fn resync_chunk(&mut self, sid: u64, chunk: v2::ResyncChunk) -> Checked<()> {
        if chunk.entries.len() > self.cfg.limits.max_resync_chunk_entries as usize {
            return Err(Violation::new(
                "resync chunk exceeds max_resync_chunk_entries",
            ));
        }
        let Some(session) = self.sessions.get(&sid) else {
            return Ok(());
        };
        if session.need_resync {
            return Ok(());
        }
        if !session.syncing {
            return Err(Violation::new("snapshot chunk without Begin"));
        }
        let (scope, source) = (session.scope, SourceId::Instance(session.instance));
        for entry in &chunk.entries {
            self.apply_notice(scope, source, Notice::Saved(entry_from(entry)?));
        }
        self.bound_hints(sid);
        Ok(())
    }

    pub(super) fn resync_end(&mut self, sid: u64, _: v2::ResyncEnd) -> Checked<()> {
        let Some(session) = self.sessions.get_mut(&sid) else {
            return Ok(());
        };
        if session.need_resync {
            return Ok(());
        }
        if !session.syncing {
            return Err(Violation::new("snapshot End without Begin"));
        }
        session.syncing = false;
        self.send(sid, M::ResyncAck(v2::ResyncAck {}));
        Ok(())
    }
}
