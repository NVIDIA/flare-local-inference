pub(crate) struct OwnerLease<T>(pub(crate) Option<T>);

impl<T> OwnerLease<T> {
    pub(crate) fn take_after_drain(&mut self) -> Option<T> {
        self.0.take()
    }
}

impl<T> Drop for OwnerLease<T> {
    fn drop(&mut self) {
        // An unconfirmed drain must never release resources still used by native work.
        if let Some(owner) = self.0.take() {
            std::mem::forget(owner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    struct Resource(Arc<AtomicBool>);

    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn releases_owner_only_after_confirmed_drain() {
        let released = Arc::new(AtomicBool::new(false));
        let mut lease = OwnerLease(Some(Resource(released.clone())));
        assert!(!released.load(Ordering::SeqCst));
        drop(lease.take_after_drain());
        assert!(released.load(Ordering::SeqCst));
        drop(lease.take_after_drain());
    }

    #[test]
    fn unconfirmed_drain_keeps_owner_alive() {
        let released = Arc::new(AtomicBool::new(false));
        drop(OwnerLease(Some(Resource(released.clone()))));
        assert!(!released.load(Ordering::SeqCst));
    }
}
