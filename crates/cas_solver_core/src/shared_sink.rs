//! Shared mutable sink wrapper for solver context threading.
//!
//! Centralizes interior mutability in `cas_solver_core` so engine-side
//! orchestration can avoid direct `Rc<RefCell<...>>` handling.

use std::cell::RefCell;
use std::rc::Rc;

/// Cloneable shared sink with interior mutability.
#[derive(Debug)]
pub struct SharedSink<T> {
    inner: Rc<RefCell<T>>,
}

impl<T> Clone for SharedSink<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<T> SharedSink<T> {
    /// Create a sink from an initial value.
    pub fn new(value: T) -> Self {
        Self {
            inner: Rc::new(RefCell::new(value)),
        }
    }

    /// Borrow the sink immutably for the closure duration.
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        let borrowed = self.inner.borrow();
        f(&borrowed)
    }

    /// Borrow the sink mutably for the closure duration.
    pub fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut borrowed = self.inner.borrow_mut();
        f(&mut borrowed)
    }
}

impl<T: Default> Default for SharedSink<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

#[cfg(test)]
mod tests {
    use super::SharedSink;

    #[test]
    fn clone_shares_underlying_state() {
        let sink = SharedSink::new(vec![1_u32]);
        let other = sink.clone();

        other.with_mut(|items| items.push(2));
        let snapshot = sink.with(|items| items.clone());

        assert_eq!(snapshot, vec![1, 2]);
    }

    #[test]
    fn default_uses_inner_default() {
        let sink: SharedSink<Vec<u8>> = SharedSink::default();
        let len = sink.with(|items| items.len());
        assert_eq!(len, 0);
    }
}
