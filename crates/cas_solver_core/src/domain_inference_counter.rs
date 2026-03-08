//! Thread-local counter for implicit-domain inference calls.
//!
//! Kept in solver-core so both engine and higher layers can reuse the same
//! instrumentation primitive without owning the storage in runtime crates.

use std::cell::Cell;

thread_local! {
    static INFER_DOMAIN_CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Reset the domain inference call counter.
#[inline]
pub fn reset() {
    INFER_DOMAIN_CALLS.with(|c| c.set(0));
}

/// Read the current domain inference call counter.
#[inline]
pub fn get() -> usize {
    INFER_DOMAIN_CALLS.with(|c| c.get())
}

/// Increment the domain inference call counter.
#[inline]
pub fn inc() {
    INFER_DOMAIN_CALLS.with(|c| c.set(c.get() + 1));
}
