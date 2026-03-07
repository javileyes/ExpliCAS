//! Helpers to project engine-backed `EvalOutput` into solver-owned view types.
//!
//! These adapters keep conversion logic in one place so frontend crates don't
//! need to access raw `EvalOutput` transport fields directly.

mod accessors;
#[cfg(test)]
mod tests;
mod view;

pub use self::accessors::*;
pub use self::view::*;
