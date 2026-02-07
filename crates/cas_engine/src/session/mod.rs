//! Session storage for expressions and equations with auto-incrementing IDs.
//!
//! Provides a "notebook-style" storage where each input gets a unique `#id`
//! that can be referenced in subsequent commands.

mod resolve;
mod store;

#[cfg(test)]
mod tests;

pub use resolve::*;
pub use store::*;
