//! Shared session snapshot I/O helpers for CLI entrypoints.

mod load;
mod run;
mod save;

pub use load::load_or_new_session;
pub(crate) use run::{run_read_only_with_domain_session, run_with_domain_session};
pub use save::save_session;
