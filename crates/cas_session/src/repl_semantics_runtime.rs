//! Runtime adapters that synchronize semantics-related REPL changes with `CasConfig`.

mod autoexpand;
mod context;
mod semantics;
mod sync;

pub use autoexpand::evaluate_autoexpand_command_on_repl;
pub use context::evaluate_context_command_on_repl;
pub use semantics::evaluate_semantics_command_on_repl;
