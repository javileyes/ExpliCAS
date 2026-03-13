//! Public session-owned REPL API.

#[path = "repl_semantics_runtime/autoexpand.rs"]
mod autoexpand;
#[path = "repl_runtime_core/build.rs"]
mod build;
#[path = "repl_runtime_core/config_command.rs"]
mod config_command;
#[path = "repl_semantics_runtime/context.rs"]
mod context;
#[path = "repl_runtime_core/reset.rs"]
mod reset;
#[path = "repl_semantics_runtime/semantics.rs"]
mod semantics;
#[path = "repl_semantics_runtime/sync.rs"]
mod sync;

pub use crate::config::CasConfig;
pub use crate::repl_core::ReplCore;
pub use autoexpand::evaluate_autoexpand_command_on_repl;
pub use build::build_repl_core_with_config;
pub use config_command::evaluate_and_apply_config_command_on_repl;
pub use context::evaluate_context_command_on_repl;
pub use reset::{reset_repl_core_full_with_config, reset_repl_core_with_config};
pub use semantics::evaluate_semantics_command_on_repl;
