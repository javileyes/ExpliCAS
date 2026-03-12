//! Public session-owned API for REPL/session runtime orchestration.

pub use crate::config::CasConfig;
pub use crate::repl_core::ReplCore;
pub use crate::repl_runtime_core::{
    build_repl_core_with_config, evaluate_and_apply_config_command_on_repl,
    reset_repl_core_full_with_config, reset_repl_core_with_config,
};
pub use crate::repl_semantics_runtime::{
    evaluate_autoexpand_command_on_repl, evaluate_context_command_on_repl,
    evaluate_semantics_command_on_repl,
};
