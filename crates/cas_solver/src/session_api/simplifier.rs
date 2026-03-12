//! Simplifier-facing API re-exported for session clients.

pub use crate::repl_simplifier_runtime::{
    evaluate_unary_command_message_on_runtime, ReplSimplifierRuntimeContext,
};
pub use crate::simplifier_setup_build::build_simplifier_with_rule_config;
pub use crate::simplifier_setup_toggle::{
    apply_simplifier_toggle_config, set_simplifier_toggle_rule,
};
pub use crate::unary_command_eval::{
    evaluate_unary_command_lines, evaluate_unary_function_command_lines,
};
pub use cas_solver_core::simplifier_config::{SimplifierRuleConfig, SimplifierToggleConfig};
