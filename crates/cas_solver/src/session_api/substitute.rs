//! Substitute command APIs re-exported for session clients.

pub use crate::repl_simplifier_runtime::evaluate_substitute_invocation_user_message_on_runtime as evaluate_substitute_invocation_user_message_on_repl_core;
pub use crate::substitute_command_eval::{
    evaluate_substitute_command_lines, evaluate_substitute_invocation_lines,
    evaluate_substitute_invocation_message, evaluate_substitute_invocation_user_message,
};
pub use crate::substitute_command_format::{
    format_substitute_eval_lines, format_substitute_parse_error_message,
    substitute_render_mode_from_display_mode,
};
pub use cas_api_models::SubstituteRenderMode;
