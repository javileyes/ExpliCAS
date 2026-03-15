//! Substitute command entrypoints exposed for CLI/frontends.

pub use crate::substitute::SubstituteSimplifyEvalOutput;
pub use crate::substitute::{
    detect_substitute_strategy, substitute_auto, substitute_auto_with_strategy,
    substitute_power_aware, substitute_with_steps, SubstituteOptions, SubstituteStrategy,
};
pub use crate::substitute_command_eval::{
    evaluate_substitute_command_lines, evaluate_substitute_invocation_lines,
    evaluate_substitute_invocation_message, evaluate_substitute_invocation_user_message,
};
pub use crate::substitute_command_format::{
    format_substitute_eval_lines, format_substitute_parse_error_message,
    substitute_render_mode_from_display_mode,
};
pub use crate::substitute_subcommand_eval::evaluate_substitute_subcommand;
pub use crate::substitute_subcommand_text::parse_substitute_wire_text_lines;
pub use crate::substitute_subcommand_wire::evaluate_substitute_subcommand_wire;
pub use cas_api_models::{
    SubstituteCommandMode, SubstituteParseError, SubstituteRenderMode, SubstituteSubcommandOutput,
};
