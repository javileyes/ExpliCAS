//! Analysis command APIs re-exported for session clients.

pub use crate::analysis_command_explain::{
    evaluate_explain_command_lines, evaluate_explain_command_message,
    evaluate_explain_invocation_message,
};
pub use crate::analysis_command_format_errors::{
    format_explain_command_error_message, format_visualize_command_error_message,
};
pub use crate::analysis_command_format_explain::format_explain_gcd_eval_lines;
pub use crate::analysis_command_parse::{
    extract_equiv_command_tail, extract_explain_command_tail, extract_visualize_command_tail,
};
pub use crate::analysis_command_visualize::{
    evaluate_visualize_command_dot, evaluate_visualize_command_output,
    evaluate_visualize_invocation_output,
};
pub use crate::analysis_input_parse::parse_expr_pair;
pub use crate::equiv_command::{
    evaluate_equiv_command_lines, evaluate_equiv_command_message, evaluate_equiv_invocation_message,
};
pub use crate::equiv_format::{
    format_equivalence_result_lines, format_expr_pair_parse_error_message,
};
pub use crate::repl_simplifier_runtime::{
    evaluate_equiv_invocation_message_on_runtime as evaluate_equiv_invocation_message_on_repl_core,
    evaluate_explain_invocation_message_on_runtime as evaluate_explain_invocation_message_on_repl_core,
    evaluate_visualize_invocation_output_on_runtime as evaluate_visualize_invocation_output_on_repl_core,
};
pub use cas_solver_core::analysis_command_types::{
    ExplainCommandEvalError, ExplainGcdEvalOutput, ParseExprPairError, VisualizeCommandOutput,
    VisualizeEvalError,
};
