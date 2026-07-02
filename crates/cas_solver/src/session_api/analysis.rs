//! Analysis command APIs re-exported for session clients.

pub use crate::derive_command::{evaluate_derive_command_lines_with_resolver, DeriveEvalError};
pub use crate::full_simplify_display::FullSimplifyDisplayMode;
pub use crate::repl_simplifier_runtime::{
    evaluate_equiv_invocation_message_on_runtime as evaluate_equiv_invocation_message_on_repl_core,
    evaluate_explain_invocation_message_on_runtime as evaluate_explain_invocation_message_on_repl_core,
    evaluate_visualize_invocation_output_on_runtime as evaluate_visualize_invocation_output_on_repl_core,
};
pub use crate::repl_solve_runtime::{
    evaluate_derive_command_lines_on_runtime as evaluate_derive_command_lines_on_repl_core,
    ReplSolveRuntimeContext,
};
pub use cas_api_models::{
    ExplainCommandEvalError, ExplainGcdEvalOutput, ParseExprPairError, VisualizeCommandOutput,
    VisualizeEvalError,
};
