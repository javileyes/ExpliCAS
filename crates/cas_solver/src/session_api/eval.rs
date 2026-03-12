//! Eval-facing session APIs re-exported for session clients.

pub use crate::command_api::eval::{
    build_eval_command_render_plan, evaluate_eval_command_output,
    evaluate_eval_text_simplify_with_session, EvalCommandError, EvalCommandOutput,
    EvalCommandRenderPlan,
};
pub use crate::eval_command_runtime::evaluate_eval_with_session;
pub use crate::output_clean::clean_result_output_line;
pub use crate::repl_eval_runtime::{
    evaluate_eval_command_render_plan_on_runtime as evaluate_eval_command_render_plan_on_repl_core,
    evaluate_expand_command_render_plan_on_runtime as evaluate_expand_command_render_plan_on_repl_core,
    profile_cache_len_on_runtime as profile_cache_len_on_repl_core, ReplEvalRuntimeContext,
};
