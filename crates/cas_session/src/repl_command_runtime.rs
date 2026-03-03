//! Runtime adapters for REPL command evaluation on `ReplCore`.

pub use crate::repl_command_analysis_runtime::{
    evaluate_expand_log_invocation_message_on_repl_core,
    evaluate_explain_invocation_message_on_repl_core,
    evaluate_linear_system_command_message_on_repl_core,
    evaluate_telescope_invocation_message_on_repl_core,
    evaluate_visualize_invocation_output_on_repl_core,
    evaluate_weierstrass_invocation_message_on_repl_core,
};
pub use crate::repl_command_core_runtime::{
    evaluate_equiv_invocation_message_on_repl_core,
    evaluate_full_simplify_command_lines_on_repl_core,
    evaluate_health_command_message_on_repl_core, evaluate_rationalize_command_lines_on_repl_core,
    evaluate_solve_command_message_on_repl_core,
    evaluate_substitute_invocation_user_message_on_repl_core, update_health_report_on_repl_core,
};
pub use crate::repl_command_eval_runtime::{
    evaluate_eval_command_render_plan_on_repl_core,
    evaluate_expand_command_render_plan_on_repl_core, profile_cache_len_on_repl_core,
};
pub use crate::repl_command_session_runtime::{
    evaluate_assignment_command_message_on_repl_core, evaluate_clear_command_lines_on_repl_core,
    evaluate_delete_history_command_message_on_repl_core,
    evaluate_history_command_message_on_repl_core,
    evaluate_let_assignment_command_message_on_repl_core,
    evaluate_profile_cache_command_lines_on_repl_core, evaluate_show_command_lines_on_repl_core,
    evaluate_solve_budget_command_message_on_repl_core, evaluate_vars_command_message_on_repl_core,
};
pub use crate::repl_command_unary_runtime::{
    evaluate_det_command_message_on_repl_core, evaluate_trace_command_message_on_repl_core,
    evaluate_transpose_command_message_on_repl_core,
};
