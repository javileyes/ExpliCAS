//! Formatting, parsing and presentation helpers re-exported for session clients.

pub use crate::format_solve_budget_command_message;
pub use crate::parse_expr_pair;
pub use crate::set_simplifier_toggle_rule;
pub use crate::{
    format_assignment_command_output_message, format_assignment_error_message,
    format_assignment_success_message, format_let_assignment_parse_error_message,
    let_assignment_usage_message, parse_let_assignment_input,
};
pub use crate::{
    format_binding_overview_lines, format_clear_bindings_result_lines, vars_empty_message,
};
pub use crate::{
    format_delete_history_error_message, format_delete_history_result_message,
    format_history_overview_lines, history_empty_message,
};
pub use crate::{format_equivalence_result_lines, format_expr_pair_parse_error_message};
pub use crate::{
    format_health_failed_tests_warning_line, format_health_invalid_category_message,
    format_health_missing_category_arg_message, format_health_report_lines,
    format_health_status_running_message, format_health_usage_message, health_usage_message,
    resolve_health_category_filter,
};
pub use crate::{
    format_history_entry_inspection_lines, format_inspect_history_entry_error_message,
};
pub use crate::{
    format_semantics_axis_lines, format_semantics_overview_lines,
    format_semantics_unknown_subcommand_message, semantics_help_message,
};
pub use crate::{
    format_show_history_command_lines, format_show_history_command_lines_with_context,
};
pub use crate::{inspect_history_entry, inspect_history_entry_input};
pub use crate::{parse_history_entry_id, parse_history_ids};
pub use crate::{render_error_with_caret, render_parse_error};
pub use cas_solver_core::analysis_command_types::ParseExprPairError;
pub use cas_solver_core::assignment_command_types::{
    AssignmentCommandOutput, AssignmentError, LetAssignmentParseError, ParsedLetAssignment,
};
pub use cas_solver_core::semantics_view_types::{
    semantics_view_state_from_options, SemanticsViewState,
};
pub use cas_solver_core::session_runtime::SolveBudgetCommandResult;
pub use cas_solver_core::simplifier_config::{SimplifierRuleConfig, SimplifierToggleConfig};
