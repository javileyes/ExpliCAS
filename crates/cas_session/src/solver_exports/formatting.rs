//! Formatting, parsing and presentation helpers re-exported for session clients.

pub use cas_solver::{
    format_assignment_command_output_message, format_assignment_error_message,
    format_assignment_success_message, format_let_assignment_parse_error_message,
    let_assignment_usage_message, parse_let_assignment_input, AssignmentCommandOutput,
    AssignmentError, LetAssignmentParseError, ParsedLetAssignment,
};
pub use cas_solver::{
    format_binding_overview_lines, format_clear_bindings_result_lines, vars_empty_message,
};
pub use cas_solver::{
    format_delete_history_error_message, format_delete_history_result_message,
    format_history_overview_lines, history_empty_message,
};
pub use cas_solver::{format_equivalence_result_lines, format_expr_pair_parse_error_message};
pub use cas_solver::{
    format_health_failed_tests_warning_line, format_health_invalid_category_message,
    format_health_missing_category_arg_message, format_health_report_lines,
    format_health_status_running_message, format_health_usage_message, health_usage_message,
    resolve_health_category_filter,
};
pub use cas_solver::{
    format_history_entry_inspection_lines, format_inspect_history_entry_error_message,
};
pub use cas_solver::{
    format_semantics_axis_lines, format_semantics_overview_lines,
    format_semantics_unknown_subcommand_message, semantics_help_message,
    semantics_view_state_from_options, SemanticsViewState,
};
pub use cas_solver::{
    format_show_history_command_lines, format_show_history_command_lines_with_context,
};
pub use cas_solver::{format_solve_budget_command_message, SolveBudgetCommandResult};
pub use cas_solver::{inspect_history_entry, inspect_history_entry_input};
pub use cas_solver::{parse_expr_pair, ParseExprPairError};
pub use cas_solver::{parse_history_entry_id, parse_history_ids};
pub use cas_solver::{render_error_with_caret, render_parse_error};
pub use cas_solver::{set_simplifier_toggle_rule, SimplifierRuleConfig, SimplifierToggleConfig};
