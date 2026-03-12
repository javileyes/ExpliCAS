//! Assumptions, warnings, and blocked-hint-facing API for session/repl consumers.

pub use crate::assumption_format::format_assumption_records_summary;
pub use crate::blocked_hint_format::{
    filter_blocked_hints_for_eval, format_eval_blocked_hints_lines,
    format_solve_assumption_and_blocked_sections,
};
pub use cas_solver_core::assumption_render::{
    format_blocked_hint_lines, format_diagnostics_requires_lines, format_domain_warning_lines,
    format_normalized_condition_lines, format_required_condition_lines,
};
pub use cas_solver_core::assumption_usage::{
    collect_assumed_conditions_from_steps, format_assumed_conditions_report_lines,
    group_assumed_conditions_by_rule,
};
pub use cas_solver_core::solve_assumption_types::SolveAssumptionSectionConfig;
