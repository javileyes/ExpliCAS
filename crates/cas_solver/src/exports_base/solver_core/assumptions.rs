pub use cas_solver_core::assume_scope::AssumeScope;
pub use cas_solver_core::assumption_model::AssumptionRecord;
pub use cas_solver_core::assumption_model::{
    assumption_condition_text, assumption_key_dedupe_fingerprint, blocked_hint_suggestion,
    collect_assumption_records, collect_assumption_records_from_iter, collect_blocked_hint_items,
    format_assumption_records_conditions, format_assumption_records_section_lines,
    format_blocked_hint_condition, format_blocked_simplifications_section_lines,
    group_blocked_hint_conditions_by_rule, AssumptionCollector, AssumptionEvent, AssumptionKey,
    AssumptionKind,
};
pub use cas_solver_core::assumption_reporting::AssumptionReporting;
pub use cas_solver_core::blocked_hint::BlockedHint;
pub use cas_solver_core::blocked_hint_store::{
    clear_blocked_hints, register_blocked_hint, take_blocked_hints,
};
