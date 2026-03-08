use crate::semantics_set_parse::evaluate_semantics_set_args;
use crate::semantics_set_types::{semantics_set_state_from_options, SemanticsSetState};

/// Apply semantic state to both simplifier options and runtime eval options.
pub fn apply_semantics_set_state_to_options(
    next: SemanticsSetState,
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
) {
    simplify_options.shared.semantics.domain_mode = next.domain_mode;
    simplify_options.shared.semantics.value_domain = next.value_domain;
    simplify_options.shared.semantics.branch = next.branch;
    simplify_options.shared.semantics.inv_trig = next.inv_trig;
    simplify_options.shared.semantics.assume_scope = next.assume_scope;
    simplify_options.shared.assumption_reporting = next.assumption_reporting;

    eval_options.shared.semantics.domain_mode = next.domain_mode;
    eval_options.shared.semantics.value_domain = next.value_domain;
    eval_options.shared.semantics.branch = next.branch;
    eval_options.shared.semantics.inv_trig = next.inv_trig;
    eval_options.shared.semantics.assume_scope = next.assume_scope;
    eval_options.shared.assumption_reporting = next.assumption_reporting;

    eval_options.const_fold = next.const_fold;
    eval_options.hints_enabled = next.hints_enabled;
    eval_options.check_solutions = next.check_solutions;
    eval_options.requires_display = next.requires_display;
}

/// Parse `semantics set` args and apply the resulting state to runtime options.
pub fn apply_semantics_set_args_to_options(
    args: &[&str],
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
) -> Result<SemanticsSetState, String> {
    let current = semantics_set_state_from_options(simplify_options, eval_options);
    let next = evaluate_semantics_set_args(args, current)?;
    apply_semantics_set_state_to_options(next, simplify_options, eval_options);
    Ok(next)
}

/// Parse `semantics set` args, apply them, and return updated overview lines.
pub fn evaluate_semantics_set_args_to_overview_lines(
    args: &[&str],
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
) -> Result<Vec<String>, String> {
    apply_semantics_set_args_to_options(args, simplify_options, eval_options)?;
    let view_state = crate::semantics_view_state_from_options(simplify_options, eval_options);
    Ok(crate::format_semantics_overview_lines(&view_state))
}
