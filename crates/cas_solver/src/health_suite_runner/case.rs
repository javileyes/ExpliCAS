mod parse_error;
mod thresholds;

use crate::health_suite_types::{HealthCase, HealthCaseResult};
use crate::{Simplifier, SimplifyOptions, SimplifyPhase};

use super::cycle::detect_cycle;

/// Run a single health case and return the result
pub fn run_case(case: &HealthCase, simplifier: &mut Simplifier) -> HealthCaseResult {
    simplifier.profiler.enable_health();
    simplifier.profiler.clear_run();

    let expr_id = match cas_parser::parse(case.expr, &mut simplifier.context) {
        Ok(id) => id,
        Err(e) => return parse_error::parse_error_result(case, e.to_string()),
    };

    let opts = SimplifyOptions::default();
    let (_result, _steps, stats) = simplifier.simplify_with_stats(expr_id, opts);

    let total_rewrites = stats.total_rewrites;
    let core_rewrites = stats.core.rewrites_used;
    let transform_rewrites = stats.transform.rewrites_used;
    let rationalize_rewrites = stats.rationalize.rewrites_used;
    let post_rewrites = stats.post_cleanup.rewrites_used;
    let growth = simplifier.profiler.total_positive_growth();
    let shrink = simplifier.profiler.total_negative_growth().abs();
    let cycle_detected = detect_cycle(&stats);
    let top_rules = simplifier
        .profiler
        .top_applied_for_phase(SimplifyPhase::Transform, 3);

    let failure_reason = thresholds::failure_reason_for_case(
        case,
        total_rewrites,
        growth,
        transform_rewrites,
        &cycle_detected,
    );
    let warning = thresholds::warning_for_case(
        case,
        total_rewrites,
        transform_rewrites,
        &cycle_detected,
        failure_reason.is_none(),
    );

    HealthCaseResult {
        case: case.clone(),
        passed: failure_reason.is_none(),
        total_rewrites,
        core_rewrites,
        transform_rewrites,
        rationalize_rewrites,
        post_rewrites,
        growth,
        shrink,
        cycle_detected,
        top_rules,
        failure_reason,
        warning,
    }
}
