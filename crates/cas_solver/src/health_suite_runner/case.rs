mod parse_error;
mod result;
mod stats;
mod thresholds;

use crate::health_suite_models::{HealthCase, HealthCaseResult};
use crate::{Simplifier, SimplifyOptions};

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

    let run_stats = stats::collect_run_stats(simplifier, &stats);

    let failure_reason = thresholds::failure_reason_for_case(
        case,
        run_stats.total_rewrites,
        run_stats.growth,
        run_stats.transform_rewrites,
        &run_stats.cycle_detected,
    );
    let warning = thresholds::warning_for_case(
        case,
        run_stats.total_rewrites,
        run_stats.transform_rewrites,
        &run_stats.cycle_detected,
        failure_reason.is_none(),
    );

    result::build_case_result(case, run_stats, failure_reason, warning)
}
