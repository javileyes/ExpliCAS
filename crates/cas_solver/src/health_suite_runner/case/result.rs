use super::stats::HealthRunStats;
use cas_solver_core::health_suite_models::{HealthCase, HealthCaseResult};

pub(super) fn build_case_result(
    case: &HealthCase,
    run_stats: HealthRunStats,
    failure_reason: Option<String>,
    warning: Option<String>,
) -> HealthCaseResult {
    HealthCaseResult {
        case: case.clone(),
        passed: failure_reason.is_none(),
        total_rewrites: run_stats.total_rewrites,
        core_rewrites: run_stats.core_rewrites,
        transform_rewrites: run_stats.transform_rewrites,
        rationalize_rewrites: run_stats.rationalize_rewrites,
        post_rewrites: run_stats.post_rewrites,
        growth: run_stats.growth,
        shrink: run_stats.shrink,
        cycle_detected: run_stats.cycle_detected,
        top_rules: run_stats.top_rules,
        failure_reason,
        warning,
    }
}
