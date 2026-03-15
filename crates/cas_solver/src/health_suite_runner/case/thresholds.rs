use crate::health_suite_models::HealthCase;
use crate::SimplifyPhase;

pub(super) fn failure_reason_for_case(
    case: &HealthCase,
    total_rewrites: usize,
    growth: i64,
    transform_rewrites: usize,
    cycle_detected: &Option<(SimplifyPhase, usize)>,
) -> Option<String> {
    if total_rewrites > case.limits.max_total_rewrites {
        Some(format!(
            "rewrites={} > max={}",
            total_rewrites, case.limits.max_total_rewrites
        ))
    } else if growth > case.limits.max_growth {
        Some(format!(
            "growth={} > max={}",
            growth, case.limits.max_growth
        ))
    } else if transform_rewrites > case.limits.max_transform_rewrites {
        Some(format!(
            "transform_rewrites={} > max={}",
            transform_rewrites, case.limits.max_transform_rewrites
        ))
    } else if case.limits.forbid_cycles {
        cycle_detected
            .as_ref()
            .map(|(phase, period)| format!("cycle detected: {:?} period={}", phase, period))
    } else {
        None
    }
}

pub(super) fn warning_for_case(
    case: &HealthCase,
    total_rewrites: usize,
    transform_rewrites: usize,
    cycle_detected: &Option<(SimplifyPhase, usize)>,
    passed: bool,
) -> Option<String> {
    if !passed {
        return None;
    }

    if let Some((phase, period)) = cycle_detected.as_ref() {
        if !case.limits.forbid_cycles {
            return Some(format!("cycle (allowed): {:?} period={}", phase, period));
        }
    }

    let rewrite_pct = (total_rewrites * 100) / case.limits.max_total_rewrites.max(1);
    let transform_pct = (transform_rewrites * 100) / case.limits.max_transform_rewrites.max(1);

    if rewrite_pct >= 80 {
        Some(format!("near limit: rewrites={}%", rewrite_pct))
    } else if transform_pct >= 80 {
        Some(format!("near limit: transform={}%", transform_pct))
    } else {
        None
    }
}
