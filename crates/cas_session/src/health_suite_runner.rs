use crate::health_suite_catalog::default_suite;
use crate::health_suite_types::{Category, HealthCase, HealthCaseResult};
use cas_solver::{PipelineStats, Simplifier, SimplifyOptions, SimplifyPhase};

/// Run a single health case and return the result
pub fn run_case(case: &HealthCase, simplifier: &mut Simplifier) -> HealthCaseResult {
    // Reset profiler for this run
    simplifier.profiler.enable_health();
    simplifier.profiler.clear_run();

    // Parse expression directly into the simplifier's context
    let expr_id = match cas_parser::parse(case.expr, &mut simplifier.context) {
        Ok(id) => id,
        Err(e) => {
            return HealthCaseResult {
                case: case.clone(),
                passed: false,
                total_rewrites: 0,
                core_rewrites: 0,
                transform_rewrites: 0,
                rationalize_rewrites: 0,
                post_rewrites: 0,
                growth: 0,
                shrink: 0,
                cycle_detected: None,
                top_rules: vec![],
                failure_reason: Some(format!("Parse error: {}", e)),
                warning: None,
            };
        }
    };

    // Simplify with stats
    let opts = SimplifyOptions::default();
    let (_result, _steps, stats) = simplifier.simplify_with_stats(expr_id, opts);

    // Collect metrics
    let total_rewrites = stats.total_rewrites;
    let core_rewrites = stats.core.rewrites_used;
    let transform_rewrites = stats.transform.rewrites_used;
    let rationalize_rewrites = stats.rationalize.rewrites_used;
    let post_rewrites = stats.post_cleanup.rewrites_used;
    let growth = simplifier.profiler.total_positive_growth();
    let shrink = simplifier.profiler.total_negative_growth().abs();

    // Check for cycles
    let cycle_detected = detect_cycle(&stats);

    // Get top rules from Transform phase (usually the hot spot)
    let top_rules = simplifier
        .profiler
        .top_applied_for_phase(SimplifyPhase::Transform, 3);

    // Determine pass/fail
    let mut failure_reason = None;

    if total_rewrites > case.limits.max_total_rewrites {
        failure_reason = Some(format!(
            "rewrites={} > max={}",
            total_rewrites, case.limits.max_total_rewrites
        ));
    } else if growth > case.limits.max_growth {
        failure_reason = Some(format!(
            "growth={} > max={}",
            growth, case.limits.max_growth
        ));
    } else if transform_rewrites > case.limits.max_transform_rewrites {
        failure_reason = Some(format!(
            "transform_rewrites={} > max={}",
            transform_rewrites, case.limits.max_transform_rewrites
        ));
    } else if case.limits.forbid_cycles {
        if let Some((phase, period)) = cycle_detected.as_ref() {
            failure_reason = Some(format!("cycle detected: {:?} period={}", phase, period));
        }
    }

    // Generate warnings for passed cases with concerning metrics
    let mut warning = None;
    if failure_reason.is_none() {
        // Warning if cycle detected but not failing (forbid_cycles=false)
        if let Some((phase, period)) = cycle_detected.as_ref() {
            if !case.limits.forbid_cycles {
                warning = Some(format!("cycle (allowed): {:?} period={}", phase, period));
            }
        }
        // Warning if near limit (>80% of max)
        let rewrite_pct = (total_rewrites * 100) / case.limits.max_total_rewrites.max(1);
        let transform_pct = (transform_rewrites * 100) / case.limits.max_transform_rewrites.max(1);
        if rewrite_pct >= 80 && warning.is_none() {
            warning = Some(format!("near limit: rewrites={}%", rewrite_pct));
        } else if transform_pct >= 80 && warning.is_none() {
            warning = Some(format!("near limit: transform={}%", transform_pct));
        }
    }

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

/// Check for cycles in any phase
fn detect_cycle(stats: &PipelineStats) -> Option<(SimplifyPhase, usize)> {
    if let Some(ref c) = stats.core.cycle {
        return Some((SimplifyPhase::Core, c.period));
    }
    if let Some(ref c) = stats.transform.cycle {
        return Some((SimplifyPhase::Transform, c.period));
    }
    if let Some(ref c) = stats.rationalize.cycle {
        return Some((SimplifyPhase::Rationalize, c.period));
    }
    if let Some(ref c) = stats.post_cleanup.cycle {
        return Some((SimplifyPhase::PostCleanup, c.period));
    }
    None
}

/// Run entire suite and return all results
#[allow(dead_code)]
pub fn run_suite(simplifier: &mut Simplifier) -> Vec<HealthCaseResult> {
    let suite = default_suite();
    suite
        .iter()
        .map(|case| run_case(case, simplifier))
        .collect()
}

/// Run suite filtered by category
pub fn run_suite_filtered(
    simplifier: &mut Simplifier,
    filter: Option<Category>,
) -> Vec<HealthCaseResult> {
    let suite = default_suite();
    let filtered: Vec<_> = match filter {
        Some(cat) => suite.into_iter().filter(|c| c.category == cat).collect(),
        None => suite,
    };
    filtered
        .iter()
        .map(|case| run_case(case, simplifier))
        .collect()
}
