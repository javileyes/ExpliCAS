//! Health Suite: Predefined test cases for engine health validation
//!
//! This module provides a shared suite of test cases used by:
//! - `health status` CLI command for interactive diagnostics
//! - Can be reused by CI tests
//!
//! Each case has expression, expected limits, and optional flags.

use cas_engine::phase::SimplifyPhase;
use cas_engine::{PipelineStats, Simplifier, SimplifyOptions};

/// Health limits for a test case
#[derive(Debug, Clone)]
pub struct HealthLimits {
    /// Maximum total rewrites across all phases
    pub max_total_rewrites: usize,
    /// Maximum positive node growth
    pub max_growth: i64,
    /// Maximum rewrites in Transform phase
    pub max_transform_rewrites: usize,
    /// Whether cycles should cause failure (default: true)
    pub forbid_cycles: bool,
}

impl Default for HealthLimits {
    fn default() -> Self {
        Self {
            max_total_rewrites: 100,
            max_growth: 200,
            max_transform_rewrites: 50,
            forbid_cycles: true,
        }
    }
}

/// A single health test case
#[derive(Debug, Clone)]
pub struct HealthCase {
    /// Human-readable name
    pub name: &'static str,
    /// Category (expansion, fraction, rationalization, etc.)
    pub category: &'static str,
    /// Input expression
    pub expr: &'static str,
    /// Health limits
    pub limits: HealthLimits,
}

/// Result of running a health case
#[derive(Debug)]
pub struct HealthCaseResult {
    pub case: HealthCase,
    pub passed: bool,
    pub total_rewrites: usize,
    pub growth: i64,
    pub transform_rewrites: usize,
    pub cycle_detected: Option<(SimplifyPhase, usize)>, // (phase, period)
    pub top_rules: Vec<(String, usize)>,
    pub failure_reason: Option<String>,
    /// Warning: cycle detected but not failing (forbid_cycles=false) or near limit
    pub warning: Option<String>,
}

/// The default health suite
pub fn default_suite() -> Vec<HealthCase> {
    vec![
        // ============ Transform-heavy (distribution/expansion) ============
        HealthCase {
            name: "distribute_basic",
            category: "transform",
            expr: "2*(x+3)",
            limits: HealthLimits {
                max_total_rewrites: 20,
                max_growth: 30,
                max_transform_rewrites: 10,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "distribute_nested",
            category: "transform",
            expr: "3*(x+(y+2))",
            limits: HealthLimits {
                max_total_rewrites: 30,
                max_growth: 50,
                max_transform_rewrites: 15,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "expand_product",
            category: "transform",
            expr: "(x+1)*(x+2)",
            limits: HealthLimits {
                max_total_rewrites: 40,
                max_growth: 60,
                max_transform_rewrites: 20,
                forbid_cycles: true,
            },
        },
        // ============ Expansion (binomial) ============
        HealthCase {
            name: "binomial_small",
            category: "expansion",
            expr: "(x+1)^3",
            limits: HealthLimits {
                max_total_rewrites: 30,
                max_growth: 50,
                max_transform_rewrites: 10,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "binomial_medium",
            category: "expansion",
            expr: "(x+1)^5",
            limits: HealthLimits {
                max_total_rewrites: 50,
                max_growth: 100,
                max_transform_rewrites: 20,
                forbid_cycles: true,
            },
        },
        // ============ Fractions ============
        HealthCase {
            name: "fraction_add",
            category: "fractions",
            expr: "x/2 + x/3",
            limits: HealthLimits {
                max_total_rewrites: 40,
                max_growth: 60,
                max_transform_rewrites: 15,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "fraction_simplify",
            category: "fractions",
            expr: "(x^2-1)/(x-1)",
            limits: HealthLimits {
                max_total_rewrites: 50,
                max_growth: 80,
                max_transform_rewrites: 20,
                forbid_cycles: true,
            },
        },
        // ============ Rationalization ============
        HealthCase {
            name: "rationalize_simple",
            category: "rationalization",
            expr: "1/sqrt(2)",
            limits: HealthLimits {
                max_total_rewrites: 30,
                max_growth: 40,
                max_transform_rewrites: 10,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "rationalize_binomial",
            category: "rationalization",
            expr: "1/(1+sqrt(2))",
            limits: HealthLimits {
                max_total_rewrites: 50,
                max_growth: 80,
                max_transform_rewrites: 20,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "rationalize_complex",
            category: "rationalization",
            expr: "1/(3-2*sqrt(5))",
            limits: HealthLimits {
                max_total_rewrites: 80,
                max_growth: 150,
                max_transform_rewrites: 30,
                forbid_cycles: true,
            },
        },
        // ============ Mixed operations ============
        HealthCase {
            name: "mixed_expression",
            category: "mixed",
            expr: "x/(1+sqrt(2)) + 2*(y+3)",
            limits: HealthLimits {
                max_total_rewrites: 80,
                max_growth: 120,
                max_transform_rewrites: 40,
                forbid_cycles: true,
            },
        },
        // ============ Baseline (no-op) ============
        HealthCase {
            name: "simple_noop",
            category: "baseline",
            expr: "x + y",
            limits: HealthLimits {
                max_total_rewrites: 15,
                max_growth: 20,
                max_transform_rewrites: 5,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "constant_fold",
            category: "baseline",
            expr: "2 + 3 * 4",
            limits: HealthLimits {
                max_total_rewrites: 10,
                max_growth: 10,
                max_transform_rewrites: 5,
                forbid_cycles: true,
            },
        },
        // ============ Roots ============
        HealthCase {
            name: "nested_root",
            category: "roots",
            expr: "sqrt(8)",
            limits: HealthLimits {
                max_total_rewrites: 20,
                max_growth: 30,
                max_transform_rewrites: 10,
                forbid_cycles: true,
            },
        },
        // ============ Powers ============
        HealthCase {
            name: "power_simplify",
            category: "powers",
            expr: "x^2 * x^3",
            limits: HealthLimits {
                max_total_rewrites: 15,
                max_growth: 20,
                max_transform_rewrites: 5,
                forbid_cycles: true,
            },
        },
    ]
}

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
                growth: 0,
                transform_rewrites: 0,
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
    let growth = simplifier.profiler.total_positive_growth();
    let transform_rewrites = stats.transform.rewrites_used;

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
    } else if case.limits.forbid_cycles && cycle_detected.is_some() {
        let (phase, period) = cycle_detected.as_ref().unwrap();
        failure_reason = Some(format!("cycle detected: {:?} period={}", phase, period));
    }

    // Generate warnings for passed cases with concerning metrics
    let mut warning = None;
    if failure_reason.is_none() {
        // Warning if cycle detected but not failing (forbid_cycles=false)
        if !case.limits.forbid_cycles && cycle_detected.is_some() {
            let (phase, period) = cycle_detected.as_ref().unwrap();
            warning = Some(format!("cycle (allowed): {:?} period={}", phase, period));
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
        growth,
        transform_rewrites,
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
pub fn run_suite(simplifier: &mut Simplifier) -> Vec<HealthCaseResult> {
    let suite = default_suite();
    suite
        .iter()
        .map(|case| run_case(case, simplifier))
        .collect()
}

/// Format suite results as a human-readable report
pub fn format_report(results: &[HealthCaseResult]) -> String {
    let mut report = String::new();
    report.push_str("Health Status Suite\n");
    report.push_str("═══════════════════════════════════════════════════════════════\n");

    let mut passed = 0;
    let mut failed = 0;

    for r in results {
        let name_padded = format!("{:25}", r.case.name);

        if r.passed {
            passed += 1;
            // Show warning indicator if present
            let status = if r.warning.is_some() { "⚠" } else { "✔" };
            report.push_str(&format!(
                "{} {}  rewrites={:3} growth={:+4} transform={:2}",
                status, name_padded, r.total_rewrites, r.growth, r.transform_rewrites
            ));
            // Append warning message
            if let Some(ref warn) = r.warning {
                report.push_str(&format!(" [{}]", warn));
            }
            report.push('\n');
        } else {
            failed += 1;
            report.push_str(&format!(
                "✘ {}  FAILED: {}\n",
                name_padded,
                r.failure_reason.as_ref().unwrap_or(&"unknown".to_string())
            ));
            // Show cycle info if present
            if let Some((phase, period)) = &r.cycle_detected {
                report.push_str(&format!("    Cycle: {:?} period={}\n", phase, period));
            }
            // Show top rules
            if !r.top_rules.is_empty() {
                let rules: Vec<_> = r
                    .top_rules
                    .iter()
                    .map(|(n, c)| format!("{}={}", n, c))
                    .collect();
                report.push_str(&format!("    Top Transform: {}\n", rules.join(", ")));
            }
        }
    }

    report.push_str("═══════════════════════════════════════════════════════════════\n");
    let total = passed + failed;
    if failed == 0 {
        report.push_str(&format!("PASSED: {}/{} cases ✓\n", passed, total));
    } else {
        report.push_str(&format!("FAILED: {}/{} cases\n", failed, total));
    }

    report
}

/// Count passed/failed for summary
pub fn count_results(results: &[HealthCaseResult]) -> (usize, usize) {
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;
    (passed, failed)
}
