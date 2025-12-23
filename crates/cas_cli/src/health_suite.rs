//! Health Suite: Predefined test cases for engine health validation
//!
//! This module provides a shared suite of test cases used by:
//! - `health status` CLI command for interactive diagnostics
//! - Can be reused by CI tests
//!
//! Each case has expression, expected limits, and optional flags.

use cas_engine::phase::SimplifyPhase;
use cas_engine::{PipelineStats, Simplifier, SimplifyOptions};
use std::str::FromStr;

/// Category of health test case
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Transform,
    Expansion,
    Fractions,
    Rationalization,
    Mixed,
    Baseline,
    Roots,
    Powers,
    Stress,
    Policy,
}

impl Category {
    /// All available categories
    pub fn all() -> &'static [Category] {
        &[
            Category::Transform,
            Category::Expansion,
            Category::Fractions,
            Category::Rationalization,
            Category::Mixed,
            Category::Baseline,
            Category::Roots,
            Category::Powers,
            Category::Stress,
            Category::Policy,
        ]
    }

    /// Short name for display
    pub fn as_str(&self) -> &'static str {
        match self {
            Category::Transform => "transform",
            Category::Expansion => "expansion",
            Category::Fractions => "fractions",
            Category::Rationalization => "rationalization",
            Category::Mixed => "mixed",
            Category::Baseline => "baseline",
            Category::Roots => "roots",
            Category::Powers => "powers",
            Category::Stress => "stress",
            Category::Policy => "policy",
        }
    }
}

impl std::fmt::Display for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for Category {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "transform" | "trans" | "t" => Ok(Category::Transform),
            "expansion" | "expand" | "exp" | "e" => Ok(Category::Expansion),
            "fractions" | "frac" | "f" => Ok(Category::Fractions),
            "rationalization" | "rational" | "rat" | "r" => Ok(Category::Rationalization),
            "mixed" | "mix" | "m" => Ok(Category::Mixed),
            "baseline" | "base" | "b" => Ok(Category::Baseline),
            "roots" | "root" => Ok(Category::Roots),
            "powers" | "pow" | "p" => Ok(Category::Powers),
            "stress" | "s" => Ok(Category::Stress),
            "policy" | "pol" => Ok(Category::Policy),
            "all" | "*" => Err("Use None for all categories".to_string()),
            _ => Err(format!("Unknown category: '{}'. Valid: transform, expansion, fractions, rationalization, mixed, baseline, roots, powers, stress", s)),
        }
    }
}

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
    /// Category
    pub category: Category,
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
    /// Per-phase rewrites
    pub core_rewrites: usize,
    pub transform_rewrites: usize,
    pub rationalize_rewrites: usize,
    pub post_rewrites: usize,
    /// Growth metrics
    pub growth: i64, // total_positive_growth
    pub shrink: i64, // total_negative_growth (absolute value)
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
            category: Category::Transform,
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
            category: Category::Transform,
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
            category: Category::Transform,
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
            category: Category::Expansion,
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
            category: Category::Expansion,
            expr: "(x+1)^5",
            limits: HealthLimits {
                max_total_rewrites: 50,
                max_growth: 100,
                max_transform_rewrites: 20,
                forbid_cycles: true,
            },
        },
        // Explicit expand() cases - should show t>0 and growth
        HealthCase {
            name: "expand_binomial",
            category: Category::Expansion,
            expr: "expand((x+1)^2)",
            limits: HealthLimits {
                max_total_rewrites: 20,
                max_growth: 30,
                max_transform_rewrites: 5,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "expand_conjugate",
            category: Category::Expansion,
            expr: "expand((x-1)*(x+1))",
            limits: HealthLimits {
                max_total_rewrites: 15,
                max_growth: 20,
                max_transform_rewrites: 3,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "expand_product_chain",
            category: Category::Expansion,
            expr: "expand((x-1)*(x+1)*(x^2+1))",
            limits: HealthLimits {
                max_total_rewrites: 40,
                max_growth: 60,
                max_transform_rewrites: 10,
                forbid_cycles: true,
            },
        },
        // ============ Fractions ============
        HealthCase {
            name: "fraction_add",
            category: Category::Fractions,
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
            category: Category::Fractions,
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
            category: Category::Rationalization,
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
            category: Category::Rationalization,
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
            category: Category::Rationalization,
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
            category: Category::Mixed,
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
            category: Category::Baseline,
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
            category: Category::Baseline,
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
            category: Category::Roots,
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
            category: Category::Powers,
            expr: "x^2 * x^3",
            limits: HealthLimits {
                max_total_rewrites: 15,
                max_growth: 20,
                max_transform_rewrites: 5,
                forbid_cycles: true,
            },
        },
        // ============ STRESS SUITE (heavy integration tests) ============
        // These cases are designed to exercise growth, multi-phase interactions,
        // and higher workloads. Expected to have higher rewrites and growth.
        HealthCase {
            name: "expand_product_chain",
            category: Category::Stress,
            // Use expand() to force Transform phase activity (bypasses binomial*binomial guard)
            expr: "expand((x+1)*(x+2)*(x+3))",
            limits: HealthLimits {
                max_total_rewrites: 200,
                max_growth: 350,
                max_transform_rewrites: 120,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "binomial_large",
            category: Category::Stress,
            expr: "(x+1)^8",
            limits: HealthLimits {
                max_total_rewrites: 220,
                max_growth: 450,
                max_transform_rewrites: 120,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "distribute_sum",
            category: Category::Stress,
            expr: "3*(x+y+z+w)",
            limits: HealthLimits {
                max_total_rewrites: 160,
                max_growth: 250,
                max_transform_rewrites: 100,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "nested_distribution",
            category: Category::Stress,
            expr: "2*(x + 3*(y+4))",
            limits: HealthLimits {
                max_total_rewrites: 200,
                max_growth: 300,
                max_transform_rewrites: 120,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "rationalize_level15_mixed",
            category: Category::Stress,
            expr: "(x+1)/(2*(1+sqrt(2))) + 2*(y+3)",
            limits: HealthLimits {
                max_total_rewrites: 180,
                max_growth: 220,
                max_transform_rewrites: 100,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "rationalize_binomial_negative",
            category: Category::Stress,
            expr: "x/(2*(3-2*sqrt(5)))",
            limits: HealthLimits {
                max_total_rewrites: 160,
                max_growth: 220,
                max_transform_rewrites: 80,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "nested_root_simplify_hard",
            category: Category::Stress,
            expr: "sqrt(5 + 2*sqrt(6))",
            limits: HealthLimits {
                max_total_rewrites: 160,
                max_growth: 200,
                max_transform_rewrites: 60,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "fraction_polynomial_combo",
            category: Category::Stress,
            expr: "x/2 + x/3 + (x+1)^6",
            limits: HealthLimits {
                max_total_rewrites: 260,
                max_growth: 500,
                max_transform_rewrites: 140,
                forbid_cycles: true,
            },
        },
        // ============ Policy A+: simplify vs expand behavior ============
        HealthCase {
            name: "policy_simplify_binomial_no_expand",
            category: Category::Policy,
            expr: "(x+1)*(x+2)",
            limits: HealthLimits {
                max_total_rewrites: 10, // Very low - should NOT expand
                max_growth: 10,
                max_transform_rewrites: 2, // Minimal transform activity
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "policy_simplify_conjugate_expands",
            category: Category::Policy,
            expr: "(x-1)*(x+1)",
            limits: HealthLimits {
                max_total_rewrites: 30, // DoS rule should fire
                max_growth: 20,
                max_transform_rewrites: 15,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "policy_expand_binomial_product",
            category: Category::Policy,
            expr: "expand((x+1)*(x+2))",
            limits: HealthLimits {
                max_total_rewrites: 50, // Expansion should work
                max_growth: 40,
                max_transform_rewrites: 30,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "policy_expand_binomial_power",
            category: Category::Policy,
            expr: "expand((x+1)^6)",
            limits: HealthLimits {
                max_total_rewrites: 200, // Binomial expansion is expensive
                max_growth: 300,
                max_transform_rewrites: 120,
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
    } else if case.limits.forbid_cycles && cycle_detected.is_some() {
        let (phase, period) = cycle_detected.as_ref().unwrap();
        failure_reason = Some(format!("cycle detected: {:?} period={}", phase, period));
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

/// Format suite results as a human-readable report
#[allow(dead_code)]
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

    // Add legend
    report.push_str("\nLegend: rw=total rewrites, c=Core, t=Transform, r=Rationalize, p=Post\n");

    report
}

/// Count passed/failed for summary
pub fn count_results(results: &[HealthCaseResult]) -> (usize, usize) {
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;
    (passed, failed)
}

/// List all available test cases
pub fn list_cases() -> String {
    let suite = default_suite();
    let mut output = format!("Available health cases ({}):\n", suite.len());
    output.push_str("─────────────────────────────────────────\n");

    // Group by category
    for cat in Category::all() {
        let cases_in_cat: Vec<_> = suite.iter().filter(|c| c.category == *cat).collect();
        if !cases_in_cat.is_empty() {
            for case in cases_in_cat {
                output.push_str(&format!("[{:14}] {}\n", cat.as_str(), case.name));
            }
        }
    }
    output
}

/// Get all available category names for autocomplete
pub fn category_names() -> Vec<&'static str> {
    Category::all().iter().map(|c| c.as_str()).collect()
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

/// Format report with category header
pub fn format_report_filtered(results: &[HealthCaseResult], category: Option<Category>) -> String {
    let mut report = String::new();

    let header = match category {
        Some(cat) => format!("Health Status Suite [category={}]\n", cat),
        None => "Health Status Suite\n".to_string(),
    };
    report.push_str(&header);
    report.push_str("═══════════════════════════════════════════════════════════════\n");

    let mut passed = 0;
    let mut failed = 0;

    for r in results {
        let name_padded = format!("{:25}", r.case.name);
        // Format per-phase rewrites compactly
        let phases = format!(
            "(c={} t={} r={} p={})",
            r.core_rewrites, r.transform_rewrites, r.rationalize_rewrites, r.post_rewrites
        );
        // Format growth/shrink
        let growth_str = if r.shrink > 0 {
            format!("+{:3}/-{}", r.growth, r.shrink)
        } else {
            format!("+{:3}", r.growth)
        };

        if r.passed {
            passed += 1;
            let status = if r.warning.is_some() { "⚠" } else { "✔" };
            report.push_str(&format!(
                "{} {}  rw={:3} {} {}\n",
                status, name_padded, r.total_rewrites, phases, growth_str
            ));
            if let Some(ref warn) = r.warning {
                report.push_str(&format!("    └─ {}\n", warn));
            }
        } else {
            failed += 1;
            report.push_str(&format!(
                "✘ {}  FAILED: {}\n",
                name_padded,
                r.failure_reason.as_ref().unwrap_or(&"unknown".to_string())
            ));
            report.push_str(&format!(
                "    rw={} {} {}\n",
                r.total_rewrites, phases, growth_str
            ));
            if let Some((phase, period)) = &r.cycle_detected {
                report.push_str(&format!("    Cycle: {:?} period={}\n", phase, period));
            }
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

    // Add legend
    report.push_str("\nLegend: rw=total rewrites, c=Core, t=Transform, r=Rationalize, p=Post\n");

    report
}
