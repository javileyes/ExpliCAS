#![allow(clippy::type_complexity)]
// Stress test for debugging rule orchestration bottlenecks
//
// PURPOSE: This test uses deliberately complex expressions to identify:
// 1. Rules that trigger too many re-simplification loops
// 2. Patterns that cause exponential rule application
// 3. Stack overflow triggers in the recursive simplifier
//
// RUN WITH:
//   STRESS_PROFILE=STRESS RUST_MIN_STACK=16777216 cargo test --package cas_engine --test stress_test -- --nocapture
//
// PROFILES (set via STRESS_PROFILE env var):
//   SAFE    - CI/CD, never overflows
//   NORMAL  - Development, balanced
//   STRESS  - Detects problems
//   EXTREME - Deep debugging

use cas_ast::{Context, DisplayExpr, Expr};
use cas_engine::Simplifier;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::{Config, TestRunner};
use std::collections::HashMap;
use std::panic;

mod strategies;
use strategies::{arb_recursive_expr_with_profile, get_active_profile, to_context, TestProfile};

// ═══════════════════════════════════════════════════════════════════════════════
// TEST STATISTICS TRACKING
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Default)]
struct TestStats {
    total_tests: usize,
    passed: usize,
    #[allow(dead_code)]
    failed: usize,
    overflows: usize,
    total_steps: usize,
    rule_hits: HashMap<String, usize>,
    max_steps_expr: Option<(String, usize)>,
    overflow_exprs: Vec<String>,
}

impl TestStats {
    fn record_success(
        &mut self,
        expr_str: &str,
        steps: usize,
        rule_counts: HashMap<String, usize>,
    ) {
        self.total_tests += 1;
        self.passed += 1;
        self.total_steps += steps;

        // Update max steps expression
        if self
            .max_steps_expr
            .as_ref()
            .is_none_or(|(_, max)| steps > *max)
        {
            self.max_steps_expr = Some((expr_str.to_string(), steps));
        }

        // Aggregate rule hits
        for (rule, count) in rule_counts {
            *self.rule_hits.entry(rule).or_default() += count;
        }
    }

    fn record_overflow(&mut self, expr_str: &str) {
        self.total_tests += 1;
        self.overflows += 1;
        self.overflow_exprs.push(expr_str.to_string());
    }

    fn print_summary(&self, profile: &TestProfile) {
        eprintln!("\n═══════════════════════════════════════════════════════════════════");
        eprintln!("                    STRESS TEST SUMMARY");
        eprintln!("═══════════════════════════════════════════════════════════════════");
        eprintln!(
            "Profile: {} (depth={}, size={}, items={})",
            profile.name, profile.depth, profile.size, profile.items
        );
        eprintln!();

        // Test results
        let pass_rate = if self.total_tests > 0 {
            (self.passed as f64 / self.total_tests as f64) * 100.0
        } else {
            0.0
        };

        eprintln!("📊 TEST RESULTS:");
        eprintln!(
            "   Passed:     {:>4} / {} ({:.1}%)",
            self.passed, self.total_tests, pass_rate
        );
        if self.overflows > 0 {
            eprintln!("   ⚠️  Overflows: {:>4}", self.overflows);
        }
        if self.failed > 0 {
            eprintln!("   ❌ Failed:    {:>4}", self.failed);
        }
        eprintln!();

        // Rule statistics
        if self.passed > 0 {
            let avg_steps = self.total_steps as f64 / self.passed as f64;
            eprintln!("📈 RULE STATISTICS:");
            eprintln!("   Total simplifications:     {}", self.passed);
            eprintln!("   Total rule applications:   {}", self.total_steps);
            eprintln!("   Average rules/expression:  {:.1}", avg_steps);
            eprintln!();

            // Top rules
            let mut sorted_rules: Vec<_> = self.rule_hits.iter().collect();
            sorted_rules.sort_by(|a, b| b.1.cmp(a.1));

            if !sorted_rules.is_empty() {
                eprintln!("   Top 10 Rules:");
                let total_rules: usize = sorted_rules.iter().map(|(_, c)| *c).sum();
                for (rule, count) in sorted_rules.iter().take(10) {
                    let pct = (**count as f64 / total_rules as f64) * 100.0;
                    eprintln!("      {:40} {:>4} ({:>5.1}%)", rule, count, pct);
                }
            }
            eprintln!();
        }

        // Most expensive expression
        if let Some((expr, steps)) = &self.max_steps_expr {
            eprintln!("🔥 MOST EXPENSIVE EXPRESSION:");
            eprintln!("   Steps: {}", steps);
            let display = if expr.len() > 60 {
                format!("{}...", &expr[..60])
            } else {
                expr.clone()
            };
            eprintln!("   Expr:  {}", display);
            eprintln!();
        }

        // Overflow expressions (IMPORTANT for debugging)
        if !self.overflow_exprs.is_empty() {
            eprintln!(
                "⚠️  STACK OVERFLOW EXPRESSIONS ({}):",
                self.overflow_exprs.len()
            );
            eprintln!("   These expressions caused stack overflow and need investigation:");
            for (i, expr) in self.overflow_exprs.iter().take(5).enumerate() {
                let display = if expr.len() > 70 {
                    format!("{}...", &expr[..70])
                } else {
                    expr.clone()
                };
                eprintln!("   {}. {}", i + 1, display);
            }
            if self.overflow_exprs.len() > 5 {
                eprintln!("   ... and {} more", self.overflow_exprs.len() - 5);
            }
            eprintln!();
            eprintln!("   💡 TIP: Copy these expressions to test_stress_single() for debugging.");
        }

        eprintln!("═══════════════════════════════════════════════════════════════════\n");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRESS TESTS - With overflow detection
// ═══════════════════════════════════════════════════════════════════════════════

/// Safely simplify an expression, catching panics/overflows
fn try_simplify(
    ctx: Context,
    expr: cas_ast::ExprId,
) -> Result<
    (
        cas_ast::ExprId,
        Vec<cas_engine::Step>,
        HashMap<String, usize>,
    ),
    String,
> {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx;
        simplifier.profiler = cas_engine::profiler::RuleProfiler::new(true);

        let (result, steps) = simplifier.simplify(expr);

        // Extract rule counts from profiler
        let mut rule_counts = HashMap::new();
        for step in &steps {
            *rule_counts.entry(step.rule_name.clone()).or_default() += 1;
        }

        (result, steps, rule_counts, simplifier.context)
    }));

    match result {
        Ok((result, steps, counts, _ctx)) => Ok((result, steps, counts)),
        Err(_) => Err("Stack overflow or panic".to_string()),
    }
}

/// Main stress test with comprehensive statistics
#[test]
fn test_stress_with_profiling() {
    let profile = get_active_profile();
    let num_cases = 20u32;
    let config = Config {
        cases: num_cases,
        ..Config::default()
    };
    let mut runner = TestRunner::new(config);

    let strategy = arb_recursive_expr_with_profile(profile);
    let mut stats = TestStats::default();

    for _ in 0..num_cases {
        if let Ok(tree) = strategy.new_tree(&mut runner) {
            let re = tree.current();
            let (ctx, expr) = to_context(re.clone());
            let display = DisplayExpr {
                context: &ctx,
                id: expr,
            };
            let expr_str = display.to_string();

            match try_simplify(ctx, expr) {
                Ok((_result, steps, rule_counts)) => {
                    stats.record_success(&expr_str, steps.len(), rule_counts);
                }
                Err(_) => {
                    stats.record_overflow(&expr_str);
                }
            }
        }
    }

    // Print summary at the end
    stats.print_summary(&profile);

    // Assert at least some tests passed
    assert!(stats.passed > 0, "No tests passed!");
}

// ═══════════════════════════════════════════════════════════════════════════════
// SINGLE EXPRESSION DEBUGGING
// ═══════════════════════════════════════════════════════════════════════════════

/// Test a single known problematic expression with full debugging
#[test]
fn test_stress_single() {
    let mut ctx = Context::new();

    // CUSTOMIZE: Replace with the expression you want to debug
    // Example: sin(x)^2 + cos(x)^2 nested in complex structure
    let x = ctx.var("x");
    let two = ctx.num(2);
    let three = ctx.num(3);
    let one = ctx.num(1);
    let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![x]);
    let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![x]);
    let sin_sq = ctx.add(Expr::Pow(sin_x, two));
    let cos_sq = ctx.add(Expr::Pow(cos_x, two));
    let sum = ctx.add(Expr::Add(sin_sq, cos_sq));

    // Add more complexity
    let nested = ctx.add(Expr::Pow(sum, three));
    let expr = ctx.add(Expr::Add(nested, one));

    let display = DisplayExpr {
        context: &ctx,
        id: expr,
    };
    eprintln!("\n═══ DEBUGGING EXPRESSION ═══");
    eprintln!("Input: {}", display);

    let mut simplifier = Simplifier::with_default_rules();
    simplifier.context = ctx;
    simplifier.profiler = cas_engine::profiler::RuleProfiler::new(true);

    let (result, steps) = simplifier.simplify(expr);

    let result_display = DisplayExpr {
        context: &simplifier.context,
        id: result,
    };

    eprintln!("\n--- STEP TRACE ({} steps) ---", steps.len());
    for (i, step) in steps.iter().enumerate() {
        let before = DisplayExpr {
            context: &simplifier.context,
            id: step.before,
        };
        let after = DisplayExpr {
            context: &simplifier.context,
            id: step.after,
        };
        eprintln!("{:3}. [{}] {} → {}", i + 1, step.rule_name, before, after);
    }

    eprintln!("\n--- PROFILER REPORT ---");
    eprintln!("{}", simplifier.profiler.report());

    eprintln!("\n--- RESULT ---");
    eprintln!("Output: {}", result_display);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BATCH STRESS TEST - For finding overflow-causing expressions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run many random expressions and collect overflow statistics
#[test]
#[ignore] // Run with: cargo test test_batch_overflow_finder -- --ignored --nocapture
fn test_batch_overflow_finder() {
    let profile = get_active_profile();
    let num_cases = 100u32;
    let config = Config {
        cases: num_cases,
        ..Config::default()
    };
    let mut runner = TestRunner::new(config);

    let strategy = arb_recursive_expr_with_profile(profile);

    let mut stats = TestStats::default();

    eprintln!(
        "\n🔍 Running batch overflow finder with profile {}...",
        profile.name
    );
    eprintln!("   Generating {} expressions...\n", num_cases);

    for i in 0..num_cases {
        if let Ok(tree) = strategy.new_tree(&mut runner) {
            let re = tree.current();
            let (ctx, expr) = to_context(re.clone());
            let display = DisplayExpr {
                context: &ctx,
                id: expr,
            };
            let expr_str = display.to_string();

            match try_simplify(ctx, expr) {
                Ok((_result, steps, rule_counts)) => {
                    stats.record_success(&expr_str, steps.len(), rule_counts);
                    if (i + 1) % 20 == 0 {
                        eprint!(".");
                    }
                }
                Err(_) => {
                    stats.record_overflow(&expr_str);
                    eprint!("💥");
                }
            }
        }
    }
    eprintln!();

    stats.print_summary(&profile);
}

/// Count total recursive depth in an expression
#[allow(dead_code)]
fn count_depth(ctx: &Context, expr: cas_ast::ExprId) -> usize {
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => 1,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            1 + count_depth(ctx, *l).max(count_depth(ctx, *r))
        }
        Expr::Neg(e) => 1 + count_depth(ctx, *e),
        Expr::Hold(e) => 1 + count_depth(ctx, *e),
        Expr::Function(_, args) => 1 + args.iter().map(|a| count_depth(ctx, *a)).max().unwrap_or(0),
        Expr::Matrix { data, .. } => {
            1 + data.iter().map(|e| count_depth(ctx, *e)).max().unwrap_or(0)
        }
    }
}
