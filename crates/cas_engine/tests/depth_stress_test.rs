//! Depth Stress Tests — detect stack overflows and budget exhaustion
//!
//! Generates expressions with systematically increasing nesting depth
//! across three families:
//!   1. Nested trig:        sin(sin(sin(...sin(x)...)))
//!   2. Power towers:       ((x+1)²+1)²+1)²
//!   3. Continued fractions: 1/(1 + 1/(1 + 1/(1 + ...)))
//!
//! Each CI test uses conservative depths that must NEVER overflow.
//! The `#[ignore]` sweep test finds the breaking point.
//!
//! RUN:
//!   cargo test --release -p cas_engine --test depth_stress_test                          # CI-safe
//!   cargo test --release -p cas_engine --test depth_stress_test -- --ignored --nocapture  # full sweep

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_engine::Simplifier;

use std::collections::HashMap;
use std::panic;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════════
// EXPRESSION BUILDERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Build sin(sin(sin(...sin(x)...))) with `depth` layers.
fn build_nested_trig(ctx: &mut Context, depth: usize) -> ExprId {
    let mut expr = ctx.var("x");
    for _ in 0..depth {
        expr = ctx.call_builtin(BuiltinFn::Sin, vec![expr]);
    }
    expr
}

/// Build ((((x+1)²+1)²+1)²+1)² with `depth` squarings.
///
/// Each level: prev = (prev + 1)²
fn build_power_tower(ctx: &mut Context, depth: usize) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let mut expr = ctx.var("x");
    for _ in 0..depth {
        let sum = ctx.add(Expr::Add(expr, one));
        expr = ctx.add(Expr::Pow(sum, two));
    }
    expr
}

/// Build 1/(1 + 1/(1 + 1/(1 + ... 1/(1+x) ...))) with `depth` layers.
fn build_continued_fraction(ctx: &mut Context, depth: usize) -> ExprId {
    let one = ctx.num(1);
    let mut expr = ctx.var("x");
    for _ in 0..depth {
        let denom = ctx.add(Expr::Add(one, expr));
        expr = ctx.add(Expr::Div(one, denom));
    }
    expr
}

/// Build ln(ln(ln(...ln(x)...))) with `depth` layers.
fn build_nested_ln(ctx: &mut Context, depth: usize) -> ExprId {
    let mut expr = ctx.var("x");
    for _ in 0..depth {
        expr = ctx.call_builtin(BuiltinFn::Ln, vec![expr]);
    }
    expr
}

/// Build x^(x^(x^...)) right-associative exponent tower with `depth` layers.
fn build_exp_tower(ctx: &mut Context, depth: usize) -> ExprId {
    let x = ctx.var("x");
    let mut expr = x;
    for _ in 1..depth {
        expr = ctx.add(Expr::Pow(x, expr));
    }
    expr
}

// ═══════════════════════════════════════════════════════════════════════════════
// NODE COUNTER
// ═══════════════════════════════════════════════════════════════════════════════

/// Re-export canonical node counter.
fn count_nodes(ctx: &Context, root: ExprId) -> usize {
    cas_ast::traversal::count_all_nodes(ctx, root)
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION HARNESS
// ═══════════════════════════════════════════════════════════════════════════════

struct DepthResult {
    depth: usize,
    input_nodes: usize,
    status: DepthStatus,
}

enum DepthStatus {
    Ok {
        steps: usize,
        output_nodes: usize,
        elapsed: Duration,
    },
    Overflow,
    Timeout,
}

/// Time limit per single expression (prevents infinite hangs).
const TIMEOUT_SECS: u64 = 10;

/// Safely simplify an expression, catching panics (stack overflow) and timeouts.
fn try_simplify_timed(
    ctx: Context,
    expr: ExprId,
) -> Result<(ExprId, usize, Context, Duration), &'static str> {
    let (tx, rx) = std::sync::mpsc::channel();

    let handle = std::thread::Builder::new()
        .name("depth-stress".into())
        .stack_size(16 * 1024 * 1024) // 16 MB stack
        .spawn(move || {
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                let mut simplifier = Simplifier::with_default_rules();
                simplifier.context = ctx;
                let start = Instant::now();
                let (result, steps) = simplifier.simplify(expr);
                let elapsed = start.elapsed();
                (result, steps.len(), simplifier.context, elapsed)
            }));
            let _ = tx.send(result);
        })
        .expect("Failed to spawn depth-stress thread");

    match rx.recv_timeout(Duration::from_secs(TIMEOUT_SECS)) {
        Ok(Ok(tuple)) => Ok(tuple),
        Ok(Err(_panic)) => Err("overflow"),
        Err(_timeout) => {
            // Thread is still running — we can't kill it, but we report timeout.
            // The thread will eventually finish or the process will exit.
            drop(handle);
            Err("timeout")
        }
    }
}

/// Run a single depth point and return the result.
fn run_depth_point(
    family_name: &str,
    depth: usize,
    builder: fn(&mut Context, usize) -> ExprId,
) -> DepthResult {
    let mut ctx = Context::new();
    let expr = builder(&mut ctx, depth);
    let input_nodes = count_nodes(&ctx, expr);

    match try_simplify_timed(ctx, expr) {
        Ok((result, steps, out_ctx, elapsed)) => {
            let output_nodes = count_nodes(&out_ctx, result);
            let _ = family_name; // used in verbose output only
            DepthResult {
                depth,
                input_nodes,
                status: DepthStatus::Ok {
                    steps,
                    output_nodes,
                    elapsed,
                },
            }
        }
        Err("timeout") => DepthResult {
            depth,
            input_nodes,
            status: DepthStatus::Timeout,
        },
        Err(_) => DepthResult {
            depth,
            input_nodes,
            status: DepthStatus::Overflow,
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEPTH SWEEP RUNNER
// ═══════════════════════════════════════════════════════════════════════════════

struct SweepConfig {
    family_name: &'static str,
    builder: fn(&mut Context, usize) -> ExprId,
    min_depth: usize,
    max_depth: usize,
    step: usize,
}

struct SweepSummary {
    family_name: &'static str,
    max_safe_depth: usize,
    first_failure_depth: Option<usize>,
    first_failure_kind: Option<&'static str>,
    results: Vec<DepthResult>,
}

fn run_sweep(config: &SweepConfig) -> SweepSummary {
    let mut results = Vec::new();
    let mut max_safe = 0;
    let mut first_failure = None;
    let mut first_failure_kind = None;

    let depths: Vec<usize> = (config.min_depth..=config.max_depth)
        .step_by(config.step)
        .collect();

    for &d in &depths {
        let r = run_depth_point(config.family_name, d, config.builder);
        match &r.status {
            DepthStatus::Ok { .. } => max_safe = d,
            DepthStatus::Overflow => {
                if first_failure.is_none() {
                    first_failure = Some(d);
                    first_failure_kind = Some("overflow");
                }
            }
            DepthStatus::Timeout => {
                if first_failure.is_none() {
                    first_failure = Some(d);
                    first_failure_kind = Some("timeout");
                }
            }
        }
        results.push(r);

        // Stop early after 3 consecutive failures
        let recent_fails = results
            .iter()
            .rev()
            .take(3)
            .filter(|r| !matches!(r.status, DepthStatus::Ok { .. }))
            .count();
        if recent_fails >= 3 {
            break;
        }
    }

    SweepSummary {
        family_name: config.family_name,
        max_safe_depth: max_safe,
        first_failure_depth: first_failure,
        first_failure_kind,
        results,
    }
}

fn print_sweep(summary: &SweepSummary) {
    eprintln!("\n═══════════════════════════════════════════════════════");
    eprintln!("         DEPTH STRESS: {}", summary.family_name);
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!(
        "  {:>5}  {:>7}  {:>7}  {:>9}  {:>9}  status",
        "depth", "in_nds", "out_nds", "steps", "time_ms"
    );
    eprintln!("  -----  -------  -------  ---------  ---------  ----------");

    for r in &summary.results {
        match &r.status {
            DepthStatus::Ok {
                steps,
                output_nodes,
                elapsed,
            } => {
                eprintln!(
                    "  {:>5}  {:>7}  {:>7}  {:>9}  {:>9.1}  ✅",
                    r.depth,
                    r.input_nodes,
                    output_nodes,
                    steps,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
            DepthStatus::Overflow => {
                eprintln!(
                    "  {:>5}  {:>7}  {:>7}  {:>9}  {:>9}  💥 overflow",
                    r.depth, r.input_nodes, "—", "—", "—"
                );
            }
            DepthStatus::Timeout => {
                eprintln!(
                    "  {:>5}  {:>7}  {:>7}  {:>9}  {:>9}  ⏱️  timeout",
                    r.depth, r.input_nodes, "—", "—", "—"
                );
            }
        }
    }

    eprintln!();
    eprintln!("  ✅ Max safe depth: {}", summary.max_safe_depth);
    if let Some(d) = summary.first_failure_depth {
        eprintln!(
            "  💥 First failure:  {} ({})",
            d,
            summary.first_failure_kind.unwrap_or("unknown")
        );
    } else {
        eprintln!("  🎉 No failures in tested range!");
    }
    eprintln!("═══════════════════════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// CI-SAFE TESTS (must ALWAYS pass)
// ═══════════════════════════════════════════════════════════════════════════════

/// CI guard: nested sin up to depth 50 must not overflow.
#[test]
fn test_depth_nested_trig_ci() {
    for depth in [5, 10, 20, 50] {
        let r = run_depth_point("Nested Trig", depth, build_nested_trig);
        assert!(
            matches!(r.status, DepthStatus::Ok { .. }),
            "sin^{depth}(x) at depth={depth} failed (stack overflow or timeout)"
        );
    }
}

/// CI guard: power towers up to depth 5 must not overflow.
/// Note: expansion is exponential; depth 6+ may timeout (~10s+).
#[test]
fn test_depth_power_tower_ci() {
    for depth in [2, 3, 4, 5] {
        let r = run_depth_point("Power Tower", depth, build_power_tower);
        assert!(
            matches!(r.status, DepthStatus::Ok { .. }),
            "power tower at depth={depth} failed (stack overflow or timeout)"
        );
    }
}

/// CI guard: continued fractions up to depth 50 must not overflow.
#[test]
fn test_depth_continued_fraction_ci() {
    for depth in [5, 10, 20, 50] {
        let r = run_depth_point("Continued Fraction", depth, build_continued_fraction);
        assert!(
            matches!(r.status, DepthStatus::Ok { .. }),
            "continued fraction at depth={depth} failed (stack overflow or timeout)"
        );
    }
}

/// CI guard: nested ln up to depth 50 must not overflow.
#[test]
fn test_depth_nested_ln_ci() {
    for depth in [5, 10, 20, 50] {
        let r = run_depth_point("Nested Ln", depth, build_nested_ln);
        assert!(
            matches!(r.status, DepthStatus::Ok { .. }),
            "ln^{depth}(x) at depth={depth} failed (stack overflow or timeout)"
        );
    }
}

/// CI guard: exponent tower up to depth 8 must not overflow.
#[test]
fn test_depth_exp_tower_ci() {
    for depth in [2, 4, 6, 8] {
        let r = run_depth_point("Exp Tower", depth, build_exp_tower);
        assert!(
            matches!(r.status, DepthStatus::Ok { .. }),
            "exp tower at depth={depth} failed (stack overflow or timeout)"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FULL SWEEP (manual — finds breaking points)
// ═══════════════════════════════════════════════════════════════════════════════

/// Full depth sweep across all families. Finds the breaking point for each.
///
/// Run with:
///   cargo test --release -p cas_engine --test depth_stress_test \
///       test_depth_full_sweep -- --ignored --nocapture
#[test]
#[ignore]
fn test_depth_full_sweep() {
    let families = vec![
        SweepConfig {
            family_name: "Nested Trig (sin)",
            builder: build_nested_trig,
            min_depth: 10,
            max_depth: 200,
            step: 10,
        },
        SweepConfig {
            family_name: "Power Towers ((x+1)²)",
            builder: build_power_tower,
            min_depth: 2,
            max_depth: 30,
            step: 2,
        },
        SweepConfig {
            family_name: "Continued Fractions",
            builder: build_continued_fraction,
            min_depth: 10,
            max_depth: 200,
            step: 10,
        },
        SweepConfig {
            family_name: "Nested Ln",
            builder: build_nested_ln,
            min_depth: 10,
            max_depth: 200,
            step: 10,
        },
        SweepConfig {
            family_name: "Exponent Tower (x^x^x^...)",
            builder: build_exp_tower,
            min_depth: 2,
            max_depth: 30,
            step: 2,
        },
    ];

    let mut summaries = Vec::new();
    for config in &families {
        eprintln!("🔍 Sweeping: {} ...", config.family_name);
        let summary = run_sweep(config);
        print_sweep(&summary);
        summaries.push(summary);
    }

    // Final summary table
    eprintln!("\n╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║              DEPTH STRESS TEST — SUMMARY                       ║");
    eprintln!("╠══════════════════════════════╤════════════╤═════════════════════╣");
    eprintln!("║ Family                       │ Max Safe   │ First Failure       ║");
    eprintln!("╠══════════════════════════════╪════════════╪═════════════════════╣");
    for s in &summaries {
        let fail_str = match (s.first_failure_depth, s.first_failure_kind) {
            (Some(d), Some(k)) => format!("{} ({})", d, k),
            _ => "none".to_string(),
        };
        eprintln!(
            "║ {:<28} │ {:>10} │ {:<19} ║",
            s.family_name, s.max_safe_depth, fail_str
        );
    }
    eprintln!("╚══════════════════════════════╧════════════╧═════════════════════╝\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// POWER TOWER DIAGNOSTIC (find the bottleneck)
// ═══════════════════════════════════════════════════════════════════════════════

/// Profile power tower simplification at each depth to identify which rules
/// consume the most time and applications.
///
/// Run with:
///   cargo test --release -p cas_engine --test depth_stress_test \
///       test_power_tower_diagnostic -- --ignored --nocapture
#[test]
#[ignore]
fn test_power_tower_diagnostic() {
    eprintln!("\n═══════════════════════════════════════════════════════");
    eprintln!("     POWER TOWER DIAGNOSTIC (rule profiling)");
    eprintln!("═══════════════════════════════════════════════════════\n");

    for depth in 1..=8 {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let mut expr = ctx.var("x");
        for _ in 0..depth {
            let sum = ctx.add(Expr::Add(expr, one));
            expr = ctx.add(Expr::Pow(sum, two));
        }
        let input_nodes = count_nodes(&ctx, expr);

        // Run with profiling, 30s timeout
        let (tx, rx) = std::sync::mpsc::channel();
        let _ = std::thread::Builder::new()
            .name("diag".into())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    let mut simplifier = Simplifier::with_default_rules();
                    simplifier.context = ctx;
                    simplifier.profiler = cas_engine::RuleProfiler::new(true);
                    let start = Instant::now();
                    let (result, steps) = simplifier.simplify(expr);
                    let elapsed = start.elapsed();
                    let out_nodes = count_nodes(&simplifier.context, result);

                    // Collect rule counts
                    let mut rule_counts: HashMap<String, usize> = HashMap::new();
                    for step in &steps {
                        *rule_counts.entry(step.rule_name.clone()).or_default() += 1;
                    }
                    (steps.len(), out_nodes, elapsed, rule_counts)
                }));
                let _ = tx.send(result);
            });

        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(Ok((steps, out_nodes, elapsed, rule_counts))) => {
                eprintln!(
                    "depth={depth}  in={input_nodes}  out={out_nodes}  steps={steps}  \
                     time={:.1}ms",
                    elapsed.as_secs_f64() * 1000.0
                );
                // Show top 5 rules
                let mut sorted: Vec<_> = rule_counts.iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(a.1));
                for (rule, count) in sorted.iter().take(5) {
                    eprintln!("   {:40} {:>4}", rule, count);
                }
                eprintln!();

                if elapsed.as_secs() > 25 {
                    eprintln!("  ⚠️  Stopping: approaching timeout\n");
                    break;
                }
            }
            Ok(Err(_)) => {
                eprintln!("depth={depth}  in={input_nodes}  💥 stack overflow\n");
                break;
            }
            Err(_) => {
                eprintln!("depth={depth}  in={input_nodes}  ⏱️  timeout (>30s)\n");
                break;
            }
        }
    }
}
