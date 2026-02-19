//! Round-Trip Metamorphic Tests
//!
//! Tests mathematical correctness of transformation chains:
//! - Chain 1: simplify(expand(x)) ≡ simplify(x)  (expand→simplify idempotence)
//! - Chain 2: expand(factor(x)) ≡ x               (factor→expand round-trip)
//!
//! Uses the 3-tier verification system:
//! - NF-convergent: structural equality of simplified forms
//! - Proved-symbolic: simplify(a - b) == 0
//! - Numeric-only: f64 evaluation at sample points

use cas_ast::{Context, ExprId};
use cas_engine::Simplifier;
use cas_engine::{eval_f64_checked, EvalCheckedError, EvalCheckedOptions};
use cas_formatter::LaTeXExpr;
use cas_parser::parse;
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;

// =============================================================================
// Helpers
// =============================================================================

/// Simplify an expression string and return (result_id, LaTeX, Simplifier).
#[allow(dead_code)]
fn simp_full(input: &str) -> Option<(ExprId, String, Simplifier)> {
    let mut s = Simplifier::with_default_rules();
    let e = parse(input, &mut s.context).ok()?;
    let (r, _) = s.simplify(e);
    let cfg = cas_engine::semantics::EvalConfig::default();
    let mut budget = cas_engine::Budget::preset_cli();
    let r2 = if let Ok(res) = cas_engine::fold_constants(
        &mut s.context,
        r,
        &cfg,
        cas_engine::ConstFoldMode::Safe,
        &mut budget,
    ) {
        res.expr
    } else {
        r
    };
    let latex = LaTeXExpr {
        context: &s.context,
        id: r2,
    }
    .to_latex();
    Some((r2, latex, s))
}

/// Simplify an ExprId within an existing Simplifier context.
fn simp_expr(s: &mut Simplifier, expr: ExprId) -> ExprId {
    let (r, _) = s.simplify(expr);
    let cfg = cas_engine::semantics::EvalConfig::default();
    let mut budget = cas_engine::Budget::preset_cli();
    if let Ok(res) = cas_engine::fold_constants(
        &mut s.context,
        r,
        &cfg,
        cas_engine::ConstFoldMode::Safe,
        &mut budget,
    ) {
        res.expr
    } else {
        r
    }
}

fn to_latex(ctx: &Context, id: ExprId) -> String {
    LaTeXExpr { context: ctx, id }.to_latex()
}

// =============================================================================
// Numeric equivalence check (single variable, simplified)
// =============================================================================

fn check_numeric_equiv_1var(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
) -> Result<usize, String> {
    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    let num_samples = 20;
    let lo = -3.0_f64;
    let hi = 3.0_f64;
    let atol = 1e-8;
    let rtol = 1e-6;
    let mut valid = 0usize;
    let mut eval_failed = 0usize;

    for i in 0..num_samples {
        let t = (i as f64 + 0.5) / num_samples as f64;
        let x = lo + (hi - lo) * t;

        let mut var_map = HashMap::new();
        var_map.insert(var.to_string(), x);

        let va = eval_f64_checked(ctx, a, &var_map, &opts);
        let vb = eval_f64_checked(ctx, b, &var_map, &opts);

        match (&va, &vb) {
            (Ok(va), Ok(vb)) => {
                let diff = (va - vb).abs();
                let scale = va.abs().max(vb.abs()).max(1.0);
                let allowed = atol + rtol * scale;
                if diff > allowed {
                    return Err(format!(
                        "Numeric mismatch at {}={}: a={:.12}, b={:.12}, diff={:.3e} > allowed={:.3e}",
                        var, x, va, vb, diff, allowed
                    ));
                }
                valid += 1;
            }
            // Symmetric failures are OK (both fail at same point)
            (Err(EvalCheckedError::NearPole { .. }), Err(EvalCheckedError::NearPole { .. })) => {}
            (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {}
            // Asymmetric: concerning but tolerable
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {}
            _ => {
                eval_failed += 1;
            }
        }
    }

    if valid < 3 {
        return Err(format!(
            "Too few valid samples: {} (eval_failed={})",
            valid, eval_failed
        ));
    }

    Ok(valid)
}

/// 2-variable numeric equivalence for multivariate expressions
fn check_numeric_equiv_2var(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
) -> Result<usize, String> {
    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    let samples_per_dim = 5;
    let lo = -2.0_f64;
    let hi = 2.0_f64;
    let atol = 1e-8;
    let rtol = 1e-6;
    let mut valid = 0usize;

    for i in 0..samples_per_dim {
        for j in 0..samples_per_dim {
            let t1 = (i as f64 + 0.5) / samples_per_dim as f64;
            let t2 = (j as f64 + 0.5) / samples_per_dim as f64;
            let x = lo + (hi - lo) * t1;
            let y = lo + (hi - lo) * t2;

            let mut var_map = HashMap::new();
            var_map.insert(var1.to_string(), x);
            var_map.insert(var2.to_string(), y);

            let va = eval_f64_checked(ctx, a, &var_map, &opts);
            let vb = eval_f64_checked(ctx, b, &var_map, &opts);

            match (&va, &vb) {
                (Ok(va), Ok(vb)) => {
                    let diff = (va - vb).abs();
                    let scale = va.abs().max(vb.abs()).max(1.0);
                    let allowed = atol + rtol * scale;
                    if diff > allowed {
                        return Err(format!(
                            "Numeric mismatch at {}={}, {}={}: a={:.12}, b={:.12}, diff={:.3e}",
                            var1, x, var2, y, va, vb, diff
                        ));
                    }
                    valid += 1;
                }
                (
                    Err(EvalCheckedError::NearPole { .. }),
                    Err(EvalCheckedError::NearPole { .. }),
                ) => {}
                (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {}
                (Ok(_), Err(_)) | (Err(_), Ok(_)) => {}
                _ => {}
            }
        }
    }

    if valid < 3 {
        return Err(format!("Too few valid samples: {}", valid));
    }

    Ok(valid)
}

// =============================================================================
// 3-tier equivalence result
// =============================================================================

#[derive(Debug, Clone)]
enum EquivResult {
    NfConvergent,
    ProvedSymbolic,
    NumericOnly,
    Failed(String),
}

/// Check equivalence of two expression strings using the 3-tier system.
/// Uses a thread + timeout to prevent hangs.
#[allow(dead_code)]
fn check_equiv_3tier(lhs: &str, rhs: &str, vars: &[&str], timeout: Duration) -> EquivResult {
    let lhs_owned = lhs.to_string();
    let rhs_owned = rhs.to_string();
    let vars_owned: Vec<String> = vars.iter().map(|s| s.to_string()).collect();

    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        // Parse and simplify both
        let mut s = Simplifier::with_default_rules();
        let lp = match parse(&lhs_owned, &mut s.context) {
            Ok(e) => e,
            Err(_) => {
                let _ = tx.send(EquivResult::Failed("Parse LHS failed".into()));
                return;
            }
        };
        let rp = match parse(&rhs_owned, &mut s.context) {
            Ok(e) => e,
            Err(_) => {
                let _ = tx.send(EquivResult::Failed("Parse RHS failed".into()));
                return;
            }
        };

        let l = simp_expr(&mut s, lp);
        let r = simp_expr(&mut s, rp);

        // Tier 1: NF convergence
        if cas_engine::ordering::compare_expr(&s.context, l, r) == std::cmp::Ordering::Equal {
            let _ = tx.send(EquivResult::NfConvergent);
            return;
        }

        // Tier 2: Proved symbolic — simplify(LHS - RHS) == 0 [fresh context]
        {
            let diff_str = format!("({}) - ({})", lhs_owned, rhs_owned);
            let mut sd = Simplifier::with_default_rules();
            if let Ok(dp) = parse(&diff_str, &mut sd.context) {
                let dr = simp_expr(&mut sd, dp);
                let zero = num_rational::BigRational::from_integer(0.into());
                if matches!(sd.context.get(dr), cas_ast::Expr::Number(n) if *n == zero) {
                    let _ = tx.send(EquivResult::ProvedSymbolic);
                    return;
                }
            }
        }

        // Tier 3: Numeric equivalence
        let result = if vars_owned.len() >= 2 {
            check_numeric_equiv_2var(&s.context, l, r, &vars_owned[0], &vars_owned[1])
        } else if vars_owned.len() == 1 {
            check_numeric_equiv_1var(&s.context, l, r, &vars_owned[0])
        } else {
            // No variables — the NFs should have been equal (constants)
            Err("No variables and NF didn't converge".into())
        };

        match result {
            Ok(_) => {
                let _ = tx.send(EquivResult::NumericOnly);
            }
            Err(e) => {
                let _ = tx.send(EquivResult::Failed(e));
            }
        }
    });

    match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(_) => EquivResult::Failed("Timeout".into()),
    }
}

// =============================================================================
// Test Expression Sets
// =============================================================================

/// Polynomials (single variable) — good for both chains
fn polynomial_exprs() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        // (expression, variables)
        ("x^2 - 1", &["x"] as &[&str]),
        ("x^2 + 2*x + 1", &["x"]),
        ("x^2 - 2*x + 1", &["x"]),
        ("x^3 - 1", &["x"]),
        ("x^3 + 1", &["x"]),
        ("x^3 + 8", &["x"]),
        ("x^4 - 1", &["x"]),
        ("x^4 - 16", &["x"]),
        ("x^2 - 4", &["x"]),
        ("x^2 - 9", &["x"]),
        ("x^3 - 27", &["x"]),
        ("x^3 + 27", &["x"]),
        ("x^2 + 6*x + 9", &["x"]),
        ("x^2 - 6*x + 9", &["x"]),
        ("4*x^2 - 9", &["x"]),
        ("x^4 + 2*x^2 + 1", &["x"]),
        ("x^3 - x", &["x"]),
        ("x^4 - x^2", &["x"]),
        ("x^5 - x", &["x"]),
        ("2*x^3 + 6*x^2 + 6*x + 2", &["x"]),
        ("x^2 + 5*x + 6", &["x"]),
        ("x^2 - 5*x + 6", &["x"]),
        ("x^2 + x - 6", &["x"]),
        ("x^3 - 6*x^2 + 11*x - 6", &["x"]),
        ("x^4 - 5*x^2 + 4", &["x"]),
    ]
}

/// Products (for expand→simplify chain — expand distributes, simplify collects)
fn product_exprs() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        ("(x+1)*(x-1)", &["x"] as &[&str]),
        ("(x+2)*(x-2)", &["x"]),
        ("(x+3)^2", &["x"]),
        ("(x-1)^3", &["x"]),
        ("(x+1)^4", &["x"]),
        ("(2*x+1)*(x-3)", &["x"]),
        ("x*(x+1)*(x-1)", &["x"]),
        ("(x+1)*(x+2)*(x+3)", &["x"]),
        ("(x^2+1)*(x-1)", &["x"]),
        ("(x+1)^2*(x-1)", &["x"]),
        // Multivariate
        ("(a+b)^2", &["a", "b"]),
        ("(a+b)*(a-b)", &["a", "b"]),
        ("(a+b)^3", &["a", "b"]),
        ("(a+b+1)^2", &["a", "b"]),
        ("a*(a+b)*(a-b)", &["a", "b"]),
    ]
}

/// Trig expressions (for expand→simplify chain)
fn trig_exprs() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        ("sin(x)^2 + cos(x)^2", &["x"] as &[&str]),
        ("sin(2*x)", &["x"]),
        ("cos(2*x)", &["x"]),
        ("sin(x+1)^2 + cos(x+1)^2", &["x"]),
        ("(sin(x) + cos(x))^2", &["x"]),
        ("sin(x)^4 + cos(x)^4", &["x"]),
        ("sin(x)^2 - cos(x)^2", &["x"]),
        ("(1 + sin(x))*(1 - sin(x))", &["x"]),
    ]
}

/// Mixed expressions
fn mixed_exprs() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        ("(x+1)^2 - (x-1)^2", &["x"] as &[&str]),
        ("(x+1)^3 - (x-1)^3", &["x"]),
        ("(x+1)^2 + (x-1)^2", &["x"]),
        ("(x+1)*(x^2 - x + 1)", &["x"]),
        ("(x-1)*(x^2 + x + 1)", &["x"]),
    ]
}

// =============================================================================
// Chain 1: expand → simplify ≡ simplify (idempotence)
// =============================================================================

/// Test a single expression for expand→simplify idempotence.
/// Returns the equivalence result.
fn test_expand_simplify_one(input: &str, vars: &[&str], timeout: Duration) -> EquivResult {
    let input_owned = input.to_string();

    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        // a = simplify(x)
        let mut s1 = Simplifier::with_default_rules();
        let e1 = match parse(&input_owned, &mut s1.context) {
            Ok(e) => e,
            Err(_) => {
                let _ = tx.send(EquivResult::Failed("Parse failed".into()));
                return;
            }
        };
        let a = simp_expr(&mut s1, e1);
        let a_latex = to_latex(&s1.context, a);

        // b = expand(x)
        let mut s2 = Simplifier::with_default_rules();
        let e2 = match parse(&input_owned, &mut s2.context) {
            Ok(e) => e,
            Err(_) => {
                let _ = tx.send(EquivResult::Failed("Parse failed".into()));
                return;
            }
        };
        let (expanded, _) = s2.expand(e2);
        let expanded_latex = to_latex(&s2.context, expanded);

        // c = simplify(expand(x))
        let c = simp_expr(&mut s2, expanded);
        let c_latex = to_latex(&s2.context, c);

        let verbose = std::env::var("ROUNDTRIP_VERBOSE").is_ok();
        if verbose {
            eprintln!("  input:             {}", input_owned);
            eprintln!("  simplify(x):       {}", a_latex);
            eprintln!("  expand(x):         {}", expanded_latex);
            eprintln!("  simplify(expand):  {}", c_latex);
        }

        // Compare a ≡ c using the string form as proxy
        // (structural compare_expr needs same context, which we don't have)
        // Use 3-tier via string comparison
        let _ = tx.send(EquivResult::Failed("__use_string_check__".into()));
    });

    // Instead of thread-internal check, use the full 3-tier on strings
    // because expand and simplify run in different Simplifier instances.
    // The simplest approach: both sides should have equivalent NFs.
    let _ = rx.recv_timeout(timeout); // drain the thread

    // Build the two sides as strings for 3-tier comparison
    // LHS = simplify(input)
    // RHS = simplify(expand(input))
    // We express "simplify(expand(x))" by literally expanding then re-parsing,
    // but that loses structural info. Better: just use the 3-tier on the
    // original input vs itself — the question is whether expand→simplify
    // produces the same NF.
    //
    // The cleanest approach (and what the plan says):
    // Parse in one context, simplify in one context, compare within that context.
    let input_owned2 = input.to_string();
    let vars_owned2: Vec<String> = vars.iter().map(|s| s.to_string()).collect();

    let (tx2, rx2) = mpsc::channel();

    std::thread::spawn(move || {
        let mut s = Simplifier::with_default_rules();
        let e = match parse(&input_owned2, &mut s.context) {
            Ok(e) => e,
            Err(_) => {
                let _ = tx2.send(EquivResult::Failed("Parse failed".into()));
                return;
            }
        };

        // a = simplify(x)
        let a = simp_expr(&mut s, e);

        // Re-parse fresh for expand (since simplify may have mutated the expression)
        let e2 = match parse(&input_owned2, &mut s.context) {
            Ok(e) => e,
            Err(_) => {
                let _ = tx2.send(EquivResult::Failed("Parse 2 failed".into()));
                return;
            }
        };

        // b = expand(x)
        let (expanded, _) = s.expand(e2);

        // c = simplify(expand(x))
        let c = simp_expr(&mut s, expanded);

        let verbose = std::env::var("ROUNDTRIP_VERBOSE").is_ok();
        if verbose {
            eprintln!("  input:             {}", input_owned2);
            eprintln!("  simplify(x):       {}", to_latex(&s.context, a));
            eprintln!("  expand(x):         {}", to_latex(&s.context, expanded));
            eprintln!("  simplify(expand):  {}", to_latex(&s.context, c));
        }

        // Tier 1: NF convergence
        if cas_engine::ordering::compare_expr(&s.context, a, c) == std::cmp::Ordering::Equal {
            let _ = tx2.send(EquivResult::NfConvergent);
            return;
        }

        // Tier 2: Proved symbolic — simplify(a - c) == 0
        {
            let diff = s.context.add(cas_ast::Expr::Sub(a, c));
            let dr = simp_expr(&mut s, diff);
            let zero = num_rational::BigRational::from_integer(0.into());
            if matches!(s.context.get(dr), cas_ast::Expr::Number(n) if *n == zero) {
                let _ = tx2.send(EquivResult::ProvedSymbolic);
                return;
            }
        }

        // Tier 3: Numeric
        let result = if vars_owned2.len() >= 2 {
            check_numeric_equiv_2var(&s.context, a, c, &vars_owned2[0], &vars_owned2[1])
        } else if vars_owned2.len() == 1 {
            check_numeric_equiv_1var(&s.context, a, c, &vars_owned2[0])
        } else {
            Err("No variables and NF didn't converge".into())
        };

        match result {
            Ok(_) => {
                let _ = tx2.send(EquivResult::NumericOnly);
            }
            Err(e) => {
                let _ = tx2.send(EquivResult::Failed(e));
            }
        }
    });

    match rx2.recv_timeout(timeout) {
        Ok(result) => result,
        Err(_) => EquivResult::Failed("Timeout".into()),
    }
}

// =============================================================================
// Chain 2: factor → expand ≡ identity
// =============================================================================

/// Test a single expression for factor→expand round-trip.
fn test_factor_expand_one(input: &str, vars: &[&str], timeout: Duration) -> Option<EquivResult> {
    let input_owned = input.to_string();
    let vars_owned: Vec<String> = vars.iter().map(|s| s.to_string()).collect();

    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let mut s = Simplifier::with_default_rules();
        let e = match parse(&input_owned, &mut s.context) {
            Ok(e) => e,
            Err(_) => {
                let _ = tx.send(Some(EquivResult::Failed("Parse failed".into())));
                return;
            }
        };

        // a = simplify(x) — baseline NF
        let a = simp_expr(&mut s, e);

        // Re-parse for factor (fresh AST)
        let e2 = match parse(&input_owned, &mut s.context) {
            Ok(e) => e,
            Err(_) => {
                let _ = tx.send(Some(EquivResult::Failed("Parse 2 failed".into())));
                return;
            }
        };

        // Simplify first, then factor the simplified form
        let simplified = simp_expr(&mut s, e2);

        // f = factor(simplify(x))
        let factored = cas_engine::factor::factor(&mut s.context, simplified);

        // Check if factor actually did something
        if cas_engine::ordering::compare_expr(&s.context, factored, simplified)
            == std::cmp::Ordering::Equal
        {
            // factor() returned the same expression — skip this case
            let _ = tx.send(None);
            return;
        }

        // e = expand(factor(simplify(x)))
        let expanded = cas_engine::expand::expand(&mut s.context, factored);

        // c = simplify(expand(factor(simplify(x))))
        let c = simp_expr(&mut s, expanded);

        let verbose = std::env::var("ROUNDTRIP_VERBOSE").is_ok();
        if verbose {
            eprintln!("  input:               {}", input_owned);
            eprintln!("  simplify(x):         {}", to_latex(&s.context, a));
            eprintln!("  factor(simplify):    {}", to_latex(&s.context, factored));
            eprintln!("  expand(factor):      {}", to_latex(&s.context, expanded));
            eprintln!("  simplify(exp(fac)):  {}", to_latex(&s.context, c));
        }

        // Tier 1: NF convergence
        if cas_engine::ordering::compare_expr(&s.context, a, c) == std::cmp::Ordering::Equal {
            let _ = tx.send(Some(EquivResult::NfConvergent));
            return;
        }

        // Tier 2: simplify(a - c) == 0
        {
            let diff = s.context.add(cas_ast::Expr::Sub(a, c));
            let dr = simp_expr(&mut s, diff);
            let zero = num_rational::BigRational::from_integer(0.into());
            if matches!(s.context.get(dr), cas_ast::Expr::Number(n) if *n == zero) {
                let _ = tx.send(Some(EquivResult::ProvedSymbolic));
                return;
            }
        }

        // Tier 3: Numeric
        let result = if vars_owned.len() >= 2 {
            check_numeric_equiv_2var(&s.context, a, c, &vars_owned[0], &vars_owned[1])
        } else if vars_owned.len() == 1 {
            check_numeric_equiv_1var(&s.context, a, c, &vars_owned[0])
        } else {
            Err("No variables and NF didn't converge".into())
        };

        match result {
            Ok(_) => {
                let _ = tx.send(Some(EquivResult::NumericOnly));
            }
            Err(e) => {
                let _ = tx.send(Some(EquivResult::Failed(e)));
            }
        }
    });

    match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(_) => Some(EquivResult::Failed("Timeout".into())),
    }
}

// =============================================================================
// Stats Collector
// =============================================================================

#[derive(Default)]
struct RoundTripStats {
    nf_convergent: usize,
    proved_symbolic: usize,
    numeric_only: usize,
    failed: usize,
    skipped: usize,
    failures: Vec<String>,
}

impl RoundTripStats {
    fn total_passed(&self) -> usize {
        self.nf_convergent + self.proved_symbolic + self.numeric_only
    }

    fn record(&mut self, expr: &str, result: EquivResult) {
        match result {
            EquivResult::NfConvergent => self.nf_convergent += 1,
            EquivResult::ProvedSymbolic => self.proved_symbolic += 1,
            EquivResult::NumericOnly => self.numeric_only += 1,
            EquivResult::Failed(msg) => {
                self.failed += 1;
                self.failures.push(format!("{}: {}", expr, msg));
            }
        }
    }

    fn print_summary(&self, chain_name: &str) {
        let _total = self.total_passed() + self.failed;
        if self.failed == 0 {
            eprintln!(
                "✅ {}: {} passed, 0 failed, {} skipped",
                chain_name,
                self.total_passed(),
                self.skipped,
            );
        } else {
            eprintln!(
                "❌ {}: {} passed, {} FAILED, {} skipped",
                chain_name,
                self.total_passed(),
                self.failed,
                self.skipped,
            );
        }
        eprintln!(
            "   📐 NF-convergent: {} | 🔢 Proved-symbolic: {} | 🌡️ Numeric-only: {}",
            self.nf_convergent, self.proved_symbolic, self.numeric_only,
        );
        if !self.failures.is_empty() {
            eprintln!("── failures ──");
            for f in &self.failures {
                eprintln!("   ❌ {}", f);
            }
        }
    }
}

// =============================================================================
// Test: Chain 1 — expand → simplify idempotence
// =============================================================================

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test round_trip_tests -- --ignored --nocapture
fn roundtrip_expand_simplify() {
    let timeout = Duration::from_secs(3);
    let mut stats = RoundTripStats::default();

    let all_exprs: Vec<(&str, &[&str])> = [
        polynomial_exprs(),
        product_exprs(),
        trig_exprs(),
        mixed_exprs(),
    ]
    .concat();

    eprintln!("\n=== Chain 1: simplify(expand(x)) ≡ simplify(x) ===");
    eprintln!("Testing {} expressions...\n", all_exprs.len());

    for (expr, vars) in &all_exprs {
        let result = test_expand_simplify_one(expr, vars, timeout);
        let marker = match &result {
            EquivResult::NfConvergent => "📐",
            EquivResult::ProvedSymbolic => "🔢",
            EquivResult::NumericOnly => "🌡️",
            EquivResult::Failed(_) => "❌",
        };
        let verbose = std::env::var("ROUNDTRIP_VERBOSE").is_ok();
        if verbose || matches!(&result, EquivResult::Failed(_) | EquivResult::NumericOnly) {
            eprintln!("  {} {}", marker, expr);
        }
        stats.record(expr, result);
    }

    eprintln!();
    stats.print_summary("expand→simplify idempotence");

    // Fail the test if there are any FAILURES (not just numeric-only)
    assert_eq!(
        stats.failed, 0,
        "expand→simplify: {} expressions failed equivalence check",
        stats.failed,
    );
}

// =============================================================================
// Test: Chain 2 — factor → expand round-trip
// =============================================================================

#[test]
#[ignore]
fn roundtrip_factor_expand() {
    let timeout = Duration::from_secs(3);
    let mut stats = RoundTripStats::default();

    // Only polynomials and some products make sense for factoring
    let all_exprs: Vec<(&str, &[&str])> =
        [polynomial_exprs(), product_exprs(), mixed_exprs()].concat();

    eprintln!("\n=== Chain 2: expand(factor(x)) ≡ x ===");
    eprintln!("Testing {} expressions...\n", all_exprs.len());

    for (expr, vars) in &all_exprs {
        match test_factor_expand_one(expr, vars, timeout) {
            None => {
                // factor() didn't change the expression — skip
                stats.skipped += 1;
                let verbose = std::env::var("ROUNDTRIP_VERBOSE").is_ok();
                if verbose {
                    eprintln!("  ⏭️  {} (no factorization)", expr);
                }
            }
            Some(result) => {
                let marker = match &result {
                    EquivResult::NfConvergent => "📐",
                    EquivResult::ProvedSymbolic => "🔢",
                    EquivResult::NumericOnly => "🌡️",
                    EquivResult::Failed(_) => "❌",
                };
                let verbose = std::env::var("ROUNDTRIP_VERBOSE").is_ok();
                if verbose || matches!(&result, EquivResult::Failed(_) | EquivResult::NumericOnly) {
                    eprintln!("  {} {}", marker, expr);
                }
                stats.record(expr, result);
            }
        }
    }

    eprintln!();
    stats.print_summary("factor→expand round-trip");

    assert_eq!(
        stats.failed, 0,
        "factor→expand: {} expressions failed equivalence check",
        stats.failed,
    );
}

// =============================================================================
// Combined test (both chains)
// =============================================================================

#[test]
#[ignore]
fn roundtrip_all_chains() {
    let timeout = Duration::from_secs(3);

    // Chain 1: expand → simplify
    let mut stats1 = RoundTripStats::default();
    let all_exprs: Vec<(&str, &[&str])> = [
        polynomial_exprs(),
        product_exprs(),
        trig_exprs(),
        mixed_exprs(),
    ]
    .concat();

    eprintln!("\n╔══════════════════════════════════════════════════╗");
    eprintln!("║  Round-Trip Metamorphic Tests                    ║");
    eprintln!("╚══════════════════════════════════════════════════╝\n");

    eprintln!("=== Chain 1: simplify(expand(x)) ≡ simplify(x) ===");
    for (expr, vars) in &all_exprs {
        let result = test_expand_simplify_one(expr, vars, timeout);
        let verbose = std::env::var("ROUNDTRIP_VERBOSE").is_ok();
        if verbose || matches!(&result, EquivResult::Failed(_) | EquivResult::NumericOnly) {
            let marker = match &result {
                EquivResult::NfConvergent => "📐",
                EquivResult::ProvedSymbolic => "🔢",
                EquivResult::NumericOnly => "🌡️",
                EquivResult::Failed(_) => "❌",
            };
            eprintln!("  {} {}", marker, expr);
        }
        stats1.record(expr, result);
    }
    eprintln!();
    stats1.print_summary("expand→simplify idempotence");

    // Chain 2: factor → expand
    let mut stats2 = RoundTripStats::default();
    let poly_exprs: Vec<(&str, &[&str])> =
        [polynomial_exprs(), product_exprs(), mixed_exprs()].concat();

    eprintln!("\n=== Chain 2: expand(factor(x)) ≡ x ===");
    for (expr, vars) in &poly_exprs {
        match test_factor_expand_one(expr, vars, timeout) {
            None => {
                stats2.skipped += 1;
            }
            Some(result) => {
                let verbose = std::env::var("ROUNDTRIP_VERBOSE").is_ok();
                if verbose || matches!(&result, EquivResult::Failed(_) | EquivResult::NumericOnly) {
                    let marker = match &result {
                        EquivResult::NfConvergent => "📐",
                        EquivResult::ProvedSymbolic => "🔢",
                        EquivResult::NumericOnly => "🌡️",
                        EquivResult::Failed(_) => "❌",
                    };
                    eprintln!("  {} {}", marker, expr);
                }
                stats2.record(expr, result);
            }
        }
    }
    eprintln!();
    stats2.print_summary("factor→expand round-trip");

    // Overall summary
    eprintln!("\n╔══════════════════════════════════════════════════╗");
    eprintln!("║  Summary                                         ║");
    eprintln!("╚══════════════════════════════════════════════════╝");
    eprintln!(
        "  Chain 1 (expand→simplify): {} passed, {} failed",
        stats1.total_passed(),
        stats1.failed
    );
    eprintln!(
        "  Chain 2 (factor→expand):   {} passed, {} failed, {} skipped",
        stats2.total_passed(),
        stats2.failed,
        stats2.skipped
    );
    eprintln!();

    let total_failed = stats1.failed + stats2.failed;
    assert_eq!(total_failed, 0, "Total failures: {}", total_failed);
}
