//! Test Utilities for Mathematical Equivalence Testing
//!
//! This module provides helpers for testing mathematical correctness without
//! relying on string matching. Tests can verify:
//! - Symbolic equivalence (difference simplifies to zero)
//! - Numeric equivalence (evaluates to same value at sample points)
//!
//! # Usage
//!
//! ```ignore
//! use test_utils::*;
//!
//! // Symbolic: verify expr simplifies to expected
//! assert_simplifies_to("sin(x)^2 + cos(x)^2", "1");
//!
//! // Numeric: verify expressions are equivalent over a range
//! assert_equiv_numeric_1var("tan(atan(x))", "x", "x", -10.0, 10.0, 100, 1e-9, |_| true);
//! ```

#![allow(dead_code)] // These are helpers for other tests to use

use cas_ast::{Context, Expr, ExprId};
use cas_engine::eval_f64;
use cas_engine::helpers::is_zero;
use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use std::collections::HashMap;

/// Tolerance for numeric comparison
pub const DEFAULT_TOL: f64 = 1e-9;

// =============================================================================
// Numeric Equivalence Options
// =============================================================================

/// Options for numeric equivalence testing with min_valid and atol/rtol
#[derive(Clone, Copy, Debug)]
pub struct NumericEquivOptions {
    /// Number of sample points to test
    pub samples: usize,
    /// Minimum number of valid (non-None) samples required
    pub min_valid: usize,
    /// Absolute tolerance for comparisons
    pub atol: f64,
    /// Relative tolerance for comparisons
    pub rtol: f64,
}

impl NumericEquivOptions {
    /// Create new options with sensible defaults
    /// min_valid defaults to samples/2 (at least 32)
    pub fn new(samples: usize) -> Self {
        let min_valid = (samples / 2).max(32);
        Self {
            samples,
            min_valid,
            atol: 1e-10,
            rtol: 1e-10,
        }
    }

    /// Set absolute and relative tolerance
    pub fn with_tol(mut self, atol: f64, rtol: f64) -> Self {
        self.atol = atol;
        self.rtol = rtol;
        self
    }

    /// Set minimum valid samples required
    pub fn with_min_valid(mut self, min_valid: usize) -> Self {
        self.min_valid = min_valid;
        self
    }
}

impl Default for NumericEquivOptions {
    fn default() -> Self {
        Self::new(200)
    }
}

/// Check if two f64 values are approximately equal using absolute tolerance only
pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    approx_eq_atol_rtol(a, b, tol, 0.0)
}

/// Check if two f64 values are approximately equal using both absolute and relative tolerance
/// |a - b| <= atol + rtol * max(|a|, |b|)
pub fn approx_eq_atol_rtol(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    diff <= atol + rtol * scale
}

/// Pretty-print an expression
pub fn pretty(ctx: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

// =============================================================================
// Symbolic Equivalence
// =============================================================================

/// Assert that two expressions are equivalent by checking (a - b) simplifies to 0.
///
/// # Example
/// ```ignore
/// let mut ctx = Context::new();
/// let a = parse("sin(x)^2 + cos(x)^2", &mut ctx).unwrap();
/// let b = parse("1", &mut ctx).unwrap();
/// assert_equiv_simplify(&mut Simplifier::with_default_rules(), a, b);
/// ```
pub fn assert_equiv_simplify(simplifier: &mut Simplifier, a: ExprId, b: ExprId) {
    let diff = simplifier.context.add(Expr::Sub(a, b));
    let (simplified, _) = simplifier.simplify(diff);

    assert!(
        is_zero(&simplifier.context, simplified),
        "Not equivalent.\na: {}\nb: {}\nDiff simplified to: {} (expected 0)",
        pretty(&simplifier.context, a),
        pretty(&simplifier.context, b),
        pretty(&simplifier.context, simplified)
    );
}

/// Assert that an input expression simplifies to an expected expression.
///
/// # Example
/// ```ignore
/// assert_simplifies_to("sin(x)^2 + cos(x)^2", "1");
/// ```
pub fn assert_simplifies_to(input: &str, expected: &str) {
    let mut simplifier = Simplifier::with_default_rules();
    let a = parse(input, &mut simplifier.context).expect("Failed to parse input");
    let b = parse(expected, &mut simplifier.context).expect("Failed to parse expected");

    let (a_simplified, _) = simplifier.simplify(a);
    let (b_simplified, _) = simplifier.simplify(b);

    assert_equiv_simplify(&mut simplifier, a_simplified, b_simplified);
}

/// Assert that input simplifies to zero.
///
/// # Example
/// ```ignore
/// assert_simplifies_to_zero("sin(x)^2 + cos(x)^2 - 1");
/// ```
pub fn assert_simplifies_to_zero(input: &str) {
    assert_simplifies_to(input, "0");
}

// =============================================================================
// Numeric Equivalence
// =============================================================================

/// Assert numeric equivalence for expressions with one variable.
///
/// Evaluates both expressions at sample points and checks they're equal within tolerance.
/// Use `filter` to exclude singular points (e.g., where denominator is zero).
///
/// **Enforces minimum valid samples**: If less than 50% of samples are valid,
/// the test fails to prevent false positives from over-restrictive filters.
///
/// # Example
/// ```ignore
/// assert_equiv_numeric_1var(
///     "tan(atan(x))",
///     "x",
///     "x",
///     -10.0, 10.0,
///     100,
///     1e-9,
///     |_| true
/// );
/// ```
#[allow(clippy::too_many_arguments)]
pub fn assert_equiv_numeric_1var(
    input: &str,
    expected: &str,
    var: &str,
    lo: f64,
    hi: f64,
    samples: usize,
    tol: f64,
    filter: impl Fn(f64) -> bool,
) {
    // Use conservative min_valid: at least 50% of samples must be valid
    let min_valid = (samples / 2).max(10);

    let mut simplifier = Simplifier::with_default_rules();
    let a = parse(input, &mut simplifier.context).expect("Failed to parse input");
    let b = parse(expected, &mut simplifier.context).expect("Failed to parse expected");

    let (a_simplified, _) = simplifier.simplify(a);
    let (b_simplified, _) = simplifier.simplify(b);

    let mut valid = 0usize;
    let mut filtered_out = 0usize;
    let mut eval_failed = 0usize;
    let mut failed = false;
    let mut fail_msg = String::new();

    for i in 0..samples {
        let t = (i as f64 + 0.5) / samples as f64;
        let x = lo + (hi - lo) * t;

        if !filter(x) {
            filtered_out += 1;
            continue;
        }

        let mut var_map = HashMap::new();
        var_map.insert(var.to_string(), x);

        let va = eval_f64(&simplifier.context, a_simplified, &var_map);
        let vb = eval_f64(&simplifier.context, b_simplified, &var_map);

        match (va, vb) {
            (Some(va), Some(vb)) => {
                valid += 1;
                // Use both absolute and relative tolerance
                if !approx_eq_atol_rtol(va, vb, tol, tol) {
                    failed = true;
                    let diff = (va - vb).abs();
                    let scale = va.abs().max(vb.abs()).max(1.0);
                    fail_msg = format!(
                        "Numeric mismatch at {}={}:\n  input  -> {} = {:.15}\n  expected -> {} = {:.15}\n  |diff| = {:.3e}\n  allowed = atol({:.3e}) + rtol({:.3e})*scale({:.3e}) = {:.3e}",
                        var, x,
                        pretty(&simplifier.context, a_simplified), va,
                        pretty(&simplifier.context, b_simplified), vb,
                        diff, tol, tol, scale, tol + tol * scale
                    );
                    break;
                }
            }
            (None, _) | (_, None) => {
                eval_failed += 1;
                continue;
            }
        }
    }

    assert!(!failed, "{}", fail_msg);

    // Enforce minimum valid samples
    assert!(
        valid >= min_valid,
        "Too few valid samples: {} valid < {} min_valid (out of {} samples)\n  \
         filtered_out={}, eval_failed={}\n  \
         Hint: Filter may be too restrictive or evaluation is failing too often\n  \
         input: {}\n  expected: {}",
        valid,
        min_valid,
        samples,
        filtered_out,
        eval_failed,
        input,
        expected
    );

    // Warn about fragile tests that barely pass
    let fragility_ratio = valid as f64 / samples as f64;
    if fragility_ratio < 0.6 || eval_failed > samples / 4 {
        eprintln!(
            "⚠️  FRAGILE TEST: {:.0}% valid samples, {:.0}% eval failures\n   \
             input: {}\n   \
             Consider: wider range, looser filter, or investigate eval failures",
            fragility_ratio * 100.0,
            eval_failed as f64 / samples as f64 * 100.0,
            input
        );
    }
}

/// Assert numeric equivalence for expressions with two variables.
#[allow(clippy::too_many_arguments)]
pub fn assert_equiv_numeric_2var(
    input: &str,
    expected: &str,
    var1: &str,
    lo1: f64,
    hi1: f64,
    var2: &str,
    lo2: f64,
    hi2: f64,
    samples_per_var: usize,
    tol: f64,
    filter: impl Fn(f64, f64) -> bool,
) {
    let mut simplifier = Simplifier::with_default_rules();
    let a = parse(input, &mut simplifier.context).expect("Failed to parse input");
    let b = parse(expected, &mut simplifier.context).expect("Failed to parse expected");

    let (a_simplified, _) = simplifier.simplify(a);
    let (b_simplified, _) = simplifier.simplify(b);

    let mut tested = 0;

    for i in 0..samples_per_var {
        for j in 0..samples_per_var {
            let t1 = (i as f64 + 0.5) / samples_per_var as f64;
            let t2 = (j as f64 + 0.5) / samples_per_var as f64;
            let x = lo1 + (hi1 - lo1) * t1;
            let y = lo2 + (hi2 - lo2) * t2;

            if !filter(x, y) {
                continue;
            }

            let mut var_map = HashMap::new();
            var_map.insert(var1.to_string(), x);
            var_map.insert(var2.to_string(), y);

            let va = eval_f64(&simplifier.context, a_simplified, &var_map);
            let vb = eval_f64(&simplifier.context, b_simplified, &var_map);

            if let (Some(va), Some(vb)) = (va, vb) {
                tested += 1;
                assert!(
                    approx_eq(va, vb, tol),
                    "Numeric mismatch at {}={}, {}={}:\n  {} = {}\n  {} = {}",
                    var1,
                    x,
                    var2,
                    y,
                    pretty(&simplifier.context, a_simplified),
                    va,
                    pretty(&simplifier.context, b_simplified),
                    vb
                );
            }
        }
    }

    assert!(tested > 0, "No valid samples tested");
}

// =============================================================================
// Simplification result helpers
// =============================================================================

/// Simplify an expression and return the result as a string
pub fn simplify_to_string(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("Failed to parse");
    let (result, _) = simplifier.simplify(expr);
    pretty(&simplifier.context, result)
}

/// Check if simplification produces a specific string result
/// Use this sparingly - prefer assert_simplifies_to for equivalence
pub fn simplifies_to_exactly(input: &str, expected_str: &str) -> bool {
    simplify_to_string(input) == expected_str
}
