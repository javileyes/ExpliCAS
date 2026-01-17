//! Step Validation Module - Educational Tool for Validating Student Steps
//!
//! This module provides infrastructure for validating whether a student's
//! intermediate expression `B` is a valid "partial simplification" of an
//! initial expression `A`.
//!
//! # Two Validation Routes
//!
//! 1. **Timeline Match** (preferred): Check if `B` appears in the simplification
//!    trace of `A`. Returns "natural" steps A→...→B.
//!
//! 2. **Equivalence Proof** (fallback): Prove that `A - B → 0`. Returns
//!    proof steps (not transformation steps).

use crate::implicit_domain::ImplicitCondition;
use crate::step::Step;
use cas_ast::{Context, Expr, ExprId};
use num_traits::ToPrimitive;
use std::collections::HashMap;

// =============================================================================
// Core Types
// =============================================================================

/// Result of validating a student's intermediate step
#[derive(Debug, Clone)]
pub enum StepCheckVerdict {
    /// B is equivalent to A and reduces complexity
    ValidAndSimpler {
        /// Conditions required for equivalence (e.g., x > 0)
        requires: Vec<ImplicitCondition>,
        /// How the validation was performed
        route: ValidationRoute,
        /// Complexity difference: negative = simpler
        complexity_delta: i32,
    },

    /// B is equivalent to A but doesn't reduce complexity
    ValidButNotSimpler {
        /// Conditions required for equivalence
        requires: Vec<ImplicitCondition>,
        /// How the validation was performed
        route: ValidationRoute,
        /// Complexity difference: positive or zero
        complexity_delta: i32,
    },

    /// B is NOT equivalent to A
    Invalid {
        /// Numeric counterexample showing the expressions differ
        counterexample: Option<CounterExample>,
        /// Human-readable reason
        reason: String,
    },

    /// Engine cannot determine equivalence
    Unknown {
        /// Why the engine couldn't determine equivalence
        reason: String,
    },
}

/// How the validation was performed
#[derive(Debug, Clone)]
pub enum ValidationRoute {
    /// B found in simplification timeline of A (ideal educational path)
    DirectTimeline {
        /// Steps from A to B (natural simplification steps)
        steps: Vec<Step>,
    },

    /// Proved via (A - B) → 0
    EquivalenceProof {
        /// Steps showing A - B simplifies to 0
        diff_steps: Vec<Step>,
    },
}

/// Numeric counterexample showing expressions differ
#[derive(Debug, Clone)]
pub struct CounterExample {
    /// Variable values used
    pub variable_values: HashMap<String, f64>,
    /// Value of expression A
    pub a_value: f64,
    /// Value of expression B
    pub b_value: f64,
    /// Absolute difference
    pub difference: f64,
}

// =============================================================================
// Soft Matching for Timeline Search
// =============================================================================

/// Check if two expressions match, allowing for commutative reordering
///
/// This uses a three-level matching strategy:
/// 1. Exact structural match (ExprId equality)
/// 2. Normalized match (sort terms/factors, normalize trivials)
/// 3. Lightweight canonical form comparison
pub fn expressions_match_soft(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    // Level 1: Exact structural match
    if a == b {
        return true;
    }

    // Level 2: Compute normalized keys and compare
    let key_a = compute_match_key(ctx, a);
    let key_b = compute_match_key(ctx, b);

    key_a == key_b
}

/// Compute a canonical string key for matching
fn compute_match_key(ctx: &Context, expr: ExprId) -> String {
    compute_match_key_inner(ctx, expr)
}

fn compute_match_key_inner(ctx: &Context, expr: ExprId) -> String {
    match ctx.get(expr) {
        Expr::Number(n) => {
            // Normalize rational to decimal string for matching
            if let (Some(num), Some(den)) = (n.numer().to_i64(), n.denom().to_i64()) {
                if den == 1 {
                    format!("{}", num)
                } else {
                    format!("{}/{}", num, den)
                }
            } else {
                format!("{:?}", n)
            }
        }
        Expr::Variable(name) => name.clone(),
        Expr::Constant(c) => format!("{:?}", c),

        Expr::Add(_lhs, _rhs) => {
            // Flatten and sort for commutative matching
            let mut terms = collect_add_terms(ctx, expr);
            terms.sort();
            format!("(+{})", terms.join(","))
        }

        Expr::Mul(_lhs, _rhs) => {
            // Flatten and sort for commutative matching
            let mut factors = collect_mul_factors(ctx, expr);
            factors.sort();
            format!("(*{})", factors.join(","))
        }

        Expr::Sub(lhs, rhs) => {
            let lhs_key = compute_match_key_inner(ctx, *lhs);
            let rhs_key = compute_match_key_inner(ctx, *rhs);
            format!("(-{},{})", lhs_key, rhs_key)
        }

        Expr::Div(lhs, rhs) => {
            let lhs_key = compute_match_key_inner(ctx, *lhs);
            let rhs_key = compute_match_key_inner(ctx, *rhs);
            format!("(/{},{})", lhs_key, rhs_key)
        }

        Expr::Pow(base, exp) => {
            let base_key = compute_match_key_inner(ctx, *base);
            let exp_key = compute_match_key_inner(ctx, *exp);
            format!("(^{},{})", base_key, exp_key)
        }

        Expr::Neg(inner) => {
            let inner_key = compute_match_key_inner(ctx, *inner);
            format!("(neg{})", inner_key)
        }

        Expr::Function(name, args) => {
            let arg_keys: Vec<String> = args
                .iter()
                .map(|a| compute_match_key_inner(ctx, *a))
                .collect();
            format!("(fn:{},{})", name, arg_keys.join(","))
        }

        Expr::Matrix { rows, cols, data } => {
            let data_keys: Vec<String> = data
                .iter()
                .map(|d| compute_match_key_inner(ctx, *d))
                .collect();
            format!("(matrix:{}x{},{})", rows, cols, data_keys.join(","))
        }

        Expr::SessionRef(id) => format!("(#{})", id),
    }
}

/// Collect all additive terms (flatten nested Add)
fn collect_add_terms(ctx: &Context, expr: ExprId) -> Vec<String> {
    let mut terms = Vec::new();
    collect_add_terms_inner(ctx, expr, &mut terms);
    terms
}

fn collect_add_terms_inner(ctx: &Context, expr: ExprId, terms: &mut Vec<String>) {
    match ctx.get(expr) {
        Expr::Add(lhs, rhs) => {
            collect_add_terms_inner(ctx, *lhs, terms);
            collect_add_terms_inner(ctx, *rhs, terms);
        }
        _ => {
            terms.push(compute_match_key_inner(ctx, expr));
        }
    }
}

/// Collect all multiplicative factors (flatten nested Mul)
fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<String> {
    let mut factors = Vec::new();
    collect_mul_factors_inner(ctx, expr, &mut factors);
    factors
}

fn collect_mul_factors_inner(ctx: &Context, expr: ExprId, factors: &mut Vec<String>) {
    match ctx.get(expr) {
        Expr::Mul(lhs, rhs) => {
            collect_mul_factors_inner(ctx, *lhs, factors);
            collect_mul_factors_inner(ctx, *rhs, factors);
        }
        _ => {
            factors.push(compute_match_key_inner(ctx, expr));
        }
    }
}

// =============================================================================
// Complexity Metric
// =============================================================================

/// Compute complexity score for an expression
///
/// Uses weighted node count. Lower score = simpler expression.
pub fn compute_complexity(ctx: &Context, expr: ExprId) -> i32 {
    compute_complexity_inner(ctx, expr) as i32
}

fn compute_complexity_inner(ctx: &Context, expr: ExprId) -> f64 {
    match ctx.get(expr) {
        Expr::Number(n) => {
            // Larger rationals are more complex
            if n.denom().to_i64() == Some(1) {
                1.0
            } else {
                1.5 // Fractions slightly more complex
            }
        }
        Expr::Variable(_) => 1.0,
        Expr::Constant(_) => 1.0,

        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
            1.0 + compute_complexity_inner(ctx, *lhs) + compute_complexity_inner(ctx, *rhs)
        }

        Expr::Mul(lhs, rhs) | Expr::Div(lhs, rhs) => {
            1.0 + compute_complexity_inner(ctx, *lhs) + compute_complexity_inner(ctx, *rhs)
        }

        Expr::Pow(base, exp) => {
            let exp_complexity = compute_complexity_inner(ctx, *exp);
            let base_complexity = compute_complexity_inner(ctx, *base);

            // Non-integer powers are more complex
            let power_weight = if is_integer_expr(ctx, *exp) { 1.0 } else { 2.0 };

            power_weight + base_complexity + exp_complexity
        }

        Expr::Neg(inner) => 0.5 + compute_complexity_inner(ctx, *inner),

        Expr::Function(_, args) => {
            let args_complexity: f64 = args.iter().map(|a| compute_complexity_inner(ctx, *a)).sum();
            2.0 + args_complexity
        }

        Expr::Matrix { data, .. } => {
            let data_complexity: f64 = data.iter().map(|d| compute_complexity_inner(ctx, *d)).sum();
            2.0 + data_complexity
        }

        Expr::SessionRef(_) => 1.0,
    }
}

/// Check if expression is an integer
fn is_integer_expr(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => n.denom().to_i64() == Some(1),
        Expr::Neg(inner) => is_integer_expr(ctx, *inner),
        _ => false,
    }
}

// =============================================================================
// Timeline Search
// =============================================================================

/// Search for target expression in simplification timeline
///
/// Returns the index of the step where target appears, if found.
pub fn find_in_timeline(ctx: &Context, steps: &[Step], target: ExprId) -> Option<usize> {
    for (idx, step) in steps.iter().enumerate() {
        // Check if the step's "after" matches our target
        if expressions_match_soft(ctx, step.after, target) {
            return Some(idx);
        }
    }
    None
}

// =============================================================================
// Counterexample Search
// =============================================================================

/// Attempt to find a numeric counterexample showing A ≠ B
///
/// Uses a fixed set of test values to avoid rand dependency.
/// Returns None if no counterexample found.
pub fn find_counterexample(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    vars: &[String],
) -> Option<CounterExample> {
    // Fixed test values (no rand dependency)
    // Mix of small integers, rationals, and special values
    let test_values: &[f64] = &[
        0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        3.0,
        -3.0,
        0.25,
        0.1,
        -0.1,
        std::f64::consts::PI,
        std::f64::consts::E,
        1.5,
        -1.5,
        4.0,
        5.0,
        std::f64::consts::FRAC_1_SQRT_2, // sqrt(2)/2
        std::f64::consts::SQRT_2,        // sqrt(2)
    ];

    // For single variable, test all values
    if vars.len() == 1 {
        for &val in test_values {
            let mut variable_values = HashMap::new();
            variable_values.insert(vars[0].clone(), val);

            if let Some(ce) = try_counterexample(ctx, a, b, &variable_values) {
                return Some(ce);
            }
        }
    } else if vars.len() == 2 {
        // For two variables, test a grid
        for &v1 in &test_values[..10] {
            for &v2 in &test_values[..10] {
                let mut variable_values = HashMap::new();
                variable_values.insert(vars[0].clone(), v1);
                variable_values.insert(vars[1].clone(), v2);

                if let Some(ce) = try_counterexample(ctx, a, b, &variable_values) {
                    return Some(ce);
                }
            }
        }
    } else {
        // For more variables, test a subset
        for i in 0..50 {
            let mut variable_values = HashMap::new();
            for (j, var) in vars.iter().enumerate() {
                let idx = (i + j) % test_values.len();
                variable_values.insert(var.clone(), test_values[idx]);
            }

            if let Some(ce) = try_counterexample(ctx, a, b, &variable_values) {
                return Some(ce);
            }
        }
    }

    None
}

fn try_counterexample(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    variable_values: &HashMap<String, f64>,
) -> Option<CounterExample> {
    let a_result = evaluate_expr(ctx, a, variable_values);
    let b_result = evaluate_expr(ctx, b, variable_values);

    match (a_result, b_result) {
        (Some(a_val), Some(b_val)) => {
            let diff = (a_val - b_val).abs();
            // Use relative tolerance for larger values
            let tol = 1e-9 * (1.0 + a_val.abs().max(b_val.abs()));

            if diff > tol && a_val.is_finite() && b_val.is_finite() {
                Some(CounterExample {
                    variable_values: variable_values.clone(),
                    a_value: a_val,
                    b_value: b_val,
                    difference: diff,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Simple numeric evaluation of an expression
fn evaluate_expr(ctx: &Context, expr: ExprId, vars: &HashMap<String, f64>) -> Option<f64> {
    match ctx.get(expr) {
        Expr::Number(n) => {
            let num = n.numer().to_f64()?;
            let den = n.denom().to_f64()?;
            Some(num / den)
        }
        Expr::Variable(name) => vars.get(name).copied(),
        Expr::Constant(c) => {
            use cas_ast::Constant;
            match c {
                Constant::Pi => Some(std::f64::consts::PI),
                Constant::E => Some(std::f64::consts::E),
                Constant::I => None, // Can't evaluate complex
                Constant::Phi => Some(1.618033988749895),
                Constant::Infinity | Constant::Undefined => None,
            }
        }

        Expr::Add(lhs, rhs) => {
            let l = evaluate_expr(ctx, *lhs, vars)?;
            let r = evaluate_expr(ctx, *rhs, vars)?;
            Some(l + r)
        }
        Expr::Sub(lhs, rhs) => {
            let l = evaluate_expr(ctx, *lhs, vars)?;
            let r = evaluate_expr(ctx, *rhs, vars)?;
            Some(l - r)
        }
        Expr::Mul(lhs, rhs) => {
            let l = evaluate_expr(ctx, *lhs, vars)?;
            let r = evaluate_expr(ctx, *rhs, vars)?;
            Some(l * r)
        }
        Expr::Div(lhs, rhs) => {
            let l = evaluate_expr(ctx, *lhs, vars)?;
            let r = evaluate_expr(ctx, *rhs, vars)?;
            if r.abs() < 1e-15 {
                None // Division by zero
            } else {
                Some(l / r)
            }
        }
        Expr::Pow(base, exp) => {
            let b = evaluate_expr(ctx, *base, vars)?;
            let e = evaluate_expr(ctx, *exp, vars)?;
            let result = b.powf(e);
            if result.is_nan() || result.is_infinite() {
                None
            } else {
                Some(result)
            }
        }
        Expr::Neg(inner) => {
            let val = evaluate_expr(ctx, *inner, vars)?;
            Some(-val)
        }

        Expr::Function(name, args) => {
            if args.len() != 1 {
                return None; // Only single-arg functions for now
            }
            let arg = evaluate_expr(ctx, args[0], vars)?;

            let result = match name.as_str() {
                "sin" => arg.sin(),
                "cos" => arg.cos(),
                "tan" => arg.tan(),
                "exp" => arg.exp(),
                "ln" | "log" => {
                    if arg <= 0.0 {
                        return None;
                    }
                    arg.ln()
                }
                "sqrt" => {
                    if arg < 0.0 {
                        return None;
                    }
                    arg.sqrt()
                }
                "abs" => arg.abs(),
                "arcsin" | "asin" => {
                    if arg.abs() > 1.0 {
                        return None;
                    }
                    arg.asin()
                }
                "arccos" | "acos" => {
                    if arg.abs() > 1.0 {
                        return None;
                    }
                    arg.acos()
                }
                "arctan" | "atan" => arg.atan(),
                "sinh" => arg.sinh(),
                "cosh" => arg.cosh(),
                "tanh" => arg.tanh(),
                _ => return None,
            };

            if result.is_nan() || result.is_infinite() {
                None
            } else {
                Some(result)
            }
        }

        Expr::Matrix { .. } | Expr::SessionRef(_) => None,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx_with_expr(input: &str) -> (Context, ExprId) {
        let mut ctx = Context::new();
        let expr = cas_parser::parse(input, &mut ctx).expect("parse failed");
        (ctx, expr)
    }

    #[test]
    fn test_expressions_match_soft_identical() {
        let (ctx, a) = make_ctx_with_expr("x + 1");
        assert!(expressions_match_soft(&ctx, a, a));
    }

    #[test]
    fn test_expressions_match_soft_commutative_add() {
        let mut ctx = Context::new();
        let a = cas_parser::parse("x + 1", &mut ctx).unwrap();
        let b = cas_parser::parse("1 + x", &mut ctx).unwrap();
        assert!(expressions_match_soft(&ctx, a, b));
    }

    #[test]
    fn test_expressions_match_soft_commutative_mul() {
        let mut ctx = Context::new();
        let a = cas_parser::parse("x * y", &mut ctx).unwrap();
        let b = cas_parser::parse("y * x", &mut ctx).unwrap();
        assert!(expressions_match_soft(&ctx, a, b));
    }

    #[test]
    fn test_expressions_no_match_different() {
        let mut ctx = Context::new();
        let a = cas_parser::parse("x + 1", &mut ctx).unwrap();
        let b = cas_parser::parse("x + 2", &mut ctx).unwrap();
        assert!(!expressions_match_soft(&ctx, a, b));
    }

    #[test]
    fn test_complexity_simple_vs_complex() {
        let mut ctx = Context::new();
        let simple = cas_parser::parse("x", &mut ctx).unwrap();
        let complex = cas_parser::parse("x^2 + 2*x + 1", &mut ctx).unwrap();

        let c_simple = compute_complexity(&ctx, simple);
        let c_complex = compute_complexity(&ctx, complex);

        assert!(
            c_simple < c_complex,
            "x should be simpler than x^2 + 2x + 1"
        );
    }

    #[test]
    fn test_complexity_factored_vs_expanded() {
        let mut ctx = Context::new();
        let factored = cas_parser::parse("(x + 1)^2", &mut ctx).unwrap();
        let expanded = cas_parser::parse("x^2 + 2*x + 1", &mut ctx).unwrap();

        let c_factored = compute_complexity(&ctx, factored);
        let c_expanded = compute_complexity(&ctx, expanded);

        // Factored form should be simpler
        assert!(
            c_factored < c_expanded,
            "factored ({}) should be simpler than expanded ({})",
            c_factored,
            c_expanded
        );
    }

    #[test]
    fn test_counterexample_different_exprs() {
        let mut ctx = Context::new();
        let a = cas_parser::parse("x^2 + 1", &mut ctx).unwrap();
        let b = cas_parser::parse("(x + 1)^2", &mut ctx).unwrap();

        let vars = vec!["x".to_string()];
        let ce = find_counterexample(&ctx, a, b, &vars);

        assert!(
            ce.is_some(),
            "Should find counterexample for x^2+1 vs (x+1)^2"
        );
    }

    #[test]
    fn test_counterexample_equivalent_exprs() {
        let mut ctx = Context::new();
        let a = cas_parser::parse("x + 1", &mut ctx).unwrap();
        let b = cas_parser::parse("1 + x", &mut ctx).unwrap();

        let vars = vec!["x".to_string()];
        let ce = find_counterexample(&ctx, a, b, &vars);

        assert!(
            ce.is_none(),
            "Should NOT find counterexample for equivalent exprs"
        );
    }
}
