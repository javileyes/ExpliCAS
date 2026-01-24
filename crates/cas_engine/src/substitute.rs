//! Power-aware algebraic substitution.
//!
//! Extends simple structural substitution to recognize power relationships:
//! - `substitute(x^4, x^2, y)` → `y^2` (recognizes x^4 = (x^2)^2)
//! - `substitute(x^3, x^2, y)` → `y*x` (with remainder)
//!
//! This is crucial for calculus operations like u-substitution in integration.

use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;

use crate::ordering::compare_expr;

/// Options for power-aware substitution.
#[derive(Clone, Copy, Debug)]
pub struct SubstituteOptions {
    /// If true, recognizes power patterns: x^4 with target x^2 → y^2.
    /// If false, only exact structural matches are replaced.
    pub power_aware: bool,

    /// If true, allows partial matches with remainder: x^3 with target x^2 → y*x.
    /// If false, only exact divisible exponents are replaced.
    /// Only relevant when power_aware is true.
    pub allow_remainder: bool,

    /// If true, collect substitution steps for traceability.
    pub collect_steps: bool,
}

impl Default for SubstituteOptions {
    fn default() -> Self {
        Self {
            power_aware: true,
            allow_remainder: true,
            collect_steps: false,
        }
    }
}

impl SubstituteOptions {
    /// Create options for exact match only (no power awareness).
    pub fn exact() -> Self {
        Self {
            power_aware: false,
            allow_remainder: false,
            collect_steps: false,
        }
    }

    /// Create power-aware options without remainder.
    pub fn power_aware_no_remainder() -> Self {
        Self {
            power_aware: true,
            allow_remainder: false,
            collect_steps: false,
        }
    }

    /// Enable step collection.
    pub fn with_steps(mut self) -> Self {
        self.collect_steps = true;
        self
    }
}

/// A single substitution step for traceability.
#[derive(Clone, Debug)]
pub struct SubstituteStep {
    /// Rule name: "SubstituteExact", "SubstitutePowerMultiple", "SubstitutePowOfTarget"
    pub rule: String,
    /// Expression before substitution (as string)
    pub before: String,
    /// Expression after substitution (as string)
    pub after: String,
    /// Optional note (e.g., "n=4, k=2, m=2")
    pub note: Option<String>,
}

/// Result of substitution including optional steps.
#[derive(Clone, Debug)]
pub struct SubstituteResult {
    pub expr: ExprId,
    pub steps: Vec<SubstituteStep>,
}

/// Extract integer exponent from a Number expression.
fn as_int_exponent(ctx: &Context, e: ExprId) -> Option<BigInt> {
    if let Expr::Number(q) = ctx.get(e) {
        if q.is_integer() {
            return Some(q.to_integer());
        }
    }
    None
}

/// Extract power pattern: returns (base, exponent) if expr is Pow(base, int).
fn as_power_int(ctx: &Context, e: ExprId) -> Option<(ExprId, BigInt)> {
    if let Expr::Pow(base, exp) = ctx.get(e) {
        if let Some(k) = as_int_exponent(ctx, *exp) {
            if k > BigInt::zero() {
                return Some((*base, k));
            }
        }
    }
    None
}

/// Create a Number node from BigInt.
fn mk_int(ctx: &mut Context, n: &BigInt) -> ExprId {
    ctx.add(Expr::Number(BigRational::from_integer(n.clone())))
}

/// Create Pow(base, exp) with integer exponent, handling special cases.
fn mk_pow_int(ctx: &mut Context, base: ExprId, exp: &BigInt) -> ExprId {
    if exp.is_zero() {
        // x^0 = 1
        return ctx.add(Expr::Number(BigRational::one()));
    }
    if exp == &BigInt::one() {
        // x^1 = x
        return base;
    }
    let exp_id = mk_int(ctx, exp);
    ctx.add(Expr::Pow(base, exp_id))
}

/// Try to match a power expression against a power target.
/// Returns Some(new_expr) if matched, None otherwise.
fn try_power_substitution(
    ctx: &mut Context,
    node: ExprId,
    target_base: ExprId,
    target_exp: &BigInt,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> Option<ExprId> {
    // Node must also be a power with same base
    let (node_base, node_exp) = as_power_int(ctx, node)?;

    // Check if bases are structurally equal
    if compare_expr(ctx, node_base, target_base) != Ordering::Equal {
        return None;
    }

    // Compute quotient and remainder: node_exp = q * target_exp + r
    let q = &node_exp / target_exp;
    let r = &node_exp % target_exp;

    if r.is_zero() {
        // Exact divisible: x^4 with x^2 → y^2
        return Some(mk_pow_int(ctx, replacement, &q));
    }

    // Remainder case: x^3 with x^2 → y^1 * x^1
    if opts.allow_remainder && q > BigInt::zero() {
        let yq = mk_pow_int(ctx, replacement, &q);
        let xr = mk_pow_int(ctx, node_base, &r);
        return Some(ctx.add(Expr::Mul(yq, xr)));
    }

    None
}

/// Perform power-aware substitution.
///
/// Replaces occurrences of `target` with `replacement` in the expression tree.
/// When `opts.power_aware` is true and target is a power expression (e.g., x^2),
/// it also recognizes higher powers of the same base (e.g., x^4 → y^2).
///
/// # Example
/// ```ignore
/// // x^4 + x^2 + 1 with x^2 → y  becomes  y^2 + y + 1
/// let result = substitute_power_aware(ctx, expr, target, replacement, SubstituteOptions::default());
/// ```
pub fn substitute_power_aware(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> ExprId {
    // Pre-compute target power pattern if power_aware
    let target_power = if opts.power_aware {
        as_power_int(ctx, target)
    } else {
        None
    };

    substitute_inner(
        ctx,
        root,
        target,
        replacement,
        target_power.as_ref(),
        opts,
        &mut Vec::new(),
    )
}

/// Perform power-aware substitution with step collection.
///
/// Same as `substitute_power_aware` but returns steps for traceability.
pub fn substitute_with_steps(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> SubstituteResult {
    use cas_ast::DisplayExpr;

    // Pre-compute target power pattern if power_aware
    let target_power = if opts.power_aware {
        as_power_int(ctx, target)
    } else {
        None
    };

    let mut steps = Vec::new();
    let expr = substitute_inner(
        ctx,
        root,
        target,
        replacement,
        target_power.as_ref(),
        opts,
        &mut steps,
    );

    // Convert ExprId steps to string steps
    let string_steps: Vec<SubstituteStep> = steps
        .into_iter()
        .map(|(rule, before, after, note)| SubstituteStep {
            rule,
            before: format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: before
                }
            ),
            after: format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: after
                }
            ),
            note,
        })
        .collect();

    SubstituteResult {
        expr,
        steps: string_steps,
    }
}

/// Inner recursive substitution function.
/// Steps tuple: (rule_name, before_expr, after_expr, optional_note)
fn substitute_inner(
    ctx: &mut Context,
    node: ExprId,
    target: ExprId,
    replacement: ExprId,
    target_power: Option<&(ExprId, BigInt)>,
    opts: SubstituteOptions,
    steps: &mut Vec<(String, ExprId, ExprId, Option<String>)>,
) -> ExprId {
    // 1) Exact structural match (highest priority)
    if node == target {
        steps.push(("SubstituteExact".to_string(), node, replacement, None));
        return replacement;
    }

    // 2) Power pattern match (if target is a power)
    if let Some((target_base, target_exp)) = target_power {
        if let Some(substituted) =
            try_power_substitution(ctx, node, *target_base, target_exp, replacement, opts)
        {
            // Determine which power rule was applied
            if as_power_int(ctx, node).map(|(b, _)| b) == Some(*target_base) {
                // Power multiple: base^n -> repl^(n/k)
                if let Some((_, node_exp)) = as_power_int(ctx, node) {
                    let m = &node_exp / target_exp;
                    steps.push((
                        "SubstitutePowerMultiple".to_string(),
                        node,
                        substituted,
                        Some(format!("n={}, k={}, m={}", node_exp, target_exp, m)),
                    ));
                } else {
                    steps.push((
                        "SubstitutePowerMultiple".to_string(),
                        node,
                        substituted,
                        None,
                    ));
                }
            } else {
                // Pow of target
                steps.push(("SubstitutePowOfTarget".to_string(), node, substituted, None));
            };
            return substituted;
        }
    }

    // 3) Recurse into children
    let expr = ctx.get(node).clone();
    match expr {
        Expr::Add(l, r) => {
            let nl = substitute_inner(ctx, l, target, replacement, target_power, opts, steps);
            let nr = substitute_inner(ctx, r, target, replacement, target_power, opts, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                node
            }
        }
        Expr::Sub(l, r) => {
            let nl = substitute_inner(ctx, l, target, replacement, target_power, opts, steps);
            let nr = substitute_inner(ctx, r, target, replacement, target_power, opts, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                node
            }
        }
        Expr::Mul(l, r) => {
            let nl = substitute_inner(ctx, l, target, replacement, target_power, opts, steps);
            let nr = substitute_inner(ctx, r, target, replacement, target_power, opts, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Mul(nl, nr))
            } else {
                node
            }
        }
        Expr::Div(l, r) => {
            let nl = substitute_inner(ctx, l, target, replacement, target_power, opts, steps);
            let nr = substitute_inner(ctx, r, target, replacement, target_power, opts, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                node
            }
        }
        Expr::Pow(b, e) => {
            let nb = substitute_inner(ctx, b, target, replacement, target_power, opts, steps);
            let ne = substitute_inner(ctx, e, target, replacement, target_power, opts, steps);
            if nb != b || ne != e {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                node
            }
        }
        Expr::Neg(inner) => {
            let ni = substitute_inner(ctx, inner, target, replacement, target_power, opts, steps);
            if ni != inner {
                ctx.add(Expr::Neg(ni))
            } else {
                node
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::with_capacity(args.len());
            let mut changed = false;
            for arg in args.iter() {
                let na =
                    substitute_inner(ctx, *arg, target, replacement, target_power, opts, steps);
                if na != *arg {
                    changed = true;
                }
                new_args.push(na);
            }
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                node
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let mut new_data = Vec::with_capacity(data.len());
            let mut changed = false;
            for elem in data.iter() {
                let ne =
                    substitute_inner(ctx, *elem, target, replacement, target_power, opts, steps);
                if ne != *elem {
                    changed = true;
                }
                new_data.push(ne);
            }
            if changed {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                node
            }
        }
        // Leaf nodes: Number, Variable, Constant, SessionRef - no children
        _ => node,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn parse_expr(ctx: &mut Context, s: &str) -> ExprId {
        parse(s, ctx).expect("parse failed")
    }

    /// Check if expression contains a variable
    fn contains_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
        match ctx.get(expr) {
            Expr::Variable(sym_id) => ctx.sym_name(*sym_id) == var,
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => contains_var(ctx, *l, var) || contains_var(ctx, *r, var),
            Expr::Neg(e) => contains_var(ctx, *e, var),
            Expr::Function(_, args) => args.iter().any(|a| contains_var(ctx, *a, var)),
            _ => false,
        }
    }

    #[test]
    fn test_power_substitute_exact_match() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^2 + 1");
        let target = parse_expr(&mut ctx, "x^2");
        let replacement = parse_expr(&mut ctx, "y");

        let result = substitute_power_aware(
            &mut ctx,
            expr,
            target,
            replacement,
            SubstituteOptions::default(),
        );

        // Result should contain y (replacement) and not contain x
        assert!(contains_var(&ctx, result, "y"), "Expected y in result");
        assert!(!contains_var(&ctx, result, "x"), "Expected no x in result");
    }

    #[test]
    fn test_power_substitute_divisible() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^4");
        let target = parse_expr(&mut ctx, "x^2");
        let replacement = parse_expr(&mut ctx, "y");

        let result = substitute_power_aware(
            &mut ctx,
            expr,
            target,
            replacement,
            SubstituteOptions::default(),
        );

        // x^4 = (x^2)^2 → y^2, should be Pow(y, 2)
        if let Expr::Pow(base, exp) = ctx.get(result) {
            assert!(
                matches!(ctx.get(*base), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == "y"),
                "Base should be y"
            );
            if let Expr::Number(n) = ctx.get(*exp) {
                assert_eq!(n.to_integer(), BigInt::from(2), "Exponent should be 2");
            } else {
                panic!("Exponent should be a number");
            }
        } else {
            panic!("Result should be Pow, got: {:?}", ctx.get(result));
        }
    }

    #[test]
    fn test_power_substitute_with_remainder() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^3");
        let target = parse_expr(&mut ctx, "x^2");
        let replacement = parse_expr(&mut ctx, "y");

        let result = substitute_power_aware(
            &mut ctx,
            expr,
            target,
            replacement,
            SubstituteOptions::default(),
        );

        // x^3 = x^2 * x → y * x, should be Mul(y, x)
        if let Expr::Mul(l, r) = ctx.get(result) {
            // One side should be y, the other should be x
            let has_y = matches!(ctx.get(*l), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == "y")
                || matches!(ctx.get(*r), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == "y");
            let has_x = matches!(ctx.get(*l), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == "x")
                || matches!(ctx.get(*r), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == "x");
            assert!(has_y && has_x, "Result should be y*x");
        } else {
            panic!("Result should be Mul, got: {:?}", ctx.get(result));
        }
    }

    #[test]
    fn test_power_substitute_no_match_different_base() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "z^4");
        let target = parse_expr(&mut ctx, "x^2");
        let replacement = parse_expr(&mut ctx, "y");

        let result = substitute_power_aware(
            &mut ctx,
            expr,
            target,
            replacement,
            SubstituteOptions::default(),
        );

        // z^4 should be unchanged (different base from target x^2)
        assert_eq!(result, expr, "z^4 should remain unchanged");
    }

    #[test]
    fn test_power_substitute_complex_expression() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^4 + x^2 + 1");
        let target = parse_expr(&mut ctx, "x^2");
        let replacement = parse_expr(&mut ctx, "y");

        let result = substitute_power_aware(
            &mut ctx,
            expr,
            target,
            replacement,
            SubstituteOptions::default(),
        );

        // x^4 + x^2 + 1 → y^2 + y + 1
        // Should contain y but not x
        assert!(contains_var(&ctx, result, "y"), "Expected y in result");
        assert!(!contains_var(&ctx, result, "x"), "Expected no x in result");
    }

    #[test]
    fn test_power_substitute_higher_power() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^6");
        let target = parse_expr(&mut ctx, "x^2");
        let replacement = parse_expr(&mut ctx, "y");

        let result = substitute_power_aware(
            &mut ctx,
            expr,
            target,
            replacement,
            SubstituteOptions::default(),
        );

        // x^6 = (x^2)^3 → y^3
        if let Expr::Pow(base, exp) = ctx.get(result) {
            assert!(
                matches!(ctx.get(*base), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == "y"),
                "Base should be y"
            );
            if let Expr::Number(n) = ctx.get(*exp) {
                assert_eq!(n.to_integer(), BigInt::from(3), "Exponent should be 3");
            } else {
                panic!("Exponent should be a number");
            }
        } else {
            panic!("Result should be Pow, got: {:?}", ctx.get(result));
        }
    }

    // ========== REGRESSION TESTS ==========

    /// Test 111: Power Pattern Matching
    /// Substitute y for x^2 in x^4 + x^2 + 1
    /// Expected: y^2 + y + 1
    /// The system must understand that x^4 = (x^2)^2, hence x^4 → y^2
    #[test]
    fn test_111_power_pattern_matching() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^4 + x^2 + 1");
        let target = parse_expr(&mut ctx, "x^2");
        let replacement = parse_expr(&mut ctx, "y");

        let result = substitute_power_aware(
            &mut ctx,
            expr,
            target,
            replacement,
            SubstituteOptions::default(),
        );

        // x^4 + x^2 + 1 → y^2 + y + 1
        // Should contain y but not x
        assert!(
            contains_var(&ctx, result, "y"),
            "Test 111: Expected y in result"
        );
        assert!(
            !contains_var(&ctx, result, "x"),
            "Test 111: Expected no x in result (all x^n should be substituted)"
        );
    }

    /// Test 112: Trigonometric Inverse Substitution (ADVANCED - ENGINE PRO)
    ///
    /// Substitute u for sin(x) in cos(x)^2.
    /// Expected: 1 - u^2 (using identity cos²x = 1 - sin²x).
    ///
    /// NOTE: This test is IGNORED because it requires the engine to know
    /// the Pythagorean identity and apply it before substitution.
    /// Current behavior: returns cos(x)^2 unchanged.
    #[test]
    #[ignore = "Engine Pro feature: requires automatic trig identity application"]
    fn test_112_trig_inverse_substitution() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "cos(x)^2");
        let target = parse_expr(&mut ctx, "sin(x)");
        let replacement = parse_expr(&mut ctx, "u");

        let result = substitute_power_aware(
            &mut ctx,
            expr,
            target,
            replacement,
            SubstituteOptions::default(),
        );

        // Expected: 1 - u^2 (if engine applies cos²x = 1 - sin²x identity first)
        // Current: cos(x)^2 unchanged
        assert!(
            contains_var(&ctx, result, "u"),
            "Test 112: Expected u in result (after identity application)"
        );
        assert!(
            !contains_var(&ctx, result, "x"),
            "Test 112: Expected no x in result"
        );
    }
}
