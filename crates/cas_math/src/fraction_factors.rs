//! Structural factor helpers for fraction/rationalization workflows.
//!
//! These utilities operate on AST structure only (no polynomial expansion).

use crate::build::mul2_raw;
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorCancellationAction {
    Skip,
    RemoveBoth {
        nonzero_base: ExprId,
        emit_assumption: bool,
    },
    ReplaceNumeratorRemoveDenominator {
        new_numerator_factor: ExprId,
    },
    RemoveNumeratorReplaceDenominator {
        new_denominator_factor: ExprId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancelCommonFactorsGate {
    pub allow: bool,
    pub assumed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancelCommonFactorsRewrite {
    pub rewritten: ExprId,
    pub assumed_nonzero_targets: Vec<ExprId>,
}

/// Collect multiplicative factors by flattening only `Mul(...)` nodes.
///
/// This keeps factors as-is (e.g. `Pow(x, 2)` remains one factor) and is
/// useful when rules reason structurally about product terms.
pub fn collect_mul_factors_flat(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    collect_mul_factors_flat_recursive(ctx, expr, &mut factors);
    factors
}

fn collect_mul_factors_flat_recursive(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors_flat_recursive(ctx, *left, factors);
            collect_mul_factors_flat_recursive(ctx, *right, factors);
        }
        _ => factors.push(expr),
    }
}

/// Collect multiplicative factors with integer exponents from an expression.
///
/// Rules:
/// - `Mul(...)` is flattened
/// - `Pow(base, k)` with integer `k` becomes `(base, k)`
/// - top-level `Neg(x)` is unwrapped for intersection purposes
/// - everything else becomes `(expr, 1)`
pub fn collect_mul_factors_int_pow(ctx: &Context, expr: ExprId) -> Vec<(ExprId, i64)> {
    let mut factors = Vec::new();
    let actual_expr = match ctx.get(expr) {
        Expr::Neg(inner) => *inner,
        _ => expr,
    };
    collect_mul_factors_recursive(ctx, actual_expr, 1, &mut factors);
    factors
}

fn collect_mul_factors_recursive(
    ctx: &Context,
    expr: ExprId,
    mult: i64,
    factors: &mut Vec<(ExprId, i64)>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors_recursive(ctx, *left, mult, factors);
            collect_mul_factors_recursive(ctx, *right, mult, factors);
        }
        Expr::Pow(base, exp) => {
            if let Some(k) = integer_exponent(ctx, *exp) {
                factors.push((*base, mult * k));
            } else {
                factors.push((expr, mult));
            }
        }
        _ => factors.push((expr, mult)),
    }
}

fn integer_exponent(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => integer_exponent(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

/// Build a product from factors with integer exponents.
///
/// Negative exponents are ignored (the caller typically manages denominator
/// factors separately).
pub fn build_mul_from_factors_int_pow(ctx: &mut Context, factors: &[(ExprId, i64)]) -> ExprId {
    use cas_ast::views::MulBuilder;

    let mut builder = MulBuilder::new_simple();
    for &(base, exp) in factors {
        if exp > 0 {
            builder.push_pow(base, exp);
        }
    }
    builder.build(ctx)
}

/// Structural decomposition of fraction-like expressions into factor vectors.
///
/// Supported shapes:
/// - `num / den`
/// - `den^(-1)` treated as `1/den`
/// - products containing reciprocal factors: `a * b * den^(-1)`
///
/// Returns `None` when the input is not fraction-like.
pub fn decompose_fraction_like_factors(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    let is_neg_one = |id: ExprId| -> bool {
        if let Expr::Number(n) = ctx.get(id) {
            n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into())
        } else {
            false
        }
    };

    match ctx.get(expr).clone() {
        Expr::Div(n, d) => Some((
            collect_mul_factors_flat(ctx, n),
            collect_mul_factors_flat(ctx, d),
        )),
        Expr::Pow(b, e) if is_neg_one(e) => {
            Some((vec![ctx.num(1)], collect_mul_factors_flat(ctx, b)))
        }
        Expr::Mul(_, _) => {
            let factors = collect_mul_factors_flat(ctx, expr);
            let mut num_factors = Vec::new();
            let mut den_factors = Vec::new();
            for f in factors {
                if let Expr::Pow(b, e) = ctx.get(f) {
                    if is_neg_one(*e) {
                        den_factors.extend(collect_mul_factors_flat(ctx, *b));
                        continue;
                    }
                }
                num_factors.push(f);
            }
            if den_factors.is_empty() {
                None
            } else {
                Some((num_factors, den_factors))
            }
        }
        _ => None,
    }
}

/// Build a fraction expression from numerator and denominator factor vectors.
pub fn build_fraction_from_factor_vectors(
    ctx: &mut Context,
    num_factors: &[ExprId],
    den_factors: &[ExprId],
) -> ExprId {
    let new_num = if num_factors.is_empty() {
        ctx.num(1)
    } else {
        let mut n = num_factors[0];
        for &f in num_factors.iter().skip(1) {
            n = mul2_raw(ctx, n, f);
        }
        n
    };

    let new_den = if den_factors.is_empty() {
        ctx.num(1)
    } else {
        let mut d = den_factors[0];
        for &f in den_factors.iter().skip(1) {
            d = mul2_raw(ctx, d, f);
        }
        d
    };

    ctx.add(Expr::Div(new_num, new_den))
}

/// Classify structural cancellation action for a single numerator/denominator factor pair.
///
/// This performs pure shape/exponent analysis and does not apply domain policy.
pub fn classify_factor_cancellation_action(
    ctx: &mut Context,
    num_factor: ExprId,
    den_factor: ExprId,
) -> FactorCancellationAction {
    use cas_ast::ordering::compare_expr;
    use num_traits::{One, Zero};

    if compare_expr(ctx, num_factor, den_factor) == std::cmp::Ordering::Equal {
        return FactorCancellationAction::RemoveBoth {
            nonzero_base: num_factor,
            emit_assumption: true,
        };
    }

    let num_pow = if let Expr::Pow(base, exp) = ctx.get(num_factor) {
        Some((*base, *exp))
    } else {
        None
    };
    let den_pow = if let Expr::Pow(base, exp) = ctx.get(den_factor) {
        Some((*base, *exp))
    } else {
        None
    };

    // Case 1: num = base^n, den = base
    if let Some((base, exp)) = num_pow {
        if compare_expr(ctx, base, den_factor) == std::cmp::Ordering::Equal {
            if let Expr::Number(n) = ctx.get(exp) {
                if !n.is_integer() {
                    return FactorCancellationAction::Skip;
                }
                let new_exp = n - num_rational::BigRational::one();
                if new_exp.is_zero() {
                    return FactorCancellationAction::RemoveBoth {
                        nonzero_base: base,
                        emit_assumption: true,
                    };
                }
                let new_numerator_factor = if new_exp.is_one() {
                    base
                } else {
                    let exp_node = ctx.add(Expr::Number(new_exp));
                    ctx.add(Expr::Pow(base, exp_node))
                };
                return FactorCancellationAction::ReplaceNumeratorRemoveDenominator {
                    new_numerator_factor,
                };
            }
        }
    }

    // Case 2: num = base, den = base^m
    if let Some((base, exp)) = den_pow {
        if compare_expr(ctx, num_factor, base) == std::cmp::Ordering::Equal {
            if let Expr::Number(m) = ctx.get(exp) {
                if !m.is_integer() {
                    return FactorCancellationAction::Skip;
                }
                let new_exp = m - num_rational::BigRational::one();
                if new_exp.is_zero() {
                    return FactorCancellationAction::RemoveBoth {
                        nonzero_base: base,
                        // Preserve existing behavior in engine:
                        // this specific path does not emit assumption event.
                        emit_assumption: false,
                    };
                }
                let new_denominator_factor = if new_exp.is_one() {
                    base
                } else {
                    let exp_node = ctx.add(Expr::Number(new_exp));
                    ctx.add(Expr::Pow(base, exp_node))
                };
                return FactorCancellationAction::RemoveNumeratorReplaceDenominator {
                    new_denominator_factor,
                };
            }
        }
    }

    // Case 3: num = base^n, den = base^m
    if let (Some((num_base, num_exp)), Some((den_base, den_exp))) = (num_pow, den_pow) {
        if compare_expr(ctx, num_base, den_base) == std::cmp::Ordering::Equal {
            if let (Expr::Number(n), Expr::Number(m)) = (ctx.get(num_exp), ctx.get(den_exp)) {
                if !n.is_integer() || !m.is_integer() {
                    return FactorCancellationAction::Skip;
                }
                if n > m {
                    let new_exp = n - m;
                    let new_numerator_factor = if new_exp.is_one() {
                        num_base
                    } else {
                        let exp_node = ctx.add(Expr::Number(new_exp));
                        ctx.add(Expr::Pow(num_base, exp_node))
                    };
                    return FactorCancellationAction::ReplaceNumeratorRemoveDenominator {
                        new_numerator_factor,
                    };
                }
                if m > n {
                    let new_exp = m - n;
                    let new_denominator_factor = if new_exp.is_one() {
                        den_base
                    } else {
                        let exp_node = ctx.add(Expr::Number(new_exp));
                        ctx.add(Expr::Pow(den_base, exp_node))
                    };
                    return FactorCancellationAction::RemoveNumeratorReplaceDenominator {
                        new_denominator_factor,
                    };
                }
                return FactorCancellationAction::RemoveBoth {
                    nonzero_base: num_base,
                    emit_assumption: true,
                };
            }
        }
    }

    FactorCancellationAction::Skip
}

/// Cancel common factors in numerator/denominator factor vectors.
///
/// Domain policy is injected via callback for the `RemoveBoth` cases that may
/// require `nonzero_base != 0` conditions.
pub fn cancel_common_factor_vectors_with<FAllowRemoveBoth>(
    ctx: &mut Context,
    num_factors: &mut Vec<ExprId>,
    den_factors: &mut Vec<ExprId>,
    mut allow_remove_both: FAllowRemoveBoth,
) -> bool
where
    FAllowRemoveBoth: FnMut(&mut Context, ExprId, bool) -> bool,
{
    let mut changed = false;
    let mut i = 0usize;
    while i < num_factors.len() {
        let nf = num_factors[i];
        let mut remove_num_factor = false;
        for j in 0..den_factors.len() {
            let df = den_factors[j];
            match classify_factor_cancellation_action(ctx, nf, df) {
                FactorCancellationAction::Skip => {}
                FactorCancellationAction::ReplaceNumeratorRemoveDenominator {
                    new_numerator_factor,
                } => {
                    num_factors[i] = new_numerator_factor;
                    den_factors.remove(j);
                    changed = true;
                    break;
                }
                FactorCancellationAction::RemoveNumeratorReplaceDenominator {
                    new_denominator_factor,
                } => {
                    den_factors[j] = new_denominator_factor;
                    remove_num_factor = true;
                    changed = true;
                    break;
                }
                FactorCancellationAction::RemoveBoth {
                    nonzero_base,
                    emit_assumption,
                } => {
                    if !allow_remove_both(ctx, nonzero_base, emit_assumption) {
                        continue;
                    }
                    den_factors.remove(j);
                    remove_num_factor = true;
                    changed = true;
                    break;
                }
            }
        }
        if remove_num_factor {
            num_factors.remove(i);
        } else {
            i += 1;
        }
    }
    changed
}

/// Rewrite fraction-like expressions by canceling common factors under a caller-provided
/// nonzero-policy callback.
pub fn try_rewrite_cancel_common_factors_expr_with<FGate>(
    ctx: &mut Context,
    expr: ExprId,
    mut gate: FGate,
) -> Option<CancelCommonFactorsRewrite>
where
    FGate: FnMut(&mut Context, ExprId, bool) -> CancelCommonFactorsGate,
{
    let (mut num_factors, mut den_factors) = decompose_fraction_like_factors(ctx, expr)?;
    let mut assumed_nonzero_targets = Vec::new();
    let changed = cancel_common_factor_vectors_with(
        ctx,
        &mut num_factors,
        &mut den_factors,
        |ctx, nonzero_base, emit_assumption| {
            let decision = gate(ctx, nonzero_base, emit_assumption);
            if decision.allow && emit_assumption && decision.assumed {
                assumed_nonzero_targets.push(nonzero_base);
            }
            decision.allow
        },
    );

    if !changed {
        return None;
    }

    let rewritten = build_fraction_from_factor_vectors(ctx, &num_factors, &den_factors);
    Some(CancelCommonFactorsRewrite {
        rewritten,
        assumed_nonzero_targets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly_compare::poly_eq;
    use cas_parser::parse;

    #[test]
    fn collect_flat_mul_only() {
        let mut ctx = Context::new();
        let expr = parse("(a*b)*(c^2)", &mut ctx).expect("parse");
        let factors = collect_mul_factors_flat(&ctx, expr);

        assert_eq!(factors.len(), 3);
        let expected_a = parse("a", &mut ctx).expect("parse a");
        let expected_b = parse("b", &mut ctx).expect("parse b");
        let expected_c2 = parse("c^2", &mut ctx).expect("parse c2");
        assert!(factors
            .iter()
            .any(|&f| crate::poly_compare::poly_eq(&ctx, f, expected_a)));
        assert!(factors
            .iter()
            .any(|&f| crate::poly_compare::poly_eq(&ctx, f, expected_b)));
        assert!(factors
            .iter()
            .any(|&f| crate::poly_compare::poly_eq(&ctx, f, expected_c2)));
    }

    #[test]
    fn collect_strips_top_level_neg() {
        let mut ctx = Context::new();
        let expr = parse("-(x*y^2)", &mut ctx).expect("parse");
        let factors = collect_mul_factors_int_pow(&ctx, expr);

        assert_eq!(factors.len(), 2);
        let mut exponents: Vec<i64> = factors.iter().map(|(_, e)| *e).collect();
        exponents.sort_unstable();
        assert_eq!(exponents, vec![1, 2]);
    }

    #[test]
    fn collect_recognizes_negative_integer_exponent() {
        let mut ctx = Context::new();
        let expr = parse("x^(-3)", &mut ctx).expect("parse");
        let factors = collect_mul_factors_int_pow(&ctx, expr);
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, -3);
    }

    #[test]
    fn build_product_from_factors() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse x");
        let y = parse("y", &mut ctx).expect("parse y");
        let factors = vec![(x, 2), (y, 1), (x, -1)];

        let built = build_mul_from_factors_int_pow(&mut ctx, &factors);
        let expected = parse("x^2*y", &mut ctx).expect("parse expected");
        assert!(poly_eq(&ctx, built, expected));
    }

    #[test]
    fn decompose_fraction_like_div_shape() {
        let mut ctx = Context::new();
        let expr = parse("(a*b)/(c*d)", &mut ctx).expect("parse");
        let (num, den) = decompose_fraction_like_factors(&mut ctx, expr).expect("decompose");
        assert_eq!(num.len(), 2);
        assert_eq!(den.len(), 2);
    }

    #[test]
    fn decompose_fraction_like_pow_neg_one_shape() {
        let mut ctx = Context::new();
        let expr = parse("(x*y)^(-1)", &mut ctx).expect("parse");
        let (num, den) = decompose_fraction_like_factors(&mut ctx, expr).expect("decompose");
        assert_eq!(num.len(), 1);
        assert_eq!(den.len(), 2);
    }

    #[test]
    fn decompose_fraction_like_mul_with_reciprocal_shape() {
        let mut ctx = Context::new();
        let expr = parse("a*b*c^(-1)", &mut ctx).expect("parse");
        let (num, den) = decompose_fraction_like_factors(&mut ctx, expr).expect("decompose");
        assert_eq!(num.len(), 2);
        assert_eq!(den.len(), 1);
    }

    #[test]
    fn decompose_fraction_like_rejects_non_fraction() {
        let mut ctx = Context::new();
        let expr = parse("a*b*c", &mut ctx).expect("parse");
        assert!(decompose_fraction_like_factors(&mut ctx, expr).is_none());
    }

    #[test]
    fn build_fraction_from_factor_vectors_roundtrip() {
        let mut ctx = Context::new();
        let expr = parse("(a*b)/(c*d)", &mut ctx).expect("parse");
        let (num, den) = decompose_fraction_like_factors(&mut ctx, expr).expect("decompose");
        let rebuilt = build_fraction_from_factor_vectors(&mut ctx, &num, &den);
        let (num2, den2) =
            decompose_fraction_like_factors(&mut ctx, rebuilt).expect("decompose rebuilt");
        assert_eq!(num2.len(), num.len());
        assert_eq!(den2.len(), den.len());
    }

    #[test]
    fn classify_cancellation_exact_match_removes_both() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let action = classify_factor_cancellation_action(&mut ctx, x, x);
        assert!(matches!(
            action,
            FactorCancellationAction::RemoveBoth {
                nonzero_base: _,
                emit_assumption: true
            }
        ));
    }

    #[test]
    fn classify_cancellation_pow_vs_base_reduces_numerator_exponent() {
        let mut ctx = Context::new();
        let num = parse("x^3", &mut ctx).expect("parse");
        let den = parse("x", &mut ctx).expect("parse");
        let action = classify_factor_cancellation_action(&mut ctx, num, den);
        match action {
            FactorCancellationAction::ReplaceNumeratorRemoveDenominator {
                new_numerator_factor,
            } => {
                let expected = parse("x^2", &mut ctx).expect("parse expected");
                assert!(poly_eq(&ctx, new_numerator_factor, expected));
            }
            _ => panic!("unexpected action: {action:?}"),
        }
    }

    #[test]
    fn classify_cancellation_base_vs_pow_one_removes_both_without_assumption_event() {
        let mut ctx = Context::new();
        let num = parse("x", &mut ctx).expect("parse");
        let den = parse("x^1", &mut ctx).expect("parse");
        let action = classify_factor_cancellation_action(&mut ctx, num, den);
        assert!(matches!(
            action,
            FactorCancellationAction::RemoveBoth {
                nonzero_base: _,
                emit_assumption: false
            }
        ));
    }

    #[test]
    fn classify_cancellation_fractional_exponents_skip() {
        let mut ctx = Context::new();
        let num = parse("x^(1/2)", &mut ctx).expect("parse");
        let den = parse("x", &mut ctx).expect("parse");
        let action = classify_factor_cancellation_action(&mut ctx, num, den);
        assert_eq!(action, FactorCancellationAction::Skip);
    }

    #[test]
    fn cancel_common_factor_vectors_with_cancels_basic_factor() {
        let mut ctx = Context::new();
        let expr = parse("(x*y)/x", &mut ctx).expect("parse");
        let (mut num, mut den) =
            decompose_fraction_like_factors(&mut ctx, expr).expect("decompose");
        let changed = cancel_common_factor_vectors_with(
            &mut ctx,
            &mut num,
            &mut den,
            |_ctx, _base, _emit_assumption| true,
        );
        assert!(changed);
        let rebuilt = build_fraction_from_factor_vectors(&mut ctx, &num, &den);
        let expected = parse("y", &mut ctx).expect("parse expected");
        assert!(poly_eq(&ctx, rebuilt, expected));
    }

    #[test]
    fn cancel_common_factor_vectors_with_respects_removeboth_guard() {
        let mut ctx = Context::new();
        let expr = parse("x/x", &mut ctx).expect("parse");
        let (mut num, mut den) =
            decompose_fraction_like_factors(&mut ctx, expr).expect("decompose");
        let num_before = num.clone();
        let den_before = den.clone();
        let changed = cancel_common_factor_vectors_with(
            &mut ctx,
            &mut num,
            &mut den,
            |_ctx, _base, _emit_assumption| false,
        );
        assert!(!changed);
        assert_eq!(num, num_before);
        assert_eq!(den, den_before);
    }

    #[test]
    fn rewrite_cancel_common_factors_with_allows_basic_cancel() {
        let mut ctx = Context::new();
        let expr = parse("(x*y)/x", &mut ctx).expect("parse");
        let rewrite = try_rewrite_cancel_common_factors_expr_with(
            &mut ctx,
            expr,
            |_ctx, _base, emit_assumption| CancelCommonFactorsGate {
                allow: true,
                assumed: emit_assumption,
            },
        )
        .expect("rewrite");
        let expected = parse("y", &mut ctx).expect("parse expected");
        assert!(poly_eq(&ctx, rewrite.rewritten, expected));
        assert_eq!(rewrite.assumed_nonzero_targets.len(), 1);
    }

    #[test]
    fn rewrite_cancel_common_factors_with_blocks_cancel() {
        let mut ctx = Context::new();
        let expr = parse("x/x", &mut ctx).expect("parse");
        let rewrite = try_rewrite_cancel_common_factors_expr_with(
            &mut ctx,
            expr,
            |_ctx, _base, _emit_assumption| CancelCommonFactorsGate {
                allow: false,
                assumed: false,
            },
        );
        assert!(rewrite.is_none());
    }
}
