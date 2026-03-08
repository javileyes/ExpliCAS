//! Pre-normalization helpers for equation-level additive cancellation.
//!
//! This module provides AST-local rewrites used to expose additive structure
//! before semantic cancellation checks.

use crate::cancel_support::{
    collect_additive_terms_signed as collect_additive_terms,
    rebuild_from_signed_terms as rebuild_from_terms,
};
use cas_ast::{Context, Expr, ExprId};

/// Safety classification for normalizer output.
///
/// Determines whether structurally equal terms can be cancelled without a
/// strict proof (definability-preserving), or require analytic conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OriginSafety {
    /// SplitDiv, DistributeMul, Add/Sub/Neg passthrough.
    /// These do not introduce new domain requirements.
    DefinabilityPreserving,
    /// LnOfMul, LnOfPow, LnOfSqrt.
    /// These require analytic conditions (e.g., positivity for ln).
    NeedsAnalyticConditions,
}

impl OriginSafety {
    /// Combine two safety classifications: worst wins.
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (OriginSafety::NeedsAnalyticConditions, _)
            | (_, OriginSafety::NeedsAnalyticConditions) => OriginSafety::NeedsAnalyticConditions,
            _ => OriginSafety::DefinabilityPreserving,
        }
    }
}

/// Pre-normalize expression for cancel: split fraction numerators and log
/// products to expose hidden additive terms.
///
/// Applied only within semantic cancellation pipeline, not globally.
/// Returns `(normalized_expr, safety)` where safety indicates whether
/// normalization is definability-preserving or needs analytic conditions.
///
/// Guard: max recursion depth 3, max 6 numerator/additive terms.
pub fn normalize_for_cancel(ctx: &mut Context, id: ExprId, depth: usize) -> (ExprId, OriginSafety) {
    if depth > 3 {
        return (id, OriginSafety::DefinabilityPreserving);
    }

    enum Action {
        Pass,
        Add(ExprId, ExprId),
        Sub(ExprId, ExprId),
        Neg(ExprId),
        SplitDiv(ExprId, ExprId),       // (numerator, denominator)
        LnOfMul(usize, ExprId, ExprId), // (ln_name, a, b)
        LnOfPow(usize, ExprId, ExprId), // (ln_name, base, exp)
        LnOfSqrt(usize, ExprId),        // (ln_name, sqrt_arg): ln(sqrt(x)) -> 1/2*ln(x)
        DistributeMul(ExprId, ExprId),  // (scalar, additive): k*(a+b) -> k*a + k*b
        ExpandCos2x(ExprId),            // inner arg a: cos(2a) -> 1 - 2*sin^2(a)
        ExpandSin2x(ExprId),            // inner arg a: sin(2a) -> 2*sin(a)*cos(a)
    }

    let action = match ctx.get(id) {
        Expr::Add(l, r) => Action::Add(*l, *r),
        Expr::Sub(l, r) => Action::Sub(*l, *r),
        Expr::Neg(inner) => Action::Neg(*inner),
        Expr::Div(num, den) => {
            let has_add = matches!(ctx.get(*num), Expr::Add(_, _) | Expr::Sub(_, _));
            if has_add {
                Action::SplitDiv(*num, *den)
            } else {
                Action::Pass
            }
        }
        Expr::Function(name, args) if args.len() == 1 => {
            let name_id = *name;
            let arg = args[0];
            let builtin = ctx.builtin_of(name_id);
            let is_ln = builtin.map(|b| b.name() == "ln").unwrap_or(false);
            if is_ln {
                match ctx.get(arg) {
                    Expr::Mul(a, b) => Action::LnOfMul(name_id, *a, *b),
                    Expr::Pow(base, exp) => Action::LnOfPow(name_id, *base, *exp),
                    Expr::Function(inner_name, inner_args) if inner_args.len() == 1 => {
                        let is_sqrt = ctx
                            .builtin_of(*inner_name)
                            .map(|b| b.name() == "sqrt")
                            .unwrap_or(false);
                        if is_sqrt {
                            Action::LnOfSqrt(name_id, inner_args[0])
                        } else {
                            Action::Pass
                        }
                    }
                    _ => Action::Pass,
                }
            } else {
                let is_cos = matches!(builtin, Some(cas_ast::BuiltinFn::Cos));
                let is_sin = matches!(builtin, Some(cas_ast::BuiltinFn::Sin));
                if is_cos || is_sin {
                    if let Some((true, inner)) =
                        crate::trig_roots_flatten::extract_int_multiple_additive(ctx, arg, 2)
                    {
                        if is_cos {
                            Action::ExpandCos2x(inner)
                        } else {
                            Action::ExpandSin2x(inner)
                        }
                    } else {
                        Action::Pass
                    }
                } else {
                    Action::Pass
                }
            }
        }
        Expr::Mul(a, b) => {
            let a_add = matches!(ctx.get(*a), Expr::Add(_, _) | Expr::Sub(_, _));
            let b_add = matches!(ctx.get(*b), Expr::Add(_, _) | Expr::Sub(_, _));
            if a_add && !b_add {
                Action::DistributeMul(*b, *a)
            } else if b_add && !a_add {
                Action::DistributeMul(*a, *b)
            } else {
                Action::Pass
            }
        }
        _ => Action::Pass,
    };

    match action {
        Action::Pass => (id, OriginSafety::DefinabilityPreserving),
        Action::Add(l, r) => {
            let (nl, sl) = normalize_for_cancel(ctx, l, depth);
            let (nr, sr) = normalize_for_cancel(ctx, r, depth);
            let safety = sl.merge(sr);
            if nl == l && nr == r {
                (id, safety)
            } else {
                (ctx.add(Expr::Add(nl, nr)), safety)
            }
        }
        Action::Sub(l, r) => {
            let (nl, sl) = normalize_for_cancel(ctx, l, depth);
            let (nr, sr) = normalize_for_cancel(ctx, r, depth);
            let safety = sl.merge(sr);
            if nl == l && nr == r {
                (id, safety)
            } else {
                (ctx.add(Expr::Sub(nl, nr)), safety)
            }
        }
        Action::Neg(inner) => {
            let (ni, si) = normalize_for_cancel(ctx, inner, depth);
            if ni == inner {
                (id, si)
            } else {
                (ctx.add(Expr::Neg(ni)), si)
            }
        }
        Action::SplitDiv(num, den) => {
            let mut num_terms = Vec::new();
            collect_additive_terms(ctx, num, true, &mut num_terms);
            if num_terms.len() <= 1 || num_terms.len() > 6 {
                return (id, OriginSafety::DefinabilityPreserving);
            }
            let split: Vec<(ExprId, bool)> = num_terms
                .iter()
                .map(|(t, p)| (ctx.add(Expr::Div(*t, den)), *p))
                .collect();
            (
                rebuild_from_terms(ctx, &split),
                OriginSafety::DefinabilityPreserving,
            )
        }
        Action::LnOfMul(name, a, b) => {
            let ln_a = ctx.add(Expr::Function(name, vec![a]));
            let ln_b = ctx.add(Expr::Function(name, vec![b]));
            let (nla, _) = normalize_for_cancel(ctx, ln_a, depth + 1);
            let (nlb, _) = normalize_for_cancel(ctx, ln_b, depth + 1);
            (
                ctx.add(Expr::Add(nla, nlb)),
                OriginSafety::NeedsAnalyticConditions,
            )
        }
        Action::LnOfPow(name, base, exp) => {
            let ln_base = ctx.add(Expr::Function(name, vec![base]));
            (
                ctx.add(Expr::Mul(exp, ln_base)),
                OriginSafety::NeedsAnalyticConditions,
            )
        }
        Action::LnOfSqrt(name, sqrt_arg) => {
            let half = ctx.add(Expr::Number(num_rational::BigRational::new(
                num_bigint::BigInt::from(1),
                num_bigint::BigInt::from(2),
            )));
            let ln_arg = ctx.add(Expr::Function(name, vec![sqrt_arg]));
            (
                ctx.add(Expr::Mul(half, ln_arg)),
                OriginSafety::NeedsAnalyticConditions,
            )
        }
        Action::DistributeMul(scalar, additive) => {
            let mut terms = Vec::new();
            collect_additive_terms(ctx, additive, true, &mut terms);
            if terms.len() <= 1 || terms.len() > 6 {
                return (id, OriginSafety::DefinabilityPreserving);
            }
            let split: Vec<(ExprId, bool)> = terms
                .iter()
                .map(|(t, p)| (ctx.add(Expr::Mul(scalar, *t)), *p))
                .collect();
            (
                rebuild_from_terms(ctx, &split),
                OriginSafety::DefinabilityPreserving,
            )
        }
        Action::ExpandCos2x(inner) => {
            let sin_a = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![inner]);
            let two = ctx.num(2);
            let sin_sq = ctx.add(Expr::Pow(sin_a, two));
            let two_sin_sq = ctx.add(Expr::Mul(two, sin_sq));
            let one = ctx.num(1);
            let result = ctx.add(Expr::Sub(one, two_sin_sq));
            let (nr, ns) = normalize_for_cancel(ctx, result, depth + 1);
            (nr, ns)
        }
        Action::ExpandSin2x(inner) => {
            let sin_a = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![inner]);
            let cos_a = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![inner]);
            let prod = ctx.add(Expr::Mul(sin_a, cos_a));
            let two = ctx.num(2);
            let result = ctx.add(Expr::Mul(two, prod));
            (result, OriginSafety::DefinabilityPreserving)
        }
    }
}

/// Re-normalize already collected signed terms.
///
/// Each term is normalized with `normalize_for_cancel`, safety flags are merged,
/// and if normalization expands into additive structure the term is re-split.
pub fn renormalize_signed_terms_for_cancel(
    ctx: &mut Context,
    terms: &[(ExprId, bool, OriginSafety)],
) -> Vec<(ExprId, bool, OriginSafety)> {
    let mut out = Vec::new();
    for &(t, p, s) in terms {
        let (n, ns) = normalize_for_cancel(ctx, t, 0);
        let merged = s.merge(ns);
        if n == t {
            out.push((t, p, merged));
        } else {
            let mut raw = Vec::new();
            collect_additive_terms(ctx, n, p, &mut raw);
            for (rt, rp) in raw {
                out.push((rt, rp, merged));
            }
        }
    }
    out
}

/// Re-flatten signed terms after local simplification.
///
/// If a term became additive (e.g. `a + b`) it is split into multiple signed
/// terms while preserving outer sign and safety metadata.
pub fn reflatten_signed_terms_for_cancel(
    ctx: &Context,
    terms: Vec<(ExprId, bool, OriginSafety)>,
) -> Vec<(ExprId, bool, OriginSafety)> {
    let mut out = Vec::new();
    for (t, p, s) in terms {
        let mut raw = Vec::new();
        collect_additive_terms(ctx, t, p, &mut raw);
        for (rt, rp) in raw {
            out.push((rt, rp, s));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::ordering::compare_expr;
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn normalize_split_div_sum() {
        let mut ctx = Context::new();
        let expr = parse("(a + b)/d", &mut ctx).expect("parse");
        let (normalized, safety) = normalize_for_cancel(&mut ctx, expr, 0);
        let expected = parse("a/d + b/d", &mut ctx).expect("parse expected");
        assert_eq!(safety, OriginSafety::DefinabilityPreserving);
        assert_eq!(compare_expr(&ctx, normalized, expected), Ordering::Equal);
    }

    #[test]
    fn normalize_ln_product_marks_analytic_conditions() {
        let mut ctx = Context::new();
        let expr = parse("ln(a*b)", &mut ctx).expect("parse");
        let (normalized, safety) = normalize_for_cancel(&mut ctx, expr, 0);
        let expected = parse("ln(a) + ln(b)", &mut ctx).expect("parse expected");
        assert_eq!(safety, OriginSafety::NeedsAnalyticConditions);
        assert_eq!(compare_expr(&ctx, normalized, expected), Ordering::Equal);
    }

    #[test]
    fn renormalize_splits_additive_result() {
        let mut ctx = Context::new();
        let term = parse("(a+b)/d", &mut ctx).expect("parse");
        let terms = vec![(term, true, OriginSafety::DefinabilityPreserving)];
        let out = renormalize_signed_terms_for_cancel(&mut ctx, &terms);
        assert_eq!(out.len(), 2);
        assert!(out
            .iter()
            .all(|(_, _, s)| *s == OriginSafety::DefinabilityPreserving));
    }

    #[test]
    fn reflatten_preserves_safety() {
        let mut ctx = Context::new();
        let term = parse("a+b", &mut ctx).expect("parse");
        let terms = vec![(term, true, OriginSafety::NeedsAnalyticConditions)];
        let out = reflatten_signed_terms_for_cancel(&ctx, terms);
        assert_eq!(out.len(), 2);
        assert!(out
            .iter()
            .all(|(_, _, s)| *s == OriginSafety::NeedsAnalyticConditions));
    }
}
