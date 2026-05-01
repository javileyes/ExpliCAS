//! Support for `expand(...)` function-call rewrites.

use crate::expand_estimate::estimate_expand_terms;
use crate::expand_ops::expand;
use crate::poly_store::expand_expr_modp_materialized_hold;
use crate::polynomial::Polynomial;
use cas_ast::hold::strip_all_holds;
use cas_ast::ordering::compare_expr;
use cas_ast::{collect_variables, BuiltinFn, Context, Expr, ExprId};
use num_traits::{ToPrimitive, Zero};
use std::cmp::Ordering;

/// Default limit for materializing expanded AST terms.
pub const DEFAULT_EXPAND_MAX_MATERIALIZE_TERMS: u64 = 200_000;
/// Default threshold for switching to mod-p expansion.
pub const DEFAULT_EXPAND_MODP_THRESHOLD: u64 = 1_000;
const EXPLICIT_EXPAND_POLY_COMPACT_MAX_INPUT_NODES: usize = 160;
const EXPLICIT_EXPAND_POLY_COMPACT_MAX_DEGREE: usize = 16;
const EXPLICIT_EXPAND_POLY_COMPACT_MAX_TERMS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpandCallPolicy {
    pub max_materialize_terms: u64,
    pub modp_threshold: u64,
}

impl Default for ExpandCallPolicy {
    fn default() -> Self {
        Self {
            max_materialize_terms: DEFAULT_EXPAND_MAX_MATERIALIZE_TERMS,
            modp_threshold: DEFAULT_EXPAND_MODP_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpandCallRewrite {
    pub rewritten: ExprId,
    pub kind: ExpandCallRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandCallRewriteKind {
    ModpFastPath,
    Expand,
    ExpandAtom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandCallDecision {
    Rewrite(ExpandCallRewrite),
    LeaveUnevaluatedTooLarge { estimated_terms: u64, limit: u64 },
}

/// Plan implicit conservative expansion for non-`expand(...)` expressions.
///
/// Mirrors the conservative policy:
/// - expand structurally
/// - strip `__hold`
/// - accept only if complexity does not increase
/// - reject pure ordering no-op rewrites
pub fn try_plan_conservative_implicit_expand_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expanded_raw = expand(ctx, expr);
    let rewritten = strip_all_holds(ctx, expanded_raw);
    if rewritten == expr {
        return None;
    }

    let old_count = cas_ast::count_nodes(ctx, expr);
    let new_count = cas_ast::count_nodes(ctx, rewritten);
    if new_count > old_count {
        return None;
    }
    if compare_expr(ctx, rewritten, expr) == Ordering::Equal {
        return None;
    }

    Some(rewritten)
}

/// Expand an explicit `expand(arg)` payload and apply bounded post-compaction.
pub fn expand_explicit_arg_with_post_compaction(ctx: &mut Context, arg: ExprId) -> ExprId {
    let expanded = expand(ctx, arg);
    let stripped = strip_all_holds(ctx, expanded);
    compact_explicit_expand_polynomial(ctx, stripped)
}

/// Decide rewrite for function call `expand(arg)`.
///
/// Returns `None` when `expr` is not an `expand(...)` call.
pub fn decide_expand_call_rewrite_with_policy(
    ctx: &mut Context,
    expr: ExprId,
    policy: ExpandCallPolicy,
) -> Option<ExpandCallDecision> {
    let arg = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Expand)) && args.len() == 1 =>
        {
            args[0]
        }
        _ => return None,
    };

    let estimated_terms = estimate_expand_terms(ctx, arg);
    if let Some(est) = estimated_terms {
        if est > policy.max_materialize_terms {
            return Some(ExpandCallDecision::LeaveUnevaluatedTooLarge {
                estimated_terms: est,
                limit: policy.max_materialize_terms,
            });
        }
    }

    if estimated_terms.unwrap_or(0) > policy.modp_threshold {
        if let Some(result) = expand_expr_modp_materialized_hold(ctx, arg) {
            let stripped = strip_all_holds(ctx, result);
            let rewritten = compact_explicit_expand_polynomial(ctx, stripped);
            return Some(ExpandCallDecision::Rewrite(ExpandCallRewrite {
                rewritten,
                kind: ExpandCallRewriteKind::ModpFastPath,
            }));
        }
    }

    let rewritten = expand_explicit_arg_with_post_compaction(ctx, arg);
    if rewritten != expr {
        Some(ExpandCallDecision::Rewrite(ExpandCallRewrite {
            rewritten,
            kind: ExpandCallRewriteKind::Expand,
        }))
    } else {
        Some(ExpandCallDecision::Rewrite(ExpandCallRewrite {
            rewritten: arg,
            kind: ExpandCallRewriteKind::ExpandAtom,
        }))
    }
}

fn compact_explicit_expand_polynomial(ctx: &mut Context, expr: ExprId) -> ExprId {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return expr;
    }

    let old_nodes = cas_ast::count_nodes(ctx, expr);
    if old_nodes > EXPLICIT_EXPAND_POLY_COMPACT_MAX_INPUT_NODES {
        return expr;
    }
    if contains_integer_power_above_compact_limit(ctx, expr) {
        return expr;
    }

    let variables = collect_variables(ctx, expr);
    if variables.len() != 1 {
        return expr;
    }
    let Some(var) = variables.iter().next() else {
        return expr;
    };

    let Ok(poly) = Polynomial::from_expr(ctx, expr, var) else {
        return expr;
    };
    if poly.degree() > EXPLICIT_EXPAND_POLY_COMPACT_MAX_DEGREE {
        return expr;
    }
    let output_terms = poly.coeffs.iter().filter(|coeff| !coeff.is_zero()).count();
    if output_terms > EXPLICIT_EXPAND_POLY_COMPACT_MAX_TERMS {
        return expr;
    }

    let rewritten = poly.to_expr(ctx);
    if rewritten == expr {
        return expr;
    }

    let new_nodes = cas_ast::count_nodes(ctx, rewritten);
    if new_nodes >= old_nodes {
        return expr;
    }

    rewritten
}

fn contains_integer_power_above_compact_limit(ctx: &Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];

    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Pow(base, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() {
                        match n.to_integer().to_usize() {
                            Some(exp) if exp > EXPLICIT_EXPAND_POLY_COMPACT_MAX_DEGREE => {
                                return true;
                            }
                            None => return true,
                            _ => {}
                        }
                    }
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::{
        decide_expand_call_rewrite_with_policy, try_plan_conservative_implicit_expand_expr,
        ExpandCallDecision, ExpandCallPolicy, ExpandCallRewriteKind,
        DEFAULT_EXPAND_MAX_MATERIALIZE_TERMS, DEFAULT_EXPAND_MODP_THRESHOLD,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    fn rewrite_text(input: &str) -> (Context, cas_ast::ExprId) {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx).expect("parse");
        let decision = decide_expand_call_rewrite_with_policy(
            &mut ctx,
            expr,
            ExpandCallPolicy {
                max_materialize_terms: u64::MAX,
                modp_threshold: u64::MAX,
            },
        )
        .expect("decision");
        match decision {
            ExpandCallDecision::Rewrite(rewrite) => (ctx, rewrite.rewritten),
            other => panic!("expected rewrite, got {other:?}"),
        }
    }

    #[test]
    fn rewrites_expand_atom_to_inner() {
        let mut ctx = Context::new();
        let expr = parse("expand(x)", &mut ctx).expect("parse");
        let decision = decide_expand_call_rewrite_with_policy(
            &mut ctx,
            expr,
            ExpandCallPolicy {
                max_materialize_terms: u64::MAX,
                modp_threshold: u64::MAX,
            },
        )
        .expect("decision");
        match decision {
            ExpandCallDecision::Rewrite(rewrite) => {
                assert_eq!(rewrite.kind, ExpandCallRewriteKind::Expand);
                assert_eq!(cas_formatter::render_expr(&ctx, rewrite.rewritten), "x");
            }
            other => panic!("expected rewrite, got {other:?}"),
        }
    }

    #[test]
    fn reports_too_large_when_limit_exceeded() {
        let mut ctx = Context::new();
        let expr = parse("expand((x+1)^8)", &mut ctx).expect("parse");
        let decision = decide_expand_call_rewrite_with_policy(
            &mut ctx,
            expr,
            ExpandCallPolicy {
                max_materialize_terms: 1,
                modp_threshold: u64::MAX,
            },
        )
        .expect("decision");
        assert!(matches!(
            decision,
            ExpandCallDecision::LeaveUnevaluatedTooLarge { .. }
        ));
    }

    #[test]
    fn conservative_implicit_expand_accepts_non_worsening_rewrite() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let expr = cas_ast::hold::wrap_hold(&mut ctx, x);
        let rewritten = try_plan_conservative_implicit_expand_expr(&mut ctx, expr).expect("plan");
        assert_eq!(cas_formatter::render_expr(&ctx, rewritten), "x");
    }

    #[test]
    fn default_policy_constants_match_default_impl() {
        let default = ExpandCallPolicy::default();
        assert_eq!(
            default.max_materialize_terms,
            DEFAULT_EXPAND_MAX_MATERIALIZE_TERMS
        );
        assert_eq!(default.modp_threshold, DEFAULT_EXPAND_MODP_THRESHOLD);
    }

    #[test]
    fn explicit_default_and_conservative_policies_both_rewrite() {
        let mut ctx = Context::new();
        let expr = parse("expand((x+1)^8)", &mut ctx).expect("parse");
        let defaulted =
            decide_expand_call_rewrite_with_policy(&mut ctx, expr, ExpandCallPolicy::default())
                .expect("default");
        let conservative = decide_expand_call_rewrite_with_policy(
            &mut ctx,
            expr,
            ExpandCallPolicy {
                max_materialize_terms: u64::MAX,
                modp_threshold: u64::MAX,
            },
        )
        .expect("conservative");
        assert!(matches!(defaulted, ExpandCallDecision::Rewrite(_)));
        assert!(matches!(conservative, ExpandCallDecision::Rewrite(_)));
    }

    #[test]
    fn explicit_expand_compacts_univariate_polynomial_square_terms() {
        let (mut ctx, rewritten) = rewrite_text("expand((x^2+2*x+1)^2)");
        let expected = parse("x^4 + 4*x^3 + 6*x^2 + 4*x + 1", &mut ctx).expect("expected");
        assert!(crate::poly_compare::poly_eq(&ctx, rewritten, expected));
        assert_eq!(
            cas_formatter::render_expr(&ctx, rewritten),
            "x^4 + 4 * x^3 + 6 * x^2 + 4 * x + 1"
        );
    }

    #[test]
    fn explicit_expand_compacts_subtracted_univariate_polynomial_square() {
        let (mut ctx, rewritten) = rewrite_text("expand(3-(x^2+2*x+1)^2)");
        let expected = parse("2 - x^4 - 4*x^3 - 6*x^2 - 4*x", &mut ctx).expect("expected");
        assert!(crate::poly_compare::poly_eq(&ctx, rewritten, expected));
        assert_eq!(
            cas_formatter::render_expr(&ctx, rewritten),
            "2 - x^4 - 4 * x^3 - 6 * x^2 - 4 * x"
        );
    }

    #[test]
    fn explicit_expand_compaction_rejects_high_power_residuals() {
        let (ctx, rewritten) = rewrite_text("expand(3-(x^2+2*x+1)^1000)");
        assert_eq!(
            cas_formatter::render_expr(&ctx, rewritten),
            "3 - (x^2 + 2 * x + 1)^1000"
        );
    }
}
