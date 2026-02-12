//! Targeted simplifications for Weierstrass / inverse-trig bridge identities:
//!
//!   sin(2·arctan(t)) → 2t / (1+t²)
//!   cos(2·arctan(t)) → (1−t²) / (1+t²)
//!   tan(2·arctan(t)) → 2t / (1−t²)
//!
//! These bridge rules connect the trig and inverse-trig sub-worlds directly,
//! reducing compositions to rational expressions without needing expand_mode.
//!
//! **Domain safety**: Each identity introduces an explicit division, so
//! the rule is classified as `NeedsCondition(Definability)` and emits
//! `AssumptionEvent::nonzero(denominator)`.

use cas_ast::{BuiltinFn, Expr};

use crate::define_rule;
use crate::helpers::extract_double_angle_arg;
use crate::rule::Rewrite;
use crate::target_kind::TargetKindSet;

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(WeierstrassInverseAtanRule));
}

define_rule!(
    WeierstrassInverseAtanRule,
    "Weierstrass Inverse Atan Composition",
    Some(TargetKindSet::FUNCTION),
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, _parent_ctx| {
        // Match sin|cos|tan(arg0) with a single argument
        let (fn_id, arg0) = match ctx.get(expr) {
            Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
            _ => return None,
        };

        let trig = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)) => b,
            _ => return None,
        };

        // Match arg0 = 2 * inner  (double-angle form)
        let inner = extract_double_angle_arg(ctx, arg0)?;

        // Match inner = atan(t) or arctan(t)
        let t = match ctx.get(inner) {
            Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => {
                match ctx.builtin_of(*inv_id) {
                    Some(BuiltinFn::Atan | BuiltinFn::Arctan) => inv_args[0],
                    _ => return None,
                }
            }
            _ => return None,
        };

        // Build common sub-expressions: t², 1+t², 1-t², 2t
        let two = ctx.num(2);
        let one = ctx.num(1);
        let t_sq = ctx.add(Expr::Pow(t, two));

        let (new_expr, den, desc) = match trig {
            BuiltinFn::Sin => {
                // sin(2·atan(t)) = 2t / (1+t²)
                let num = ctx.add(Expr::Mul(two, t));
                let den = ctx.add(Expr::Add(one, t_sq));
                let result = ctx.add(Expr::Div(num, den));
                (result, den, "sin(2·atan(t)) = 2t/(1+t²)")
            }
            BuiltinFn::Cos => {
                // cos(2·atan(t)) = (1-t²) / (1+t²)
                let num = ctx.add(Expr::Sub(one, t_sq));
                let den = ctx.add(Expr::Add(one, t_sq));
                let result = ctx.add(Expr::Div(num, den));
                (result, den, "cos(2·atan(t)) = (1−t²)/(1+t²)")
            }
            BuiltinFn::Tan => {
                // tan(2·atan(t)) = 2t / (1-t²)
                let num = ctx.add(Expr::Mul(two, t));
                let den = ctx.add(Expr::Sub(one, t_sq));
                let result = ctx.add(Expr::Div(num, den));
                (result, den, "tan(2·atan(t)) = 2t/(1−t²)")
            }
            _ => return None,
        };

        Some(
            Rewrite::new(new_expr)
                .desc(desc)
                .assume(crate::assumptions::AssumptionEvent::nonzero(ctx, den)),
        )
    }
);
