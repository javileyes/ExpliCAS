//! Difference of Cubes Simplification for Cube Root Expressions
//!
//! This module provides a specialized pre-order rule for simplifying quotients
//! that match the "difference of cubes" factorization pattern with cube roots.
//!
//! ## Pattern:
//! ```text
//! (x - b³) / (x^(2/3) + b·x^(1/3) + b²) → x^(1/3) - b
//! ```

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_math::difference_of_cubes_support::try_plan_cancel_cube_root_difference_expr;

fn canonicalize_nested_integer_powers(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let rebuilt = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Add(lhs, rhs))
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Sub(lhs, rhs))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Mul(lhs, rhs))
        }
        Expr::Div(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Div(lhs, rhs))
        }
        Expr::Pow(base, exp) => {
            let base = canonicalize_nested_integer_powers(ctx, base);
            let exp = canonicalize_nested_integer_powers(ctx, exp);
            let pow = ctx.add(Expr::Pow(base, exp));
            cas_math::rational_canonicalization_support::try_rewrite_nested_pow_canonical_expr(
                ctx, pow,
            )
            .map(|rewrite| rewrite.rewritten)
            .unwrap_or(pow)
        }
        Expr::Neg(inner) => {
            let inner = canonicalize_nested_integer_powers(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Function(name, args) => {
            let args = args
                .into_iter()
                .map(|arg| canonicalize_nested_integer_powers(ctx, arg))
                .collect();
            ctx.add(Expr::Function(name, args))
        }
        Expr::Matrix { rows, cols, data } => {
            let data = data
                .into_iter()
                .map(|arg| canonicalize_nested_integer_powers(ctx, arg))
                .collect();
            ctx.add(Expr::Matrix { rows, cols, data })
        }
        Expr::Hold(inner) => {
            let inner = canonicalize_nested_integer_powers(ctx, inner);
            ctx.add(Expr::Hold(inner))
        }
        Expr::SessionRef(id) => ctx.add(Expr::SessionRef(id)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => expr,
    };

    if rebuilt == expr {
        expr
    } else {
        rebuilt
    }
}

// CancelCubeRootDifferenceRule: Simplifies (x - b³) / (x^(2/3) + b·x^(1/3) + b²) → x^(1/3) - b
//
// This is a pre-order rule that catches the specific algebraic pattern before
// the general fraction simplification machinery can cause oscillation.
define_rule!(
    CancelCubeRootDifferenceRule,
    "Cancel Cube Root Difference",
    None,
    PhaseMask::CORE, // Run early in Core phase
    |ctx, expr| {
        if !matches!(ctx.get(expr), Expr::Div(_, _)) {
            return None;
        }
        let plan = try_plan_cancel_cube_root_difference_expr(ctx, expr)?;
        let b_squared = &plan.b_value * &plan.b_value;

        use crate::rule::ChainedRewrite;
        use crate::ImplicitCondition;

        let factor_rw = Rewrite::new(plan.intermediate)
            .desc(format!(
                "Factor difference of cubes: x - {} = (x^(1/3) - {})·(x^(2/3) + {}·x^(1/3) + {})",
                plan.cube_value, plan.b_value, plan.b_value, b_squared
            ))
            .local(plan.numerator, plan.factored_numerator)
            .requires(ImplicitCondition::NonZero(plan.denominator));

        let cancel = ChainedRewrite::new(plan.final_factor)
            .desc("Cancel common factor")
            .local(plan.intermediate, plan.final_factor);

        Some(factor_rw.chain(cancel))
    }
);

// CancelSumDiffCubesFractionRule: Simplifies (a^3 - b^3)/(a-b) and (a^3 + b^3)/(a+b)
// before other fraction rewrites deform the visible linear denominator.
define_rule!(
    CancelSumDiffCubesFractionRule,
    "Cancel Sum/Difference of Cubes Fraction",
    priority: 500,
    |ctx, expr, parent_ctx| {
        use crate::Predicate;

        let (num, den) = match ctx.get(expr) {
            Expr::Div(num, den) => (*num, *den),
            _ => return None,
        };

        let plan = crate::rules::algebra::fractions::try_plan_sum_diff_of_cubes_in_num(
            ctx, num, den, false,
        )?;

        let decision = crate::oracle_allows_with_hint(
            ctx,
            parent_ctx.domain_mode(),
            parent_ctx.value_domain(),
            &Predicate::NonZero(den),
            "Cancel Sum/Difference of Cubes Fraction",
        );
        if !decision.allow {
            return None;
        }

        Some(
            Rewrite::new(canonicalize_nested_integer_powers(ctx, plan.cancelled_result))
                .requires(crate::ImplicitCondition::NonZero(den))
                .desc(plan.desc),
        )
    }
);

/// Register the difference of cubes rules
pub fn register(simplifier: &mut crate::Simplifier) {
    // Register BEFORE general fraction simplification for pre-order behavior
    simplifier.add_rule(Box::new(CancelCubeRootDifferenceRule));
    simplifier.add_rule(Box::new(CancelSumDiffCubesFractionRule));
}

#[cfg(test)]
mod tests;
