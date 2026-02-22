//! Shared strategy kernels used by engine-side solver orchestration.
//!
//! These kernels keep equation applicability checks and core rewrites in
//! `cas_solver_core`, while `cas_engine` remains responsible for recursive
//! orchestration and context-aware simplification.

use crate::isolation_utils::contains_var;
use crate::solve_outcome::{eliminate_fractional_exponent_message, subtract_both_sides_message};
use cas_ast::{Context, Equation, ExprId, RelOp};

/// Didactic payload for strategy-level rewrite steps.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Rewrite payload for `CollectTermsStrategy`.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectTermsKernel {
    /// Equation after subtracting RHS from both sides.
    pub rewritten: Equation,
}

/// Build collect-terms rewrite only when the solve variable appears on both sides.
pub fn derive_collect_terms_kernel(
    ctx: &mut Context,
    eq: &Equation,
    var: &str,
) -> Option<CollectTermsKernel> {
    let lhs_has = contains_var(ctx, eq.lhs, var);
    let rhs_has = contains_var(ctx, eq.rhs, var);
    if !lhs_has || !rhs_has {
        return None;
    }
    Some(CollectTermsKernel {
        rewritten: crate::equation_rewrite::subtract_rhs_from_both_sides(ctx, eq),
    })
}

/// Build didactic narration for collect-terms subtraction.
pub fn collect_terms_message(rhs_display: &str) -> String {
    subtract_both_sides_message(rhs_display)
}

/// Build didactic payload for collect-terms rewrite.
pub fn build_collect_terms_step_with<F>(
    equation_after: Equation,
    original_rhs: ExprId,
    mut render_expr: F,
) -> StrategyDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    let rhs_desc = render_expr(original_rhs);
    StrategyDidacticStep {
        description: collect_terms_message(&rhs_desc),
        equation_after,
    }
}

/// Rewrite payload for `RationalExponentStrategy`.
#[derive(Debug, Clone, PartialEq)]
pub struct RationalExponentKernel {
    /// Rewritten equation `base^p = rhs^q`.
    pub rewritten: Equation,
    /// Denominator from the original rational exponent `p/q`.
    pub q: i64,
}

/// Rewrite an isolated rational exponent equation (`x^(p/q) = rhs`) into `x^p = rhs^q`.
pub fn derive_rational_exponent_kernel(
    ctx: &mut Context,
    eq: &Equation,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
) -> Option<RationalExponentKernel> {
    if eq.op != RelOp::Eq {
        return None;
    }
    let (rewritten, _p, q) = crate::rational_power::rewrite_isolated_rational_power_equation(
        ctx,
        eq.lhs,
        eq.rhs,
        var,
        eq.op.clone(),
        lhs_has_var,
        rhs_has_var,
    )?;
    Some(RationalExponentKernel { rewritten, q })
}

/// Build didactic narration for rational exponent elimination.
pub fn rational_exponent_message(q: i64) -> String {
    eliminate_fractional_exponent_message(&q.to_string())
}

/// Build didactic payload for rational-exponent elimination rewrite.
pub fn build_rational_exponent_step(q: i64, equation_after: Equation) -> StrategyDidacticStep {
    StrategyDidacticStep {
        description: rational_exponent_message(q),
        equation_after,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Context, Expr};

    #[test]
    fn collect_terms_kernel_requires_var_on_both_sides() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let lhs = ctx.add(Expr::Add(x, one));
        let eq = Equation {
            lhs,
            rhs: two,
            op: RelOp::Eq,
        };
        assert!(derive_collect_terms_kernel(&mut ctx, &eq, "x").is_none());
    }

    #[test]
    fn collect_terms_kernel_rewrites_when_var_on_both_sides() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = ctx.add(Expr::Add(x, two));
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        let kernel = derive_collect_terms_kernel(&mut ctx, &eq, "x");
        assert!(kernel.is_some());
    }

    #[test]
    fn rational_exponent_kernel_rewrites_isolated_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let exponent = ctx.add(Expr::Div(three, two));
        let lhs = ctx.add(Expr::Pow(x, exponent));
        let rhs = ctx.num(8);
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        let kernel = derive_rational_exponent_kernel(&mut ctx, &eq, "x", true, false)
            .expect("kernel should exist");
        assert_eq!(kernel.q, 2);

        // lhs should become x^3
        match ctx.get(kernel.rewritten.lhs) {
            Expr::Pow(base, exp) => {
                assert_eq!(*base, x);
                assert_eq!(*exp, three);
            }
            other => panic!("expected rewritten lhs pow, got {:?}", other),
        }
    }

    #[test]
    fn strategy_step_builders_use_expected_messages() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let collect = build_collect_terms_step_with(eq.clone(), y, |_| "rhs".to_string());
        assert_eq!(collect.description, "Subtract rhs from both sides");
        assert_eq!(collect.equation_after, eq);

        let rational = build_rational_exponent_step(3, eq.clone());
        assert_eq!(
            rational.description,
            "Raise both sides to power 3 to eliminate fractional exponent"
        );
        assert_eq!(rational.equation_after, eq);
    }
}
