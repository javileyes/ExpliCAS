//! Unified unwrap rewrite planning for solver strategies.

use crate::log_domain::{LogAssumption, LogSolveDecision};
use crate::rational_power::PowUnwrapPlan;
use cas_ast::{Context, Equation, Expr, ExprId, RelOp};

/// Planned unwrap rewrite from a target expression.
#[derive(Debug, Clone, PartialEq)]
pub enum UnwrapRewritePlan {
    Unary {
        equation: Equation,
        description: String,
    },
    PowVariableBase {
        equation: Equation,
        exponent: ExprId,
    },
    PowLogLinear {
        equation: Equation,
        base: ExprId,
        assumptions: Vec<LogAssumption>,
    },
}

/// Plan unwrap rewrite for a target expression (`Function`/`Pow`).
pub fn plan_unwrap_rewrite<F>(
    ctx: &mut Context,
    target: ExprId,
    other: ExprId,
    var: &str,
    op: RelOp,
    is_lhs: bool,
    mut classify_log_solve: F,
) -> Option<UnwrapRewritePlan>
where
    F: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
{
    let target_data = ctx.get(target).clone();
    match target_data {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let arg = args[0];
            let name = ctx.sym_name(fn_id).to_string();
            let (equation, description) =
                crate::function_inverse::plan_unary_inverse_rewrite_for_unwrap(
                    ctx, &name, arg, other, op, is_lhs,
                )?;
            Some(UnwrapRewritePlan::Unary {
                equation,
                description: description.to_string(),
            })
        }
        Expr::Pow(_, _) => match crate::rational_power::plan_pow_unwrap_rewrite(
            ctx,
            target,
            other,
            var,
            op,
            is_lhs,
            |core_ctx, base, other_side| classify_log_solve(core_ctx, base, other_side),
        ) {
            Some(PowUnwrapPlan::VariableBase { equation, exponent }) => {
                Some(UnwrapRewritePlan::PowVariableBase { equation, exponent })
            }
            Some(PowUnwrapPlan::LogLinear {
                equation,
                base,
                assumptions,
            }) => Some(UnwrapRewritePlan::PowLogLinear {
                equation,
                base,
                assumptions,
            }),
            None => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;

    #[test]
    fn plans_unary_unwrap_rewrite() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let target = ctx.call("ln", vec![x]);

        let plan = plan_unwrap_rewrite(&mut ctx, target, y, "x", RelOp::Eq, true, |_, _, _| {
            LogSolveDecision::Ok
        })
        .expect("expected unary plan");

        match plan {
            UnwrapRewritePlan::Unary { .. } => {}
            other => panic!("expected unary plan, got {:?}", other),
        }
    }

    #[test]
    fn plans_pow_variable_base_unwrap_rewrite() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let exponent = ctx.add(Expr::Div(one, two));
        let target = ctx.add(Expr::Pow(x, exponent));

        let plan = plan_unwrap_rewrite(&mut ctx, target, y, "x", RelOp::Eq, true, |_, _, _| {
            LogSolveDecision::Ok
        })
        .expect("expected pow variable-base plan");

        match plan {
            UnwrapRewritePlan::PowVariableBase { .. } => {}
            other => panic!("expected variable-base plan, got {:?}", other),
        }
    }
}
