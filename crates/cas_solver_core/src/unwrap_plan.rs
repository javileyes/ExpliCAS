//! Unified unwrap rewrite planning for solver strategies.

use crate::log_domain::{LogAssumption, LogSolveDecision};
use crate::rational_power::PowUnwrapPlan;
use crate::solve_outcome::take_log_base_message;
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

/// Solver-ready unwrap step derived from a rewrite plan.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwrapExecutionPlan {
    pub equation: Equation,
    pub description: String,
    pub assumptions: Vec<LogAssumption>,
    pub log_linear_base: Option<ExprId>,
}

/// Build narration for variable-base power unwrap rewrites.
pub fn pow_variable_base_unwrap_message(exponent_display: &str) -> String {
    format!("Raise both sides to 1/{}", exponent_display)
}

/// Convert a rewrite plan into an executable step payload.
///
/// `render_expr` is used only for variable-base power rewrites so callers
/// can control display format (ASCII, LaTeX, etc.).
pub fn build_unwrap_execution_plan_with<F>(
    plan: UnwrapRewritePlan,
    mut render_expr: F,
) -> UnwrapExecutionPlan
where
    F: FnMut(ExprId) -> String,
{
    match plan {
        UnwrapRewritePlan::Unary {
            equation,
            description,
        } => UnwrapExecutionPlan {
            equation,
            description,
            assumptions: vec![],
            log_linear_base: None,
        },
        UnwrapRewritePlan::PowVariableBase { equation, exponent } => UnwrapExecutionPlan {
            equation,
            description: pow_variable_base_unwrap_message(&render_expr(exponent)),
            assumptions: vec![],
            log_linear_base: None,
        },
        UnwrapRewritePlan::PowLogLinear {
            equation,
            base,
            assumptions,
        } => UnwrapExecutionPlan {
            equation,
            description: take_log_base_message("e"),
            assumptions,
            log_linear_base: Some(base),
        },
    }
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

    #[test]
    fn pow_variable_base_unwrap_message_formats_expected_text() {
        assert_eq!(
            pow_variable_base_unwrap_message("2"),
            "Raise both sides to 1/2"
        );
    }

    #[test]
    fn build_execution_plan_for_variable_base_uses_rendered_exponent() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let exponent = ctx.add(Expr::Div(one, two));
        let equation = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let plan = build_unwrap_execution_plan_with(
            UnwrapRewritePlan::PowVariableBase { equation, exponent },
            |id| format!("{:?}", ctx.get(id)),
        );
        assert!(plan.description.contains("1/Div"));
        assert_eq!(plan.log_linear_base, None);
        assert!(plan.assumptions.is_empty());
    }

    #[test]
    fn build_execution_plan_for_log_linear_keeps_assumptions() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let equation = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };
        let plan = build_unwrap_execution_plan_with(
            UnwrapRewritePlan::PowLogLinear {
                equation,
                base: x,
                assumptions: vec![LogAssumption::PositiveBase],
            },
            |_| "unused".to_string(),
        );
        assert_eq!(plan.log_linear_base, Some(x));
        assert_eq!(plan.assumptions, vec![LogAssumption::PositiveBase]);
        assert_eq!(plan.description, "Take log base e of both sides");
    }
}
