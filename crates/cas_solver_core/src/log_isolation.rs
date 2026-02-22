use crate::isolation_utils::contains_var;
use cas_ast::{Context, Equation, Expr, ExprId, RelOp};

/// Didactic payload for one logarithm isolation rewrite.
#[derive(Debug, Clone, PartialEq)]
pub struct LogIsolationStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Planned log-isolation rewrite with equation + didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct LogIsolationRewritePlan {
    pub equation: Equation,
    pub step: LogIsolationStep,
}

/// Planned transformation for isolating a logarithmic equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogIsolationPlan {
    /// `log(base, arg) = rhs` -> `arg = base^rhs`
    SolveArgument { lhs: ExprId, rhs: ExprId },
    /// `log(base, arg) = rhs` -> `base = arg^(1/rhs)`
    SolveBase { lhs: ExprId, rhs: ExprId },
}

impl LogIsolationPlan {
    /// Convert a plan into the target equation preserving the incoming relation.
    pub fn into_equation(self, op: RelOp) -> Equation {
        match self {
            Self::SolveArgument { lhs, rhs } | Self::SolveBase { lhs, rhs } => {
                Equation { lhs, rhs, op }
            }
        }
    }

    /// Build didactic narration for the selected log-isolation rewrite.
    pub fn step_description(self, base_display: &str) -> String {
        match self {
            Self::SolveArgument { .. } => {
                format!("Exponentiate both sides with base {}", base_display)
            }
            Self::SolveBase { .. } => "Isolate base of logarithm".to_string(),
        }
    }
}

/// Build display-ready step payload for a logarithm isolation plan.
pub fn build_log_isolation_step_with<F>(
    plan: LogIsolationPlan,
    base: ExprId,
    op: RelOp,
    mut render_expr: F,
) -> LogIsolationStep
where
    F: FnMut(ExprId) -> String,
{
    let base_desc = render_expr(base);
    LogIsolationStep {
        description: plan.step_description(&base_desc),
        equation_after: plan.into_equation(op),
    }
}

/// Plan logarithm isolation and build its didactic step in one call.
pub fn plan_log_isolation_step(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    rhs: ExprId,
    var: &str,
    op: RelOp,
    base_display: &str,
) -> Option<LogIsolationRewritePlan> {
    plan_log_isolation_step_with(ctx, base, arg, rhs, var, op, |_| base_display.to_string())
}

/// Plan logarithm isolation and build its didactic step by rendering the base
/// expression with a caller-provided formatter.
pub fn plan_log_isolation_step_with<F>(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    rhs: ExprId,
    var: &str,
    op: RelOp,
    render_expr: F,
) -> Option<LogIsolationRewritePlan>
where
    F: FnMut(ExprId) -> String,
{
    let plan = plan_log_isolation(ctx, base, arg, rhs, var)?;
    let step = build_log_isolation_step_with(plan, base, op, render_expr);
    let equation = step.equation_after.clone();
    Some(LogIsolationRewritePlan { equation, step })
}

/// Build the transformed equation target for `log(base, arg) = rhs`.
///
/// Returns `None` when:
/// - both `base` and `arg` contain `var`, or
/// - neither contains `var`.
pub fn plan_log_isolation(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<LogIsolationPlan> {
    let arg_has_var = contains_var(ctx, arg, var);
    let base_has_var = contains_var(ctx, base, var);

    match (arg_has_var, base_has_var) {
        (true, false) => {
            let new_rhs = ctx.add(Expr::Pow(base, rhs));
            Some(LogIsolationPlan::SolveArgument {
                lhs: arg,
                rhs: new_rhs,
            })
        }
        (false, true) => {
            let one = ctx.num(1);
            let inv_rhs = ctx.add(Expr::Div(one, rhs));
            let new_rhs = ctx.add(Expr::Pow(arg, inv_rhs));
            Some(LogIsolationPlan::SolveBase {
                lhs: base,
                rhs: new_rhs,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_log_isolation_solves_argument_when_var_in_arg_only() {
        let mut ctx = Context::new();
        let b = ctx.num(2);
        let x = ctx.var("x");
        let rhs = ctx.num(3);

        let plan = plan_log_isolation(&mut ctx, b, x, rhs, "x")
            .expect("should isolate logarithm argument");
        match plan {
            LogIsolationPlan::SolveArgument {
                lhs,
                rhs: planned_rhs,
            } => {
                assert_eq!(lhs, x);
                assert!(
                    matches!(ctx.get(planned_rhs), Expr::Pow(base, exp) if *base == b && *exp == rhs)
                );
            }
            LogIsolationPlan::SolveBase { .. } => panic!("expected SolveArgument plan"),
        }
    }

    #[test]
    fn plan_log_isolation_solves_base_when_var_in_base_only() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let arg = ctx.num(5);
        let rhs = ctx.num(2);

        let plan =
            plan_log_isolation(&mut ctx, x, arg, rhs, "x").expect("should isolate logarithm base");
        match plan {
            LogIsolationPlan::SolveBase {
                lhs,
                rhs: planned_rhs,
            } => {
                assert_eq!(lhs, x);
                assert!(matches!(ctx.get(planned_rhs), Expr::Pow(base, _) if *base == arg));
            }
            LogIsolationPlan::SolveArgument { .. } => panic!("expected SolveBase plan"),
        }
    }

    #[test]
    fn plan_log_isolation_rejects_ambiguous_or_var_free_cases() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let three = ctx.num(3);

        assert!(plan_log_isolation(&mut ctx, x, x, three, "x").is_none());
        assert!(plan_log_isolation(&mut ctx, y, three, three, "x").is_none());
    }

    #[test]
    fn into_equation_preserves_relation_and_sides() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("y");
        let eq = LogIsolationPlan::SolveArgument { lhs, rhs }.into_equation(RelOp::Geq);
        assert_eq!(eq.lhs, lhs);
        assert_eq!(eq.rhs, rhs);
        assert_eq!(eq.op, RelOp::Geq);
    }

    #[test]
    fn step_description_for_solve_argument_uses_base_display() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("y");
        let msg = LogIsolationPlan::SolveArgument { lhs, rhs }.step_description("2");
        assert_eq!(msg, "Exponentiate both sides with base 2");
    }

    #[test]
    fn step_description_for_solve_base_is_constant_text() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("y");
        let msg = LogIsolationPlan::SolveBase { lhs, rhs }.step_description("ignored");
        assert_eq!(msg, "Isolate base of logarithm");
    }

    #[test]
    fn build_log_isolation_step_with_uses_rendered_base_and_equation() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("y");
        let base = ctx.var("b");
        let plan = LogIsolationPlan::SolveArgument { lhs, rhs };
        let step = build_log_isolation_step_with(plan, base, RelOp::Eq, |_| "b".to_string());
        assert_eq!(step.description, "Exponentiate both sides with base b");
        assert_eq!(step.equation_after.lhs, lhs);
        assert_eq!(step.equation_after.rhs, rhs);
        assert_eq!(step.equation_after.op, RelOp::Eq);
    }

    #[test]
    fn plan_log_isolation_step_builds_rewrite_and_step() {
        let mut ctx = Context::new();
        let base = ctx.num(2);
        let arg = ctx.var("x");
        let rhs = ctx.num(3);

        let out = plan_log_isolation_step(&mut ctx, base, arg, rhs, "x", RelOp::Eq, "2")
            .expect("log isolation should apply");
        assert_eq!(out.equation.lhs, arg);
        assert_eq!(out.step.description, "Exponentiate both sides with base 2");
        assert_eq!(out.step.equation_after, out.equation);
    }

    #[test]
    fn plan_log_isolation_step_with_uses_renderer() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);

        let out = plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_| {
            "rendered(b)".to_string()
        })
        .expect("log isolation should apply");

        assert_eq!(
            out.step.description,
            "Exponentiate both sides with base rendered(b)"
        );
        assert_eq!(out.step.equation_after, out.equation);
    }
}
