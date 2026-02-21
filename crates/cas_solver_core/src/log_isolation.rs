use crate::isolation_utils::contains_var;
use cas_ast::{Context, Expr, ExprId};

/// Planned transformation for isolating a logarithmic equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogIsolationPlan {
    /// `log(base, arg) = rhs` -> `arg = base^rhs`
    SolveArgument { lhs: ExprId, rhs: ExprId },
    /// `log(base, arg) = rhs` -> `base = arg^(1/rhs)`
    SolveBase { lhs: ExprId, rhs: ExprId },
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

        let plan =
            plan_log_isolation(&mut ctx, b, x, rhs, "x").expect("should isolate logarithm argument");
        match plan {
            LogIsolationPlan::SolveArgument { lhs, rhs: planned_rhs } => {
                assert_eq!(lhs, x);
                assert!(matches!(ctx.get(planned_rhs), Expr::Pow(base, exp) if *base == b && *exp == rhs));
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
            LogIsolationPlan::SolveBase { lhs, rhs: planned_rhs } => {
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
}
