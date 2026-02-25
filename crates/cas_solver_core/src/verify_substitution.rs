use crate::substitution::substitute_named_var;
use cas_ast::{Context, Equation, Expr, ExprId};

/// Substitute a candidate solution into both sides of an equation.
pub fn substitute_equation_sides(
    ctx: &mut Context,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> (ExprId, ExprId) {
    let lhs_sub = substitute_named_var(ctx, equation.lhs, var, solution);
    let rhs_sub = substitute_named_var(ctx, equation.rhs, var, solution);
    (lhs_sub, rhs_sub)
}

/// Substitute a candidate solution and return `lhs_sub - rhs_sub`.
pub fn substitute_equation_diff(
    ctx: &mut Context,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> ExprId {
    let (lhs_sub, rhs_sub) = substitute_equation_sides(ctx, equation, var, solution);
    ctx.add(Expr::Sub(lhs_sub, rhs_sub))
}

/// Verify a candidate solution with caller-provided substitution/simplification hooks.
pub fn verify_solution_with<FSubstitute, FSimplify, FEquivalent>(
    equation: &Equation,
    var: &str,
    solution: ExprId,
    mut substitute_sides: FSubstitute,
    mut simplify_expr: FSimplify,
    mut are_equivalent: FEquivalent,
) -> bool
where
    FSubstitute: FnMut(&Equation, &str, ExprId) -> (ExprId, ExprId),
    FSimplify: FnMut(ExprId) -> ExprId,
    FEquivalent: FnMut(ExprId, ExprId) -> bool,
{
    let (lhs_sub, rhs_sub) = substitute_sides(equation, var, solution);
    let lhs_sim = simplify_expr(lhs_sub);
    let rhs_sim = simplify_expr(rhs_sub);
    are_equivalent(lhs_sim, rhs_sim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::RelOp;

    #[test]
    fn substitute_equation_sides_replaces_named_var() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let five = ctx.num(5);
        let three = ctx.num(3);

        let lhs = ctx.add(Expr::Add(x, two)); // x + 2
        let eq = Equation {
            lhs,
            rhs: five,
            op: RelOp::Eq,
        };

        let (lhs_sub, rhs_sub) = substitute_equation_sides(&mut ctx, &eq, "x", three);
        assert!(matches!(ctx.get(lhs_sub), Expr::Add(_, _)));
        assert!(matches!(ctx.get(rhs_sub), Expr::Number(_)));
    }

    #[test]
    fn substitute_equation_diff_builds_sub_expression() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let zero = ctx.num(0);

        let eq = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };
        let diff = substitute_equation_diff(&mut ctx, &eq, "x", zero);
        assert!(matches!(ctx.get(diff), Expr::Sub(_, _)));
    }

    #[test]
    fn substitute_equation_sides_does_not_touch_other_variables() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };
        let one = ctx.num(1);
        let (lhs_sub, rhs_sub) = substitute_equation_sides(&mut ctx, &eq, "x", one);
        assert!(matches!(ctx.get(lhs_sub), Expr::Number(_)));
        assert!(matches!(ctx.get(rhs_sub), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == "y"));
    }

    #[test]
    fn verify_solution_with_runs_substitute_simplify_and_compare() {
        let mut context = Context::new();
        let x = context.var("x");
        let one = context.num(1);
        let equation = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };
        let context_cell = std::cell::RefCell::new(context);
        let simplify_calls = std::cell::Cell::new(0usize);
        let compare_calls = std::cell::Cell::new(0usize);

        let ok = verify_solution_with(
            &equation,
            "x",
            one,
            |eq, name, solution| {
                let mut ctx = context_cell.borrow_mut();
                substitute_equation_sides(&mut ctx, eq, name, solution)
            },
            |expr| {
                simplify_calls.set(simplify_calls.get() + 1);
                expr
            },
            |_lhs, _rhs| {
                compare_calls.set(compare_calls.get() + 1);
                true
            },
        );
        assert!(ok);
        assert_eq!(simplify_calls.get(), 2);
        assert_eq!(compare_calls.get(), 1);
    }

    #[test]
    fn verify_solution_with_forwards_compare_result() {
        let mut context = Context::new();
        let x = context.var("x");
        let one = context.num(1);
        let equation = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };
        let context_cell = std::cell::RefCell::new(context);
        let simplify_calls = std::cell::Cell::new(0usize);
        let compare_calls = std::cell::Cell::new(0usize);

        let ok = verify_solution_with(
            &equation,
            "x",
            one,
            |eq, name, solution| {
                let mut ctx = context_cell.borrow_mut();
                substitute_equation_sides(&mut ctx, eq, name, solution)
            },
            |expr| {
                simplify_calls.set(simplify_calls.get() + 1);
                expr
            },
            |_lhs, _rhs| {
                compare_calls.set(compare_calls.get() + 1);
                false
            },
        );
        assert!(!ok);
        assert_eq!(simplify_calls.get(), 2);
        assert_eq!(compare_calls.get(), 1);
    }
}
