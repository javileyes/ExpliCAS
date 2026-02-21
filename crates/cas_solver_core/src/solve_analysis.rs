use cas_ast::{Case, ConditionPredicate, ConditionSet, Context, Expr, ExprId, SolutionSet};
use std::collections::HashSet;

/// Check if an expression is symbolic (contains variables/functions/constants).
pub fn is_symbolic_expr(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => false,
        Expr::Constant(_) => true,
        Expr::Variable(_) => true,
        Expr::Function(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            is_symbolic_expr(ctx, *l) || is_symbolic_expr(ctx, *r)
        }
        Expr::Neg(e) | Expr::Hold(e) => is_symbolic_expr(ctx, *e),
        Expr::Matrix { data, .. } => data.iter().any(|d| is_symbolic_expr(ctx, *d)),
        Expr::SessionRef(_) => true,
    }
}

/// Extract all denominators that contain the target variable.
pub fn extract_denominators_with_var(ctx: &Context, expr: ExprId, var: &str) -> Vec<ExprId> {
    let mut denoms_set: HashSet<ExprId> = HashSet::new();
    collect_denominators_into_set(ctx, expr, var, &mut denoms_set);
    denoms_set.into_iter().collect()
}

fn collect_denominators_into_set(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    denoms: &mut HashSet<ExprId>,
) {
    match ctx.get(expr) {
        Expr::Div(num, denom) => {
            if super::isolation_utils::contains_var(ctx, *denom, var) {
                denoms.insert(*denom);
            }
            collect_denominators_into_set(ctx, *num, var, denoms);
            collect_denominators_into_set(ctx, *denom, var, denoms);
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
            collect_denominators_into_set(ctx, *l, var, denoms);
            collect_denominators_into_set(ctx, *r, var, denoms);
        }
        Expr::Neg(e) | Expr::Hold(e) => {
            collect_denominators_into_set(ctx, *e, var, denoms);
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_denominators_into_set(ctx, *arg, var, denoms);
            }
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                collect_denominators_into_set(ctx, *elem, var, denoms);
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
    }
}

/// Apply non-zero exclusion guards to a solution set.
pub fn apply_nonzero_exclusion_guards(
    solution_set: SolutionSet,
    exclusions: &[ExprId],
) -> SolutionSet {
    if exclusions.is_empty() {
        return solution_set;
    }

    let mut guard = ConditionSet::empty();
    for &denom in exclusions {
        guard.push(ConditionPredicate::NonZero(denom));
    }

    let cases = vec![
        Case::new(guard, solution_set),
        Case::new(ConditionSet::empty(), SolutionSet::Empty),
    ];
    SolutionSet::Conditional(cases).simplify()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbolic_number_vs_variable() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        assert!(!is_symbolic_expr(&ctx, two));
        assert!(is_symbolic_expr(&ctx, x));
    }

    #[test]
    fn extract_denominators_basic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let div = ctx.add(Expr::Div(y, x));
        let denoms = extract_denominators_with_var(&ctx, div, "x");
        assert_eq!(denoms.len(), 1);
        assert_eq!(denoms[0], x);
    }

    #[test]
    fn apply_guards_builds_conditional() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sol = SolutionSet::Discrete(vec![ctx.num(1)]);
        let guarded = apply_nonzero_exclusion_guards(sol, &[x]);
        assert!(matches!(guarded, SolutionSet::Conditional(_)));
    }
}
