use cas_ast::{
    Case, ConditionPredicate, ConditionSet, Context, Equation, Expr, ExprId, SolutionSet,
};
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

/// Split discrete solutions into `(symbolic, non_symbolic)` buckets.
pub fn partition_discrete_symbolic(ctx: &Context, sols: &[ExprId]) -> (Vec<ExprId>, Vec<ExprId>) {
    let mut symbolic = Vec::new();
    let mut non_symbolic = Vec::new();
    for &sol in sols {
        if is_symbolic_expr(ctx, sol) {
            symbolic.push(sol);
        } else {
            non_symbolic.push(sol);
        }
    }
    (symbolic, non_symbolic)
}

/// Keep only solutions accepted by a verifier callback.
pub fn retain_verified_discrete<F>(sols: Vec<ExprId>, mut verify: F) -> Vec<ExprId>
where
    F: FnMut(ExprId) -> bool,
{
    let mut out = Vec::new();
    for sol in sols {
        if verify(sol) {
            out.push(sol);
        }
    }
    out
}

/// Merge symbolic roots with verified numeric roots.
///
/// Returns `symbolic ++ verified_numeric` preserving solver behavior.
pub fn merge_symbolic_with_verified_numeric<F>(
    mut symbolic_solutions: Vec<ExprId>,
    numeric_solutions: Vec<ExprId>,
    verify_numeric: F,
) -> Vec<ExprId>
where
    F: FnMut(ExprId) -> bool,
{
    let verified_numeric = retain_verified_discrete(numeric_solutions, verify_numeric);
    symbolic_solutions.extend(verified_numeric);
    symbolic_solutions
}

/// Decide whether a rewritten residual should replace the current one.
///
/// Accept when:
/// - The target variable was eliminated, or
/// - Tree size was reduced by more than 25% (avoids cosmetic rewrites).
pub fn should_accept_rewritten_residual(
    var_eliminated: bool,
    old_nodes: usize,
    new_nodes: usize,
) -> bool {
    var_eliminated || (old_nodes > 4 && new_nodes * 4 < old_nodes * 3)
}

/// Runtime contract for pre-solve equation-side simplification.
pub trait SolvePreprocessRuntime {
    fn context(&mut self) -> &mut Context;
    fn simplify_for_solve(&mut self, expr: ExprId) -> ExprId;
}

/// Simplify only equation sides that contain `var` and recompose `a^x / b^x` when possible.
pub fn simplify_equation_sides_for_var_with_runtime<R>(
    runtime: &mut R,
    eq: &Equation,
    var: &str,
) -> Equation
where
    R: SolvePreprocessRuntime,
{
    let mut simplified_eq = eq.clone();

    let lhs_has_var = {
        let ctx = runtime.context();
        super::isolation_utils::contains_var(ctx, eq.lhs, var)
    };
    if lhs_has_var {
        let sim_lhs = runtime.simplify_for_solve(eq.lhs);
        simplified_eq.lhs = sim_lhs;
        if let Some(recomposed) =
            super::isolation_utils::try_recompose_pow_quotient(runtime.context(), sim_lhs)
        {
            simplified_eq.lhs = recomposed;
        }
    }

    let rhs_has_var = {
        let ctx = runtime.context();
        super::isolation_utils::contains_var(ctx, eq.rhs, var)
    };
    if rhs_has_var {
        let sim_rhs = runtime.simplify_for_solve(eq.rhs);
        simplified_eq.rhs = sim_rhs;
        if let Some(recomposed) =
            super::isolation_utils::try_recompose_pow_quotient(runtime.context(), sim_rhs)
        {
            simplified_eq.rhs = recomposed;
        }
    }

    simplified_eq
}

/// Return the candidate residual when the rewrite is meaningfully better.
pub fn accept_residual_rewrite_candidate(
    ctx: &Context,
    current: ExprId,
    candidate: ExprId,
    var: &str,
) -> Option<ExprId> {
    let old_nodes = cas_ast::traversal::count_all_nodes(ctx, current);
    let new_nodes = cas_ast::traversal::count_all_nodes(ctx, candidate);
    let var_eliminated = !super::isolation_utils::contains_var(ctx, candidate, var);
    if should_accept_rewritten_residual(var_eliminated, old_nodes, new_nodes) {
        Some(candidate)
    } else {
        None
    }
}

/// Runtime contract for residual rewrite normalization.
pub trait ResidualRewriteRuntime {
    fn context(&mut self) -> &mut Context;
    fn expand_algebraic(&mut self, expr: ExprId) -> ExprId;
    fn simplify_for_solve(&mut self, expr: ExprId) -> ExprId;
    fn expand_trig(&mut self, expr: ExprId) -> ExprId;
}

/// Normalize a residual expression by applying the two engine-level fallback rewrites:
/// 1) algebraic expand + simplify
/// 2) trig expand mode
///
/// A rewrite is accepted only if it eliminates `var` or reduces tree size significantly.
pub fn normalize_variable_residual_with_runtime<R>(
    runtime: &mut R,
    residual: ExprId,
    var: &str,
) -> ExprId
where
    R: ResidualRewriteRuntime,
{
    let mut current = residual;

    let current_has_var = {
        let ctx = runtime.context();
        super::isolation_utils::contains_var(ctx, current, var)
    };
    if current_has_var {
        let expanded = runtime.expand_algebraic(current);
        let re_simplified = runtime.simplify_for_solve(expanded);
        if let Some(accepted) =
            accept_residual_rewrite_candidate(runtime.context(), current, re_simplified, var)
        {
            current = accepted;
        }
    }

    let current_has_var = {
        let ctx = runtime.context();
        super::isolation_utils::contains_var(ctx, current, var)
    };
    if current_has_var {
        let trig_expanded = runtime.expand_trig(current);
        if let Some(accepted) =
            accept_residual_rewrite_candidate(runtime.context(), current, trig_expanded, var)
        {
            current = accepted;
        }
    }

    current
}

/// Extract all denominators that contain the target variable.
pub fn extract_denominators_with_var(ctx: &Context, expr: ExprId, var: &str) -> Vec<ExprId> {
    let mut denoms_set: HashSet<ExprId> = HashSet::new();
    collect_denominators_into_set(ctx, expr, var, &mut denoms_set);
    denoms_set.into_iter().collect()
}

/// Collect unique denominator expressions containing `var` across equation sides.
pub fn collect_unique_denominators_with_var(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Vec<ExprId> {
    let mut denoms_set: HashSet<ExprId> = HashSet::new();
    denoms_set.extend(extract_denominators_with_var(ctx, lhs, var));
    denoms_set.extend(extract_denominators_with_var(ctx, rhs, var));
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
    use cas_ast::RelOp;

    struct TrackingPreprocessRuntime {
        ctx: Context,
        simplified_calls: Vec<ExprId>,
    }

    impl SolvePreprocessRuntime for TrackingPreprocessRuntime {
        fn context(&mut self) -> &mut Context {
            &mut self.ctx
        }

        fn simplify_for_solve(&mut self, expr: ExprId) -> ExprId {
            self.simplified_calls.push(expr);
            expr
        }
    }

    struct MockResidualRuntime {
        ctx: Context,
        algebraic_out: ExprId,
        trig_out: ExprId,
        algebraic_calls: usize,
        simplify_calls: usize,
        trig_calls: usize,
    }

    impl ResidualRewriteRuntime for MockResidualRuntime {
        fn context(&mut self) -> &mut Context {
            &mut self.ctx
        }

        fn expand_algebraic(&mut self, _expr: ExprId) -> ExprId {
            self.algebraic_calls += 1;
            self.algebraic_out
        }

        fn simplify_for_solve(&mut self, expr: ExprId) -> ExprId {
            self.simplify_calls += 1;
            expr
        }

        fn expand_trig(&mut self, _expr: ExprId) -> ExprId {
            self.trig_calls += 1;
            self.trig_out
        }
    }

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

    #[test]
    fn collect_unique_denominators_deduplicates_between_sides() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Div(one, x));
        let one2 = ctx.num(1);
        let rhs = ctx.add(Expr::Div(one2, x));
        let denoms = collect_unique_denominators_with_var(&ctx, lhs, rhs, "x");
        assert_eq!(denoms.len(), 1);
    }

    #[test]
    fn partition_discrete_symbolic_splits_expected() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        let (symbolic, non_symbolic) = partition_discrete_symbolic(&ctx, &[two, x]);
        assert_eq!(symbolic, vec![x]);
        assert_eq!(non_symbolic, vec![two]);
    }

    #[test]
    fn retain_verified_discrete_keeps_only_verified() {
        let sols = vec![
            cas_ast::ExprId::from_raw(1),
            cas_ast::ExprId::from_raw(2),
            cas_ast::ExprId::from_raw(3),
        ];
        let kept = retain_verified_discrete(sols, |id| id.index() % 2 == 1);
        assert_eq!(
            kept,
            vec![cas_ast::ExprId::from_raw(1), cas_ast::ExprId::from_raw(3)]
        );
    }

    #[test]
    fn accept_rewritten_residual_when_variable_eliminated() {
        assert!(should_accept_rewritten_residual(true, 100, 99));
    }

    #[test]
    fn accept_rewritten_residual_on_significant_reduction() {
        assert!(should_accept_rewritten_residual(false, 20, 14));
    }

    #[test]
    fn reject_rewritten_residual_on_cosmetic_change() {
        assert!(!should_accept_rewritten_residual(false, 20, 19));
    }

    #[test]
    fn merge_symbolic_with_verified_numeric_preserves_order_and_filters_numeric() {
        let x = cas_ast::ExprId::from_raw(11);
        let y = cas_ast::ExprId::from_raw(12);
        let two = cas_ast::ExprId::from_raw(2);
        let three = cas_ast::ExprId::from_raw(3);

        let out =
            merge_symbolic_with_verified_numeric(vec![x, y], vec![two, three], |id| id == three);
        assert_eq!(out, vec![x, y, three]);
    }

    #[test]
    fn simplify_equation_sides_for_var_only_simplifies_sides_with_variable() {
        let mut runtime = TrackingPreprocessRuntime {
            ctx: Context::new(),
            simplified_calls: vec![],
        };
        let x = runtime.ctx.var("x");
        let one = runtime.ctx.num(1);
        let two = runtime.ctx.num(2);
        let lhs = runtime.ctx.add(Expr::Add(x, one));
        let eq = Equation {
            lhs,
            rhs: two,
            op: RelOp::Eq,
        };

        let simplified = simplify_equation_sides_for_var_with_runtime(&mut runtime, &eq, "x");
        assert_eq!(simplified.lhs, lhs);
        assert_eq!(simplified.rhs, two);
        assert_eq!(runtime.simplified_calls, vec![lhs]);
    }

    #[test]
    fn accept_residual_candidate_accepts_variable_elimination() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let current = ctx.add(Expr::Add(x, one));
        let candidate = one;

        let accepted = accept_residual_rewrite_candidate(&ctx, current, candidate, "x");
        assert_eq!(accepted, Some(candidate));
    }

    #[test]
    fn accept_residual_candidate_rejects_cosmetic_rewrite() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let current = ctx.add(Expr::Add(x, one));
        let candidate = current;

        let accepted = accept_residual_rewrite_candidate(&ctx, current, candidate, "x");
        assert_eq!(accepted, None);
    }

    #[test]
    fn normalize_variable_residual_stops_after_algebraic_eliminates_variable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let residual = ctx.add(Expr::Add(x, one));

        let mut runtime = MockResidualRuntime {
            ctx,
            algebraic_out: one,
            trig_out: residual,
            algebraic_calls: 0,
            simplify_calls: 0,
            trig_calls: 0,
        };

        let normalized = normalize_variable_residual_with_runtime(&mut runtime, residual, "x");
        assert_eq!(normalized, one);
        assert_eq!(runtime.algebraic_calls, 1);
        assert_eq!(runtime.simplify_calls, 1);
        assert_eq!(runtime.trig_calls, 0);
    }

    #[test]
    fn normalize_variable_residual_uses_trig_fallback_when_algebraic_is_not_better() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let residual = ctx.add(Expr::Add(x, one));
        let algebraic_cosmetic = residual;
        let trig_eliminated = one;

        let mut runtime = MockResidualRuntime {
            ctx,
            algebraic_out: algebraic_cosmetic,
            trig_out: trig_eliminated,
            algebraic_calls: 0,
            simplify_calls: 0,
            trig_calls: 0,
        };

        let normalized = normalize_variable_residual_with_runtime(&mut runtime, residual, "x");
        assert_eq!(normalized, trig_eliminated);
        assert_eq!(runtime.algebraic_calls, 1);
        assert_eq!(runtime.simplify_calls, 1);
        assert_eq!(runtime.trig_calls, 1);
    }
}
