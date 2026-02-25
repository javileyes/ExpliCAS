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
    mut verify_numeric: F,
) -> Vec<ExprId>
where
    F: FnMut(ExprId) -> bool,
{
    let verified_numeric = retain_verified_discrete(numeric_solutions, &mut verify_numeric);
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

/// Variable presence classification across equation sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquationVarPresence {
    None,
    LhsOnly,
    RhsOnly,
    BothSides,
}

/// Classify where `var` appears in an equation.
pub fn classify_equation_var_presence(
    ctx: &Context,
    equation: &Equation,
    var: &str,
) -> EquationVarPresence {
    let lhs_has = super::isolation_utils::contains_var(ctx, equation.lhs, var);
    let rhs_has = super::isolation_utils::contains_var(ctx, equation.rhs, var);
    match (lhs_has, rhs_has) {
        (false, false) => EquationVarPresence::None,
        (true, false) => EquationVarPresence::LhsOnly,
        (false, true) => EquationVarPresence::RhsOnly,
        (true, true) => EquationVarPresence::BothSides,
    }
}

/// Simplify only equation sides that contain `var` and recompose `a^x / b^x` when possible
/// using caller-provided hooks.
pub fn simplify_equation_sides_for_var_with<FContains, FSimplify, FRecompose>(
    eq: &Equation,
    var: &str,
    mut contains_var: FContains,
    mut simplify_for_solve: FSimplify,
    mut try_recompose_pow_quotient: FRecompose,
) -> Equation
where
    FContains: FnMut(ExprId, &str) -> bool,
    FSimplify: FnMut(ExprId) -> ExprId,
    FRecompose: FnMut(ExprId) -> Option<ExprId>,
{
    let mut simplified_eq = eq.clone();

    if contains_var(eq.lhs, var) {
        let sim_lhs = simplify_for_solve(eq.lhs);
        simplified_eq.lhs = sim_lhs;
        if let Some(recomposed) = try_recompose_pow_quotient(sim_lhs) {
            simplified_eq.lhs = recomposed;
        }
    }

    if contains_var(eq.rhs, var) {
        let sim_rhs = simplify_for_solve(eq.rhs);
        simplified_eq.rhs = sim_rhs;
        if let Some(recomposed) = try_recompose_pow_quotient(sim_rhs) {
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

/// Normalize a residual expression by applying the two engine-level fallback rewrites:
/// 1) algebraic expand + simplify
/// 2) trig expand mode
///
/// A rewrite is accepted only if it eliminates `var` or reduces tree size significantly.
pub fn normalize_variable_residual_with<
    FContains,
    FExpandAlgebraic,
    FSimplifyForSolve,
    FExpandTrig,
    FAcceptCandidate,
>(
    residual: ExprId,
    var: &str,
    mut contains_var: FContains,
    mut expand_algebraic: FExpandAlgebraic,
    mut simplify_for_solve: FSimplifyForSolve,
    mut expand_trig: FExpandTrig,
    mut accept_candidate: FAcceptCandidate,
) -> ExprId
where
    FContains: FnMut(ExprId, &str) -> bool,
    FExpandAlgebraic: FnMut(ExprId) -> ExprId,
    FSimplifyForSolve: FnMut(ExprId) -> ExprId,
    FExpandTrig: FnMut(ExprId) -> ExprId,
    FAcceptCandidate: FnMut(ExprId, ExprId, &str) -> Option<ExprId>,
{
    let mut current = residual;

    if contains_var(current, var) {
        let expanded = expand_algebraic(current);
        let re_simplified = simplify_for_solve(expanded);
        if let Some(accepted) = accept_candidate(current, re_simplified, var) {
            current = accepted;
        }
    }

    if contains_var(current, var) {
        let trig_expanded = expand_trig(current);
        if let Some(accepted) = accept_candidate(current, trig_expanded, var) {
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

/// Apply non-zero exclusion guards only when exclusions exist.
pub fn apply_nonzero_exclusion_guards_if_any(
    solution_set: SolutionSet,
    exclusions: &[ExprId],
) -> SolutionSet {
    if exclusions.is_empty() {
        solution_set
    } else {
        apply_nonzero_exclusion_guards(solution_set, exclusions)
    }
}

/// Lift guard application over solved `(SolutionSet, payload)` results.
pub fn guard_solved_result_with_exclusions<T, E>(
    result: Result<(SolutionSet, T), E>,
    exclusions: &[ExprId],
) -> Result<(SolutionSet, T), E> {
    result.map(|(solution_set, payload)| {
        (
            apply_nonzero_exclusion_guards_if_any(solution_set, exclusions),
            payload,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::RelOp;
    use std::cell::{Cell, RefCell};

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
    fn apply_guards_if_any_keeps_solution_unchanged_for_empty_exclusions() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let sol = SolutionSet::Discrete(vec![one]);
        let guarded = apply_nonzero_exclusion_guards_if_any(sol.clone(), &[]);
        assert_eq!(guarded, sol);
    }

    #[test]
    fn guard_solved_result_with_exclusions_wraps_solution_set() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let result: Result<(SolutionSet, usize), ()> =
            Ok((SolutionSet::Discrete(vec![ctx.num(2)]), 7));
        let guarded = guard_solved_result_with_exclusions(result, &[x])
            .expect("guarding should preserve successful result");
        assert!(matches!(guarded.0, SolutionSet::Conditional(_)));
        assert_eq!(guarded.1, 7);
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
    fn retain_verified_discrete_invokes_verifier_for_each_solution() {
        let sols = vec![
            cas_ast::ExprId::from_raw(1),
            cas_ast::ExprId::from_raw(2),
            cas_ast::ExprId::from_raw(3),
        ];
        let calls = Cell::new(0usize);
        let kept = retain_verified_discrete(sols, |solution| {
            calls.set(calls.get() + 1);
            solution.index() % 2 == 1
        });
        assert_eq!(
            kept,
            vec![cas_ast::ExprId::from_raw(1), cas_ast::ExprId::from_raw(3)]
        );
        assert_eq!(calls.get(), 3);
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
    fn merge_symbolic_with_verified_numeric_invokes_numeric_verifier_for_each_candidate() {
        let x = cas_ast::ExprId::from_raw(11);
        let y = cas_ast::ExprId::from_raw(12);
        let two = cas_ast::ExprId::from_raw(2);
        let three = cas_ast::ExprId::from_raw(3);
        let calls = Cell::new(0usize);
        let out = merge_symbolic_with_verified_numeric(vec![x, y], vec![two, three], |solution| {
            calls.set(calls.get() + 1);
            solution == three
        });
        assert_eq!(out, vec![x, y, three]);
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn classify_equation_var_presence_none() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let eq = Equation {
            lhs: one,
            rhs: two,
            op: RelOp::Eq,
        };
        assert_eq!(
            classify_equation_var_presence(&ctx, &eq, "x"),
            EquationVarPresence::None
        );
    }

    #[test]
    fn classify_equation_var_presence_lhs_only() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };
        assert_eq!(
            classify_equation_var_presence(&ctx, &eq, "x"),
            EquationVarPresence::LhsOnly
        );
    }

    #[test]
    fn classify_equation_var_presence_rhs_only() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: one,
            rhs: x,
            op: RelOp::Eq,
        };
        assert_eq!(
            classify_equation_var_presence(&ctx, &eq, "x"),
            EquationVarPresence::RhsOnly
        );
    }

    #[test]
    fn classify_equation_var_presence_both_sides() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = ctx.add(Expr::Sub(x, one));
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        assert_eq!(
            classify_equation_var_presence(&ctx, &eq, "x"),
            EquationVarPresence::BothSides
        );
    }

    #[test]
    fn simplify_equation_sides_for_var_only_simplifies_sides_with_variable() {
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
        let simplified_calls = RefCell::new(Vec::new());

        let simplified = simplify_equation_sides_for_var_with(
            &eq,
            "x",
            |expr, _| expr == lhs,
            |expr| {
                simplified_calls.borrow_mut().push(expr);
                expr
            },
            |_expr| None,
        );
        assert_eq!(simplified.lhs, lhs);
        assert_eq!(simplified.rhs, two);
        assert_eq!(*simplified_calls.borrow(), vec![lhs]);
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
        let algebraic_calls = Cell::new(0usize);
        let simplify_calls = Cell::new(0usize);
        let trig_calls = Cell::new(0usize);

        let normalized = normalize_variable_residual_with(
            residual,
            "x",
            |expr, _| expr == residual,
            |_expr| {
                algebraic_calls.set(algebraic_calls.get() + 1);
                one
            },
            |expr| {
                simplify_calls.set(simplify_calls.get() + 1);
                expr
            },
            |_expr| {
                trig_calls.set(trig_calls.get() + 1);
                residual
            },
            |current, candidate, var| {
                accept_residual_rewrite_candidate(&ctx, current, candidate, var)
            },
        );
        assert_eq!(normalized, one);
        assert_eq!(algebraic_calls.get(), 1);
        assert_eq!(simplify_calls.get(), 1);
        assert_eq!(trig_calls.get(), 0);
    }

    #[test]
    fn normalize_variable_residual_uses_trig_fallback_when_algebraic_is_not_better() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let residual = ctx.add(Expr::Add(x, one));
        let algebraic_cosmetic = residual;
        let trig_eliminated = one;
        let algebraic_calls = Cell::new(0usize);
        let simplify_calls = Cell::new(0usize);
        let trig_calls = Cell::new(0usize);

        let normalized = normalize_variable_residual_with(
            residual,
            "x",
            |expr, _| expr == residual,
            |_expr| {
                algebraic_calls.set(algebraic_calls.get() + 1);
                algebraic_cosmetic
            },
            |expr| {
                simplify_calls.set(simplify_calls.get() + 1);
                expr
            },
            |_expr| {
                trig_calls.set(trig_calls.get() + 1);
                trig_eliminated
            },
            |current, candidate, var| {
                accept_residual_rewrite_candidate(&ctx, current, candidate, var)
            },
        );
        assert_eq!(normalized, trig_eliminated);
        assert_eq!(algebraic_calls.get(), 1);
        assert_eq!(simplify_calls.get(), 1);
        assert_eq!(trig_calls.get(), 1);
    }
}
