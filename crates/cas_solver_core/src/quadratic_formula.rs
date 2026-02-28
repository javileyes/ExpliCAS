use crate::solution_set::get_number;
use cas_ast::{Context, Expr, ExprId, RelOp, SolutionSet};
use num_rational::BigRational;

/// Build `sqrt(radicand)` as `radicand^(1/2)` in AST form.
pub fn sqrt_expr(ctx: &mut Context, radicand: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let half = ctx.add(Expr::Div(one, two));
    ctx.add(Expr::Pow(radicand, half))
}

/// Build both quadratic-formula roots from `a`, `b`, and a precomputed `sqrt(delta)`.
///
/// Returns `(x1, x2)` where:
/// - `x1 = (-b - sqrt(delta)) / (2a)`
/// - `x2 = (-b + sqrt(delta)) / (2a)`
pub fn roots_from_a_b_and_sqrt(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    sqrt_delta: ExprId,
) -> (ExprId, ExprId) {
    let neg_b = ctx.add(Expr::Neg(b));
    let two = ctx.num(2);
    let two_a = ctx.add(Expr::Mul(two, a));

    let num1 = ctx.add(Expr::Sub(neg_b, sqrt_delta));
    let x1 = ctx.add(Expr::Div(num1, two_a));

    let num2 = ctx.add(Expr::Add(neg_b, sqrt_delta));
    let x2 = ctx.add(Expr::Div(num2, two_a));

    (x1, x2)
}

/// Build both quadratic-formula roots from `a`, `b`, and `delta`.
pub fn roots_from_a_b_delta(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    delta: ExprId,
) -> (ExprId, ExprId) {
    let sqrt_delta = sqrt_expr(ctx, delta);
    roots_from_a_b_and_sqrt(ctx, a, b, sqrt_delta)
}

/// Build both quadratic-formula roots from `a`, `b`, and an already
/// simplified discriminant expression, applying square-factor extraction
/// before root construction.
pub fn roots_from_a_b_and_simplified_delta(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    simplified_delta: ExprId,
) -> (ExprId, ExprId) {
    let sqrt_delta_raw = sqrt_expr(ctx, simplified_delta);
    let sqrt_delta = crate::quadratic_sqrt::pull_square_from_sqrt(ctx, sqrt_delta_raw);
    roots_from_a_b_and_sqrt(ctx, a, b, sqrt_delta)
}

/// Build symbolic discriminant expression `b^2 - 4ac`.
pub fn discriminant_expr(ctx: &mut Context, a: ExprId, b: ExprId, c: ExprId) -> ExprId {
    let two = ctx.num(2);
    let b2 = ctx.add(Expr::Pow(b, two));
    let four = ctx.num(4);
    let four_a = ctx.add(Expr::Mul(four, a));
    let four_ac = ctx.add(Expr::Mul(four_a, c));
    ctx.add(Expr::Sub(b2, four_ac))
}

/// Compute the quadratic discriminant `b^2 - 4ac`.
pub fn discriminant(a: &BigRational, b: &BigRational, c: &BigRational) -> BigRational {
    b.clone() * b.clone() - BigRational::from_integer(4.into()) * a.clone() * c.clone()
}

/// Planning error for quadratic coefficient solving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadraticCoefficientSolvePlanError {
    /// Symbolic coefficients with non-equality operators are not yet supported.
    UnsupportedSymbolicInequality,
}

/// Classification of quadratic solving mode from simplified coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuadraticCoefficientSolvePlan {
    /// All coefficients are numeric: includes exact discriminant and raw roots.
    Numeric {
        delta: BigRational,
        roots: (ExprId, ExprId),
        opens_up: bool,
    },
    /// Coefficients are symbolic: includes raw discriminant expression.
    SymbolicEq { delta_expr: ExprId },
}

/// Build a coefficient-solving plan for quadratic strategy.
///
/// Input coefficients are expected to be simplified `a`, `b`, `c` from
/// `a*x^2 + b*x + c = 0`.
pub fn build_quadratic_coefficient_solve_plan(
    ctx: &mut Context,
    op: RelOp,
    a: ExprId,
    b: ExprId,
    c: ExprId,
) -> Result<QuadraticCoefficientSolvePlan, QuadraticCoefficientSolvePlanError> {
    let a_num = get_number(ctx, a);
    let b_num = get_number(ctx, b);
    let c_num = get_number(ctx, c);

    if let (Some(a_val), Some(b_val), Some(c_val)) = (a_num, b_num, c_num) {
        let delta = discriminant(&a_val, &b_val, &c_val);
        let delta_expr = ctx.add(Expr::Number(delta.clone()));
        let roots = roots_from_a_b_delta(ctx, a, b, delta_expr);
        let opens_up = a_val > BigRational::from_integer(0.into());
        return Ok(QuadraticCoefficientSolvePlan::Numeric {
            delta,
            roots,
            opens_up,
        });
    }

    if op != RelOp::Eq {
        return Err(QuadraticCoefficientSolvePlanError::UnsupportedSymbolicInequality);
    }

    let delta_expr = discriminant_expr(ctx, a, b, c);
    Ok(QuadraticCoefficientSolvePlan::SymbolicEq { delta_expr })
}

/// Solve a precomputed quadratic coefficient plan and return the resulting
/// solution set. Expansion/simplification are injected by caller.
#[allow(clippy::too_many_arguments)]
pub fn solve_quadratic_coefficient_solve_plan_with<FExpand, FSimplify, FNumeric, FRoots>(
    op: RelOp,
    a: ExprId,
    b: ExprId,
    plan: QuadraticCoefficientSolvePlan,
    mut expand_expr: FExpand,
    mut simplify_expr: FSimplify,
    mut solve_numeric: FNumeric,
    mut roots_from_simplified_delta: FRoots,
) -> SolutionSet
where
    FExpand: FnMut(ExprId) -> ExprId,
    FSimplify: FnMut(ExprId) -> ExprId,
    FNumeric: FnMut(RelOp, BigRational, (ExprId, ExprId), bool) -> SolutionSet,
    FRoots: FnMut(ExprId, ExprId, ExprId) -> (ExprId, ExprId),
{
    match plan {
        QuadraticCoefficientSolvePlan::Numeric {
            delta,
            roots: (sol1, sol2),
            opens_up,
        } => {
            let sim_sol1 = simplify_expr(sol1);
            let sim_sol2 = simplify_expr(sol2);
            solve_numeric(op, delta, (sim_sol1, sim_sol2), opens_up)
        }
        QuadraticCoefficientSolvePlan::SymbolicEq { delta_expr } => {
            let delta_expanded = expand_expr(delta_expr);
            let sim_delta = simplify_expr(delta_expanded);

            let (sol1_raw, sol2_raw) = roots_from_simplified_delta(a, b, sim_delta);
            let sol1_expanded = expand_expr(sol1_raw);
            let sim_sol1 = simplify_expr(sol1_expanded);
            let sol2_expanded = expand_expr(sol2_raw);
            let sim_sol2 = simplify_expr(sol2_expanded);

            SolutionSet::Discrete(vec![sim_sol1, sim_sol2])
        }
    }
}

/// Stateful variant of [`solve_quadratic_coefficient_solve_plan_with`].
///
/// This allows callers to reuse one mutable state object across expansion,
/// simplification, numeric solving, and symbolic root construction callbacks.
#[allow(clippy::too_many_arguments)]
pub fn solve_quadratic_coefficient_solve_plan_with_state<T, FExpand, FSimplify, FNumeric, FRoots>(
    state: &mut T,
    op: RelOp,
    a: ExprId,
    b: ExprId,
    plan: QuadraticCoefficientSolvePlan,
    mut expand_expr: FExpand,
    mut simplify_expr: FSimplify,
    mut solve_numeric: FNumeric,
    mut roots_from_simplified_delta: FRoots,
) -> SolutionSet
where
    FExpand: FnMut(&mut T, ExprId) -> ExprId,
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FNumeric: FnMut(&mut T, RelOp, BigRational, (ExprId, ExprId), bool) -> SolutionSet,
    FRoots: FnMut(&mut T, ExprId, ExprId, ExprId) -> (ExprId, ExprId),
{
    match plan {
        QuadraticCoefficientSolvePlan::Numeric {
            delta,
            roots: (sol1, sol2),
            opens_up,
        } => {
            let sim_sol1 = simplify_expr(state, sol1);
            let sim_sol2 = simplify_expr(state, sol2);
            solve_numeric(state, op, delta, (sim_sol1, sim_sol2), opens_up)
        }
        QuadraticCoefficientSolvePlan::SymbolicEq { delta_expr } => {
            let delta_expanded = expand_expr(state, delta_expr);
            let sim_delta = simplify_expr(state, delta_expanded);

            let (sol1_raw, sol2_raw) = roots_from_simplified_delta(state, a, b, sim_delta);
            let sol1_expanded = expand_expr(state, sol1_raw);
            let sim_sol1 = simplify_expr(state, sol1_expanded);
            let sol2_expanded = expand_expr(state, sol2_raw);
            let sim_sol2 = simplify_expr(state, sol2_expanded);

            SolutionSet::Discrete(vec![sim_sol1, sim_sol2])
        }
    }
}

/// Build and solve quadratic coefficient plan in one call using the default
/// numeric ordering/inequality kernel from `solution_set`.
///
/// Expansion/simplification are still injected by caller so strategy policy
/// remains outside solver-core.
#[allow(clippy::too_many_arguments)]
pub fn execute_quadratic_coefficient_solve_plan_with_default_numeric_solution_with_state<
    T,
    E,
    FContextMut,
    FExpand,
    FSimplify,
    FMapPlanError,
>(
    state: &mut T,
    op: RelOp,
    a: ExprId,
    b: ExprId,
    c: ExprId,
    context_mut: FContextMut,
    mut expand_expr: FExpand,
    mut simplify_expr: FSimplify,
    map_plan_error: FMapPlanError,
) -> Result<SolutionSet, E>
where
    FContextMut: Fn(&mut T) -> &mut Context,
    FExpand: FnMut(&mut T, ExprId) -> ExprId,
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FMapPlanError: FnOnce(QuadraticCoefficientSolvePlanError) -> E,
{
    let context_mut = &context_mut;
    let plan = build_quadratic_coefficient_solve_plan(context_mut(state), op.clone(), a, b, c)
        .map_err(map_plan_error)?;

    Ok(solve_quadratic_coefficient_solve_plan_with_state(
        state,
        op,
        a,
        b,
        plan,
        |state, expr| expand_expr(state, expr),
        |state, expr| simplify_expr(state, expr),
        |state, eq_op, delta, (sol1, sol2), opens_up| {
            let ctx = context_mut(state);
            let (r1, r2) = crate::solution_set::order_pair_by_value(ctx, sol1, sol2);
            crate::solution_set::quadratic_numeric_solution(ctx, eq_op, &delta, opens_up, r1, r2)
        },
        |state, solve_a, solve_b, sim_delta| {
            roots_from_a_b_and_simplified_delta(context_mut(state), solve_a, solve_b, sim_delta)
        },
    ))
}

/// Solve a precomputed quadratic coefficient plan using the default numeric
/// ordering/inequality solver from `solution_set`.
///
/// Expansion/simplification remain injected by caller so engine-side policy
/// (e.g. expand strategy) stays outside the core crate.
#[allow(clippy::too_many_arguments)]
pub fn solve_quadratic_coefficient_solve_plan_with_default_numeric_solution<FExpand, FSimplify>(
    ctx: &mut Context,
    op: RelOp,
    a: ExprId,
    b: ExprId,
    plan: QuadraticCoefficientSolvePlan,
    mut expand_expr: FExpand,
    mut simplify_expr: FSimplify,
) -> SolutionSet
where
    FExpand: FnMut(ExprId) -> ExprId,
    FSimplify: FnMut(ExprId) -> ExprId,
{
    match plan {
        QuadraticCoefficientSolvePlan::Numeric {
            delta,
            roots: (sol1, sol2),
            opens_up,
        } => {
            let sim_sol1 = simplify_expr(sol1);
            let sim_sol2 = simplify_expr(sol2);
            let (r1, r2) = crate::solution_set::order_pair_by_value(ctx, sim_sol1, sim_sol2);
            crate::solution_set::quadratic_numeric_solution(ctx, op, &delta, opens_up, r1, r2)
        }
        QuadraticCoefficientSolvePlan::SymbolicEq { delta_expr } => {
            let delta_expanded = expand_expr(delta_expr);
            let sim_delta = simplify_expr(delta_expanded);

            let (sol1_raw, sol2_raw) = roots_from_a_b_and_simplified_delta(ctx, a, b, sim_delta);
            let sol1_expanded = expand_expr(sol1_raw);
            let sim_sol1 = simplify_expr(sol1_expanded);
            let sol2_expanded = expand_expr(sol2_raw);
            let sim_sol2 = simplify_expr(sol2_expanded);

            SolutionSet::Discrete(vec![sim_sol1, sim_sol2])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;

    #[test]
    fn test_sqrt_expr_builds_half_power() {
        let mut ctx = Context::new();
        let d = ctx.num(5);
        let s = sqrt_expr(&mut ctx, d);
        match ctx.get(s) {
            Expr::Pow(base, exp) => {
                assert_eq!(*base, d);
                match ctx.get(*exp) {
                    Expr::Div(n, m) => {
                        assert!(matches!(ctx.get(*n), Expr::Number(_)));
                        assert!(matches!(ctx.get(*m), Expr::Number(_)));
                    }
                    other => panic!("Expected Div exponent, got {:?}", other),
                }
            }
            other => panic!("Expected Pow, got {:?}", other),
        }
    }

    #[test]
    fn test_roots_from_a_b_delta_builds_two_divisions() {
        let mut ctx = Context::new();
        let a = ctx.num(2);
        let b = ctx.num(3);
        let d = ctx.num(1);
        let (x1, x2) = roots_from_a_b_delta(&mut ctx, a, b, d);

        assert!(matches!(ctx.get(x1), Expr::Div(_, _)));
        assert!(matches!(ctx.get(x2), Expr::Div(_, _)));
    }

    #[test]
    fn test_discriminant() {
        let a = BigRational::from_integer(1.into());
        let b = BigRational::from_integer(3.into());
        let c = BigRational::from_integer(2.into());
        let d = discriminant(&a, &b, &c);
        assert_eq!(d, BigRational::from_integer(1.into()));
    }

    #[test]
    fn test_roots_from_a_b_and_simplified_delta_builds_two_divisions() {
        let mut ctx = Context::new();
        let a = ctx.num(2);
        let b = ctx.num(3);
        let d = ctx.num(5);
        let (x1, x2) = roots_from_a_b_and_simplified_delta(&mut ctx, a, b, d);
        assert!(matches!(ctx.get(x1), Expr::Div(_, _)));
        assert!(matches!(ctx.get(x2), Expr::Div(_, _)));
    }

    #[test]
    fn test_discriminant_expr_shape() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let d = discriminant_expr(&mut ctx, a, b, c);
        assert!(matches!(ctx.get(d), Expr::Sub(_, _)));
    }

    #[test]
    fn build_quadratic_coefficient_solve_plan_returns_numeric_for_numeric_coeffs() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(-3);
        let c = ctx.num(2);
        let plan = build_quadratic_coefficient_solve_plan(&mut ctx, RelOp::Eq, a, b, c)
            .expect("numeric coefficients should build numeric plan");
        match plan {
            QuadraticCoefficientSolvePlan::Numeric {
                delta, opens_up, ..
            } => {
                assert_eq!(delta, BigRational::from_integer(1.into()));
                assert!(opens_up);
            }
            _ => panic!("expected numeric plan"),
        }
    }

    #[test]
    fn build_quadratic_coefficient_solve_plan_returns_symbolic_for_symbolic_eq() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let plan = build_quadratic_coefficient_solve_plan(&mut ctx, RelOp::Eq, a, b, c)
            .expect("symbolic equality should build symbolic plan");
        match plan {
            QuadraticCoefficientSolvePlan::SymbolicEq { delta_expr } => {
                assert!(matches!(ctx.get(delta_expr), Expr::Sub(_, _)));
            }
            _ => panic!("expected symbolic plan"),
        }
    }

    #[test]
    fn build_quadratic_coefficient_solve_plan_rejects_symbolic_inequality() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let err = build_quadratic_coefficient_solve_plan(&mut ctx, RelOp::Lt, a, b, c)
            .expect_err("symbolic inequality should be rejected");
        assert_eq!(
            err,
            QuadraticCoefficientSolvePlanError::UnsupportedSymbolicInequality
        );
    }

    #[test]
    fn solve_quadratic_coefficient_solve_plan_with_numeric_returns_discrete_roots() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(-3);
        let c = ctx.num(2);
        let plan = build_quadratic_coefficient_solve_plan(&mut ctx, RelOp::Eq, a, b, c)
            .expect("numeric coefficients should build numeric plan");

        let solved = solve_quadratic_coefficient_solve_plan_with(
            RelOp::Eq,
            a,
            b,
            plan,
            |id| id,
            |id| id,
            |_op, _delta, (r1, r2), _opens_up| SolutionSet::Discrete(vec![r1, r2]),
            |aa, bb, delta| roots_from_a_b_and_simplified_delta(&mut ctx, aa, bb, delta),
        );
        assert!(matches!(solved, SolutionSet::Discrete(_)));
    }

    #[test]
    fn solve_quadratic_coefficient_solve_plan_with_symbolic_returns_discrete_set() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let plan = build_quadratic_coefficient_solve_plan(&mut ctx, RelOp::Eq, a, b, c)
            .expect("symbolic equality should build symbolic plan");

        let solved = solve_quadratic_coefficient_solve_plan_with(
            RelOp::Eq,
            a,
            b,
            plan,
            |id| id,
            |id| id,
            |_op, _delta, (r1, r2), _opens_up| SolutionSet::Discrete(vec![r1, r2]),
            |aa, bb, delta| roots_from_a_b_and_simplified_delta(&mut ctx, aa, bb, delta),
        );

        match solved {
            SolutionSet::Discrete(solutions) => assert_eq!(solutions.len(), 2),
            _ => panic!("expected discrete symbolic solutions"),
        }
    }

    #[test]
    fn solve_quadratic_coefficient_solve_plan_with_state_numeric_returns_discrete_roots() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(-3);
        let c = ctx.num(2);
        let plan = build_quadratic_coefficient_solve_plan(&mut ctx, RelOp::Eq, a, b, c)
            .expect("numeric coefficients should build numeric plan");

        #[derive(Default)]
        struct HookState {
            simplify_calls: usize,
            numeric_calls: usize,
        }
        let mut state = HookState::default();

        let solved = solve_quadratic_coefficient_solve_plan_with_state(
            &mut state,
            RelOp::Eq,
            a,
            b,
            plan,
            |_state, id| id,
            |state, id| {
                state.simplify_calls += 1;
                id
            },
            |state, _op, _delta, (r1, r2), _opens_up| {
                state.numeric_calls += 1;
                SolutionSet::Discrete(vec![r1, r2])
            },
            |_, aa, bb, delta| roots_from_a_b_and_simplified_delta(&mut ctx, aa, bb, delta),
        );

        assert!(matches!(solved, SolutionSet::Discrete(_)));
        assert_eq!(state.simplify_calls, 2);
        assert_eq!(state.numeric_calls, 1);
    }

    #[test]
    fn solve_quadratic_coefficient_solve_plan_with_default_numeric_solution_numeric_plan() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(-3);
        let c = ctx.num(2);
        let plan = build_quadratic_coefficient_solve_plan(&mut ctx, RelOp::Eq, a, b, c)
            .expect("numeric coefficients should build numeric plan");

        let solved = solve_quadratic_coefficient_solve_plan_with_default_numeric_solution(
            &mut ctx,
            RelOp::Eq,
            a,
            b,
            plan,
            |id| id,
            |id| id,
        );

        assert!(matches!(solved, SolutionSet::Discrete(_)));
    }

    #[test]
    fn solve_quadratic_coefficient_solve_plan_with_default_numeric_solution_symbolic_plan() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let plan = build_quadratic_coefficient_solve_plan(&mut ctx, RelOp::Eq, a, b, c)
            .expect("symbolic equality should build symbolic plan");

        let solved = solve_quadratic_coefficient_solve_plan_with_default_numeric_solution(
            &mut ctx,
            RelOp::Eq,
            a,
            b,
            plan,
            |id| id,
            |id| id,
        );

        match solved {
            SolutionSet::Discrete(solutions) => assert_eq!(solutions.len(), 2),
            _ => panic!("expected discrete symbolic solutions"),
        }
    }

    #[test]
    fn execute_quadratic_coefficient_solve_plan_with_default_numeric_solution_with_state_numeric_plan(
    ) {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(-3);
        let c = ctx.num(2);

        let solved =
            execute_quadratic_coefficient_solve_plan_with_default_numeric_solution_with_state(
                &mut ctx,
                RelOp::Eq,
                a,
                b,
                c,
                |ctx| ctx,
                |_ctx, id| id,
                |_ctx, id| id,
                |_err| "plan error".to_string(),
            )
            .expect("numeric plan should solve");

        assert!(matches!(solved, SolutionSet::Discrete(_)));
    }

    #[test]
    fn execute_quadratic_coefficient_solve_plan_with_default_numeric_solution_with_state_maps_plan_error(
    ) {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        let err =
            execute_quadratic_coefficient_solve_plan_with_default_numeric_solution_with_state(
                &mut ctx,
                RelOp::Lt,
                a,
                b,
                c,
                |ctx| ctx,
                |_ctx, id| id,
                |_ctx, id| id,
                |err| err,
            )
            .expect_err("symbolic inequality should map plan error");

        assert_eq!(
            err,
            QuadraticCoefficientSolvePlanError::UnsupportedSymbolicInequality
        );
    }
}
