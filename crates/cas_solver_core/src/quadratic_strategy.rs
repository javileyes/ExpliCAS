//! Quadratic-strategy orchestration helpers shared by engine-side solver.
//!
//! These wrappers keep quadratic-strategy orchestration in `cas_solver_core`:
//! candidate preparation + didactic step construction + coefficient-plan solving.

use cas_ast::{Context, Equation, Expr, ExprId, RelOp, SolutionSet};

use crate::quadratic_didactic::{
    execute_factorized_zero_product_strategy_if_applicable_with_state,
    execute_quadratic_main_didactic_pipeline_with_default_execution_with_state,
    QuadraticExecutionItem, QuadraticSubstepExecutionItem, ZeroProductFactorExecutionItem,
};
use crate::quadratic_formula::execute_quadratic_coefficient_solve_plan_with_default_numeric_solution_with_state;
use crate::{isolation_utils::is_numeric_zero, quadratic_coeffs::extract_quadratic_coefficients};

/// Prepared quadratic candidate for `a*x^2 + b*x + c = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuadraticPreparedCandidate {
    pub sim_poly_expr: ExprId,
    pub a: ExprId,
    pub b: ExprId,
    pub c: ExprId,
}

/// Prepare quadratic strategy candidate with default coefficient extraction:
/// 1) normalize equation to `lhs - rhs`,
/// 2) simplify + expand for coefficient extraction,
/// 3) extract/simplify `(a, b, c)`,
/// 4) reject degenerate `a == 0`.
pub fn prepare_quadratic_candidate_from_simplified_polynomial_with_default_coefficient_extraction_with_state<
    T,
    FContextMut,
    FSimplify,
    FExpand,
>(
    state: &mut T,
    sim_poly_expr: ExprId,
    var: &str,
    context_mut: FContextMut,
    mut simplify_expr: FSimplify,
    mut expand_expr: FExpand,
) -> Option<(ExprId, ExprId, ExprId)>
where
    FContextMut: Fn(&mut T) -> &mut Context,
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FExpand: FnMut(&mut T, ExprId) -> ExprId,
{
    let context_mut = &context_mut;
    let expanded_expr = expand_expr(state, sim_poly_expr);
    let (a, b, c) = extract_quadratic_coefficients(context_mut(state), expanded_expr, var)?;

    let sim_a = simplify_expr(state, a);
    let sim_b = simplify_expr(state, b);
    let sim_c = simplify_expr(state, c);
    if is_numeric_zero(context_mut(state), sim_a) {
        return None;
    }

    Some((sim_a, sim_b, sim_c))
}

/// Prepare quadratic strategy candidate with default coefficient extraction:
/// 1) normalize equation to `lhs - rhs`,
/// 2) simplify the polynomial form,
/// 3) extract/simplify `(a, b, c)`,
/// 4) reject degenerate `a == 0`.
pub fn prepare_quadratic_strategy_candidate_with_default_coefficient_extraction_with_state<
    T,
    FContextMut,
    FSimplify,
    FExpand,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    context_mut: FContextMut,
    mut simplify_expr: FSimplify,
    mut expand_expr: FExpand,
) -> Option<QuadraticPreparedCandidate>
where
    FContextMut: Fn(&mut T) -> &mut Context,
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FExpand: FnMut(&mut T, ExprId) -> ExprId,
{
    let context_mut = &context_mut;
    let poly_expr = context_mut(state).add(Expr::Sub(equation.lhs, equation.rhs));
    let sim_poly_expr = simplify_expr(state, poly_expr);
    let (sim_a, sim_b, sim_c) =
        prepare_quadratic_candidate_from_simplified_polynomial_with_default_coefficient_extraction_with_state(
            state,
            sim_poly_expr,
            var,
            |state| context_mut(state),
            |state, expr| simplify_expr(state, expr),
            |state, expr| expand_expr(state, expr),
        )?;

    Some(QuadraticPreparedCandidate {
        sim_poly_expr,
        a: sim_a,
        b: sim_b,
        c: sim_c,
    })
}

/// Execute quadratic strategy after `(a, b, c)` are already extracted/simplified:
/// 1) build optional didactic main/substeps,
/// 2) solve coefficient plan with default numeric kernel.
#[allow(clippy::too_many_arguments)]
pub fn execute_quadratic_coefficient_pipeline_with_default_didactic_and_numeric_solution_with_state<
    T,
    S,
    SS,
    E,
    FContextMut,
    FSetCollect,
    FSimplify,
    FRender,
    FMapMain,
    FMapSub,
    FExpand,
    FMapPlanError,
>(
    state: &mut T,
    var: &str,
    op: RelOp,
    lhs_expr_after_simplify: ExprId,
    a: ExprId,
    b: ExprId,
    c: ExprId,
    is_real_only: bool,
    include_items: bool,
    was_collecting: bool,
    context_mut: FContextMut,
    set_collecting: FSetCollect,
    mut simplify_expr: FSimplify,
    render_expr: FRender,
    map_main_item_to_step: FMapMain,
    map_substep_item_to_step: FMapSub,
    mut expand_expr: FExpand,
    map_plan_error: FMapPlanError,
) -> Result<(SolutionSet, Vec<S>), E>
where
    SS: Clone,
    FContextMut: Fn(&mut T) -> &mut Context,
    FSetCollect: FnMut(&mut T, bool),
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FRender: Fn(&Context, ExprId) -> String,
    FMapMain: FnMut(QuadraticExecutionItem, Vec<SS>) -> S,
    FMapSub: FnMut(QuadraticSubstepExecutionItem) -> SS,
    FExpand: FnMut(&mut T, ExprId) -> ExprId,
    FMapPlanError: FnOnce(crate::quadratic_formula::QuadraticCoefficientSolvePlanError) -> E,
{
    let context_mut = &context_mut;
    let main_equation = Equation {
        lhs: lhs_expr_after_simplify,
        rhs: context_mut(state).num(0),
        op: RelOp::Eq,
    };

    let steps = execute_quadratic_main_didactic_pipeline_with_default_execution_with_state(
        state,
        var,
        a,
        b,
        c,
        is_real_only,
        main_equation,
        include_items,
        was_collecting,
        |state| context_mut(state),
        set_collecting,
        |state, expr| simplify_expr(state, expr),
        render_expr,
        map_main_item_to_step,
        map_substep_item_to_step,
    );

    let solution_set =
        execute_quadratic_coefficient_solve_plan_with_default_numeric_solution_with_state(
            state,
            op,
            a,
            b,
            c,
            |state| context_mut(state),
            |state, expr| expand_expr(state, expr),
            |state, expr| simplify_expr(state, expr),
            map_plan_error,
        )?;

    Ok((solution_set, steps))
}

/// Execute quadratic candidate pipeline in one call:
/// 1) prepare a valid quadratic candidate (`a != 0`) from equation,
/// 2) if prepared, run didactic + coefficient-plan solving pipeline.
#[allow(clippy::too_many_arguments)]
pub fn execute_quadratic_strategy_candidate_pipeline_with_default_coefficient_extraction_and_numeric_solution_with_state<
    T,
    S,
    SS,
    E,
    FContextMut,
    FSetCollect,
    FSimplify,
    FExpand,
    FRender,
    FMapMain,
    FMapSub,
    FMapPlanError,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    is_real_only: bool,
    include_items: bool,
    was_collecting: bool,
    context_mut: FContextMut,
    set_collecting: FSetCollect,
    mut simplify_expr: FSimplify,
    mut expand_expr: FExpand,
    render_expr: FRender,
    map_main_item_to_step: FMapMain,
    map_substep_item_to_step: FMapSub,
    map_plan_error: FMapPlanError,
) -> Result<Option<(SolutionSet, Vec<S>)>, E>
where
    SS: Clone,
    FContextMut: Fn(&mut T) -> &mut Context,
    FSetCollect: FnMut(&mut T, bool),
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FExpand: FnMut(&mut T, ExprId) -> ExprId,
    FRender: Fn(&Context, ExprId) -> String,
    FMapMain: FnMut(QuadraticExecutionItem, Vec<SS>) -> S,
    FMapSub: FnMut(QuadraticSubstepExecutionItem) -> SS,
    FMapPlanError: FnOnce(crate::quadratic_formula::QuadraticCoefficientSolvePlanError) -> E,
{
    let context_mut = &context_mut;
    let prepared =
        prepare_quadratic_strategy_candidate_with_default_coefficient_extraction_with_state(
            state,
            equation,
            var,
            |state| context_mut(state),
            |state, expr| simplify_expr(state, expr),
            |state, expr| expand_expr(state, expr),
        );
    let Some(prepared) = prepared else {
        return Ok(None);
    };

    let solved =
        execute_quadratic_coefficient_pipeline_with_default_didactic_and_numeric_solution_with_state(
            state,
            var,
            equation.op.clone(),
            prepared.sim_poly_expr,
            prepared.a,
            prepared.b,
            prepared.c,
            is_real_only,
            include_items,
            was_collecting,
            |state| context_mut(state),
            set_collecting,
            |state, expr| simplify_expr(state, expr),
            render_expr,
            map_main_item_to_step,
            map_substep_item_to_step,
            |state, expr| expand_expr(state, expr),
            map_plan_error,
        )?;

    Ok(Some(solved))
}

/// Execute full quadratic strategy with default pipelines:
/// 1) normalize and simplify `lhs - rhs`,
/// 2) try factorized zero-product pipeline (if applicable),
/// 3) otherwise prepare/solve quadratic coefficient pipeline.
///
/// Returns:
/// - `Some(Ok(...))` when solved (factorized or coefficient path),
/// - `Some(Err(...))` when a selected path errors,
/// - `None` when strategy does not apply.
#[allow(clippy::too_many_arguments)]
pub fn execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_with_state<
    T,
    S,
    SS,
    E,
    FContextRef,
    FContextMut,
    FSetCollect,
    FSimplify,
    FExpand,
    FRender,
    FSolveFactor,
    FMapFactorizedEntry,
    FMapFactorizedFactor,
    FMapMain,
    FMapSub,
    FMapPlanError,
    FOnQuadraticCoefficientPathSolved,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    include_items: bool,
    is_real_only: bool,
    context_ref: FContextRef,
    context_mut: FContextMut,
    set_collecting: FSetCollect,
    mut simplify_expr: FSimplify,
    mut expand_expr: FExpand,
    render_expr: FRender,
    mut solve_factor: FSolveFactor,
    map_factorized_entry_item_to_step: FMapFactorizedEntry,
    map_factorized_factor_item_to_step: FMapFactorizedFactor,
    map_main_item_to_step: FMapMain,
    map_substep_item_to_step: FMapSub,
    map_plan_error: FMapPlanError,
    mut on_quadratic_coefficient_path_solved: FOnQuadraticCoefficientPathSolved,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    SS: Clone,
    FContextRef: Fn(&mut T) -> &Context,
    FContextMut: Fn(&mut T) -> &mut Context,
    FSetCollect: FnMut(&mut T, bool),
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FExpand: FnMut(&mut T, ExprId) -> ExprId,
    FRender: Fn(&Context, ExprId) -> String,
    FSolveFactor: FnMut(&mut T, &Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapFactorizedEntry: FnMut(QuadraticExecutionItem) -> S,
    FMapFactorizedFactor: FnMut(ZeroProductFactorExecutionItem) -> S,
    FMapMain: FnMut(QuadraticExecutionItem, Vec<SS>) -> S,
    FMapSub: FnMut(QuadraticSubstepExecutionItem) -> SS,
    FMapPlanError: FnOnce(crate::quadratic_formula::QuadraticCoefficientSolvePlanError) -> E,
    FOnQuadraticCoefficientPathSolved: FnMut(&mut T),
{
    let context_ref = &context_ref;
    let context_mut = &context_mut;
    let render_expr = &render_expr;
    let poly_expr = context_mut(state).add(Expr::Sub(equation.lhs, equation.rhs));
    let sim_poly_expr = simplify_expr(state, poly_expr);
    let zero = context_mut(state).num(0);

    if let Some(outcome) = execute_factorized_zero_product_strategy_if_applicable_with_state(
        state,
        equation.op.clone(),
        sim_poly_expr,
        var,
        zero,
        include_items,
        |state| context_ref(state),
        |ctx, id| render_expr(ctx, id),
        |state, factor_equation| solve_factor(state, factor_equation),
        map_factorized_entry_item_to_step,
        map_factorized_factor_item_to_step,
    ) {
        return Some(outcome);
    }

    let (a, b, c) =
        prepare_quadratic_candidate_from_simplified_polynomial_with_default_coefficient_extraction_with_state(
            state,
            sim_poly_expr,
            var,
            |state| context_mut(state),
            |state, expr| simplify_expr(state, expr),
            |state, expr| expand_expr(state, expr),
        )?;

    let solved =
        execute_quadratic_coefficient_pipeline_with_default_didactic_and_numeric_solution_with_state(
            state,
            var,
            equation.op.clone(),
            sim_poly_expr,
            a,
            b,
            c,
            is_real_only,
            include_items,
            include_items,
            |state| context_mut(state),
            set_collecting,
            |state, expr| simplify_expr(state, expr),
            |ctx, id| render_expr(ctx, id),
            map_main_item_to_step,
            map_substep_item_to_step,
            |state, expr| expand_expr(state, expr),
            map_plan_error,
        );

    match solved {
        Ok(outcome) => {
            on_quadratic_coefficient_path_solved(state);
            Some(Ok(outcome))
        }
        Err(err) => Some(Err(err)),
    }
}

/// Execute full quadratic strategy with default pipelines and unified step
/// mappers for both main steps and substeps.
#[allow(clippy::too_many_arguments)]
pub fn execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_and_unified_step_mappers_with_state<
    T,
    S,
    SS,
    E,
    FContextRef,
    FContextMut,
    FSetCollect,
    FSimplify,
    FExpand,
    FRender,
    FSolveFactor,
    FMapStep,
    FMapSubstep,
    FMapPlanError,
    FOnQuadraticCoefficientPathSolved,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    include_items: bool,
    is_real_only: bool,
    context_ref: FContextRef,
    context_mut: FContextMut,
    set_collecting: FSetCollect,
    simplify_expr: FSimplify,
    expand_expr: FExpand,
    render_expr: FRender,
    solve_factor: FSolveFactor,
    map_step: FMapStep,
    map_substep: FMapSubstep,
    map_plan_error: FMapPlanError,
    on_quadratic_coefficient_path_solved: FOnQuadraticCoefficientPathSolved,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    SS: Clone,
    FContextRef: Fn(&mut T) -> &Context,
    FContextMut: Fn(&mut T) -> &mut Context,
    FSetCollect: FnMut(&mut T, bool),
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FExpand: FnMut(&mut T, ExprId) -> ExprId,
    FRender: Fn(&Context, ExprId) -> String,
    FSolveFactor: FnMut(&mut T, &Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation, Option<Vec<SS>>) -> S,
    FMapSubstep: FnMut(String, Equation) -> SS,
    FMapPlanError: FnOnce(crate::quadratic_formula::QuadraticCoefficientSolvePlanError) -> E,
    FOnQuadraticCoefficientPathSolved: FnMut(&mut T),
{
    let map_step = std::cell::RefCell::new(map_step);
    let map_substep = std::cell::RefCell::new(map_substep);
    execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_with_state(
        state,
        equation,
        var,
        include_items,
        is_real_only,
        context_ref,
        context_mut,
        set_collecting,
        simplify_expr,
        expand_expr,
        render_expr,
        solve_factor,
        |item| (map_step.borrow_mut())(item.description().to_string(), item.equation, None),
        |item| (map_step.borrow_mut())(item.description, item.equation, None),
        |item, substeps| {
            (map_step.borrow_mut())(
                item.description().to_string(),
                item.equation,
                Some(substeps),
            )
        },
        |item| (map_substep.borrow_mut())(item.description, item.equation),
        map_plan_error,
        on_quadratic_coefficient_path_solved,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quadratic_formula::QuadraticCoefficientSolvePlanError;
    use cas_ast::{Context, Expr};

    #[test]
    fn execute_quadratic_coefficient_pipeline_with_default_didactic_and_numeric_solution_with_state_solves_numeric_eq(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x2 = ctx.add(Expr::Pow(x, two));
        let three_x = ctx.add(Expr::Mul(three, x));
        let lhs = ctx.add(Expr::Sub(x2, three_x));
        let lhs = ctx.add(Expr::Add(lhs, two));

        let a = ctx.num(1);
        let b = ctx.num(-3);
        let c = ctx.num(2);

        let solved =
            execute_quadratic_coefficient_pipeline_with_default_didactic_and_numeric_solution_with_state(
                &mut ctx,
                "x",
                RelOp::Eq,
                lhs,
                a,
                b,
                c,
                true,
                false,
                false,
                |ctx| ctx,
                |_ctx, _collecting| {},
                |_ctx, id| id,
                |_ctx, _id| String::new(),
                |_item, _substeps| String::new(),
                |_item| String::new(),
                |_ctx, id| id,
                |_err| "plan error".to_string(),
            )
            .expect("numeric quadratic should solve");

        assert!(matches!(solved.0, SolutionSet::Discrete(_)));
        assert!(solved.1.is_empty());
    }

    #[test]
    fn execute_quadratic_coefficient_pipeline_with_default_didactic_and_numeric_solution_with_state_maps_plan_error(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let lhs = ctx.add(Expr::Pow(x, two));
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        let err =
            execute_quadratic_coefficient_pipeline_with_default_didactic_and_numeric_solution_with_state(
                &mut ctx,
                "x",
                RelOp::Lt,
                lhs,
                a,
                b,
                c,
                true,
                false,
                false,
                |ctx| ctx,
                |_ctx, _collecting| {},
                |_ctx, id| id,
                |_ctx, _id| String::new(),
                |_item, _substeps| String::new(),
                |_item| String::new(),
                |_ctx, id| id,
                |plan_err| plan_err,
            )
            .expect_err("symbolic inequality must map plan error");

        assert_eq!(
            err,
            QuadraticCoefficientSolvePlanError::UnsupportedSymbolicInequality
        );
    }

    #[test]
    fn prepare_quadratic_strategy_candidate_with_default_coefficient_extraction_with_state_accepts_quadratic(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let one = ctx.num(1);
        let x2 = ctx.add(Expr::Pow(x, two));
        let three_x = ctx.add(Expr::Mul(three, x));
        let lhs = ctx.add(Expr::Add(x2, three_x));
        let lhs = ctx.add(Expr::Add(lhs, one));
        let equation = Equation {
            lhs,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };

        let prepared =
            prepare_quadratic_strategy_candidate_with_default_coefficient_extraction_with_state(
                &mut ctx,
                &equation,
                "x",
                |ctx| ctx,
                |_ctx, id| id,
                |_ctx, id| id,
            )
            .expect("must prepare quadratic candidate");

        assert!(!is_numeric_zero(&ctx, prepared.a));
    }

    #[test]
    fn prepare_quadratic_strategy_candidate_with_default_coefficient_extraction_with_state_rejects_linear(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Add(x, one));
        let equation = Equation {
            lhs,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };

        let prepared =
            prepare_quadratic_strategy_candidate_with_default_coefficient_extraction_with_state(
                &mut ctx,
                &equation,
                "x",
                |ctx| ctx,
                |_ctx, id| id,
                |_ctx, id| id,
            );

        assert!(prepared.is_none());
    }

    #[test]
    fn execute_quadratic_strategy_candidate_pipeline_with_default_coefficient_extraction_and_numeric_solution_with_state_returns_none_for_non_quadratic(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let equation = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };

        let solved = execute_quadratic_strategy_candidate_pipeline_with_default_coefficient_extraction_and_numeric_solution_with_state(
            &mut ctx,
            &equation,
            "x",
            true,
            false,
            false,
            |ctx| ctx,
            |_ctx, _collecting| {},
            |_ctx, id| id,
            |_ctx, id| id,
            |_ctx, _id| String::new(),
            |_item, _substeps| String::new(),
            |_item| String::new(),
            |_err| "plan error".to_string(),
        )
        .expect("pipeline should not error for non-quadratic");

        assert!(solved.is_none());
    }

    #[test]
    fn prepare_quadratic_candidate_from_simplified_polynomial_with_default_coefficient_extraction_with_state_rejects_linear(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let sim_poly_expr = ctx.add(Expr::Add(x, one));

        let prepared =
            prepare_quadratic_candidate_from_simplified_polynomial_with_default_coefficient_extraction_with_state(
                &mut ctx,
                sim_poly_expr,
                "x",
                |ctx| ctx,
                |_ctx, id| id,
                |_ctx, id| id,
            );

        assert!(prepared.is_none());
    }

    #[test]
    fn execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_with_state_solves_factorized_path(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let equation = Equation {
            lhs: ctx.add(Expr::Mul(x, x_minus_one)),
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };

        let solved =
            execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_with_state(
                &mut ctx,
                &equation,
                "x",
                false,
                true,
                |ctx| ctx,
                |ctx| ctx,
                |_ctx, _collecting| {},
                |ctx, id| match ctx.get(id) {
                    Expr::Sub(lhs, rhs) if is_numeric_zero(ctx, *rhs) => *lhs,
                    _ => id,
                },
                |_ctx, id| id,
                |_ctx, _id| String::new(),
                |_ctx, factor_equation| {
                    Ok::<_, String>((SolutionSet::Discrete(vec![factor_equation.lhs]), vec![]))
                },
                |_item| String::new(),
                |_item| String::new(),
                |_item, _substeps| String::new(),
                |_item| String::new(),
                |_err| "plan error".to_string(),
                |_ctx| {},
            )
            .expect("strategy should apply")
            .expect("factorized path should solve");

        assert!(matches!(solved.0, SolutionSet::Discrete(_)));
    }

    #[test]
    fn execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_with_state_solves_quadratic_path(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x2 = ctx.add(Expr::Pow(x, two));
        let three_x = ctx.add(Expr::Mul(three, x));
        let lhs = ctx.add(Expr::Sub(x2, three_x));
        let lhs = ctx.add(Expr::Add(lhs, two));
        let equation = Equation {
            lhs,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };

        let solved =
            execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_with_state(
                &mut ctx,
                &equation,
                "x",
                false,
                true,
                |ctx| ctx,
                |ctx| ctx,
                |_ctx, _collecting| {},
                |_ctx, id| id,
                |_ctx, id| id,
                |_ctx, _id| String::new(),
                |_ctx, _factor_equation| -> Result<(SolutionSet, Vec<String>), String> {
                    panic!("factorized solver should not run for additive quadratic")
                },
                |_item| String::new(),
                |_item| String::new(),
                |_item, _substeps| String::new(),
                |_item| String::new(),
                |_err| "plan error".to_string(),
                |_ctx| {},
            )
            .expect("strategy should apply")
            .expect("quadratic path should solve");

        assert!(matches!(solved.0, SolutionSet::Discrete(_)));
    }
}
