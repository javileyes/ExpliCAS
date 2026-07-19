use crate::eval_input::{EvalNonSolveAction, PreparedEvalRequest};
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_extract::{extract_log_base_argument_view, log10_base_sentinel};
use cas_math::expr_predicates::contains_variable;
use std::collections::HashSet;

fn map_non_solve_action(action: EvalNonSolveAction) -> crate::EvalAction {
    match action {
        EvalNonSolveAction::Simplify => crate::EvalAction::Simplify,
        EvalNonSolveAction::Equiv { other } => crate::EvalAction::Equiv { other },
        EvalNonSolveAction::Limit { var, approach } => crate::EvalAction::Limit { var, approach },
        EvalNonSolveAction::Dsolve {
            func,
            var,
            conditions,
        } => crate::EvalAction::Dsolve {
            func,
            var,
            conditions,
        },
    }
}

/// Evaluate one prepared eval request with any eval session implementation.
pub(crate) fn evaluate_prepared_request_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    prepared: PreparedEvalRequest,
) -> Result<crate::EvalOutputView, String>
where
    S: crate::SolverEvalSession,
{
    let mut output_view = match prepared {
        PreparedEvalRequest::Solve {
            raw_input,
            parsed,
            original_equation,
            var,
            auto_store,
        } => crate::solve_command_eval_core::evaluate_solve_parsed_with_session(
            &mut engine.simplifier,
            session,
            raw_input,
            parsed,
            original_equation.as_ref(),
            &var,
            auto_store,
        ),
        PreparedEvalRequest::SolveSystem {
            parsed_anchor,
            exprs,
            vars,
            ..
        } => crate::linear_system_command_eval::evaluate_linear_system_eval_request_with_session(
            engine,
            session,
            parsed_anchor,
            exprs,
            vars,
        ),
        PreparedEvalRequest::Derive {
            raw_input,
            parsed,
            target,
            auto_store,
        } => crate::derive_command::evaluate_derive_request_with_session(
            engine, session, raw_input, parsed, target, auto_store,
        ),
        PreparedEvalRequest::Eval {
            raw_input,
            parsed,
            action,
            auto_store,
        } => {
            let req = crate::EvalRequest {
                raw_input,
                parsed,
                action: map_non_solve_action(action),
                auto_store,
            };
            let output = engine.eval(session, req).map_err(|e| e.to_string())?;
            Ok(crate::eval_output_view(&output))
        }
    }?;

    augment_output_view_required_conditions(&mut engine.simplifier.context, &mut output_view);
    Ok(output_view)
}

pub(crate) fn augment_output_view_required_conditions(
    ctx: &mut Context,
    output_view: &mut crate::EvalOutputView,
) {
    let input_roots = [output_view.parsed, output_view.resolved];
    let input_conditions = collect_general_log_base_nonunit_requires(ctx, &input_roots);
    output_view
        .diagnostics
        .extend_required(input_conditions, crate::RequireOrigin::InputImplicit);

    let result_exprs = result_expr_roots(&output_view.result);
    let output_conditions = collect_general_log_base_nonunit_requires(ctx, &result_exprs);
    output_view
        .diagnostics
        .extend_required(output_conditions, crate::RequireOrigin::OutputImplicit);

    output_view.diagnostics.dedup_and_sort(ctx);
    output_view.required_conditions = output_view.diagnostics.required_conditions();
}

fn result_expr_roots(result: &crate::EvalResult) -> Vec<ExprId> {
    match result {
        crate::EvalResult::Expr(expr) => vec![*expr],
        crate::EvalResult::Set(exprs) => exprs.clone(),
        _ => Vec::new(),
    }
}

fn collect_general_log_base_nonunit_requires(
    ctx: &mut Context,
    roots: &[ExprId],
) -> Vec<crate::ImplicitCondition> {
    let one = ctx.num(1);
    let mut stack = roots.to_vec();
    let mut visited = HashSet::new();
    let mut conditions = Vec::new();

    while let Some(expr) = stack.pop() {
        if !visited.insert(expr) {
            continue;
        }

        if let Some((Some(base), _arg)) = extract_log_base_argument_view(ctx, expr) {
            if base != log10_base_sentinel() && contains_variable(ctx, base) {
                let base_minus_one = ctx.add(Expr::Sub(base, one));
                conditions.push(crate::ImplicitCondition::NonZero(base_minus_one));
            }
        }

        match ctx.get(expr).clone() {
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(left);
                stack.push(right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(inner),
            Expr::Function(_, args) => stack.extend(args),
            Expr::Matrix { data, .. } => stack.extend(data),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    conditions
}
