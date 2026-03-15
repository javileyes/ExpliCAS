//! Shared verification runtime orchestration for integration crates.
//!
//! This keeps conservative-domain simplify wiring out of runtime wrappers
//! (`cas_engine`, `cas_solver`) so they only provide state-specific kernels.

use crate::domain_mode::DomainMode;
use crate::verification::{VerifyResult, VerifyStatus};
use cas_ast::{
    views::as_rational_const, BoundType, BuiltinFn, Constant, Context, Equation, Expr, ExprId,
    Interval, SolutionSet,
};
use num_rational::BigRational;
use num_traits::Zero;
use std::collections::HashMap;

/// Verify one candidate solution using conservative domain-mode simplify wiring.
#[allow(clippy::too_many_arguments)]
pub fn verify_solution_with_conservative_domain_simplify_with_state<
    T,
    FSubstituteDiff,
    FSimplifyWithOptions,
    FContainsVariable,
    FFoldNumericIslands,
    FIsZero,
    FRenderExpr,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    solution: ExprId,
    substitute_diff: FSubstituteDiff,
    mut simplify_with_options: FSimplifyWithOptions,
    contains_variable: FContainsVariable,
    fold_numeric_islands: FFoldNumericIslands,
    is_zero: FIsZero,
    render_expr: FRenderExpr,
) -> VerifyStatus
where
    FSubstituteDiff: FnMut(&mut T, &Equation, &str, ExprId) -> ExprId,
    FSimplifyWithOptions: FnMut(&mut T, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId,
    FContainsVariable: FnMut(&mut T, ExprId) -> bool,
    FFoldNumericIslands: FnMut(&mut T, ExprId) -> ExprId,
    FIsZero: FnMut(&mut T, ExprId) -> bool,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
{
    crate::verification_flow::verify_solution_with_domain_modes_with_state(
        state,
        equation,
        var,
        solution,
        substitute_diff,
        |state, expr, domain_mode: DomainMode| {
            let opts = crate::conservative_eval_config::simplify_options_for_domain(domain_mode);
            simplify_with_options(state, expr, opts)
        },
        contains_variable,
        fold_numeric_islands,
        is_zero,
        render_expr,
    )
}

/// Verify one candidate solution using shared runtime kernels:
/// - domain-mode simplify options from conservative config
/// - numeric-island fold with default guard/limits
/// - variable/zero/render checks from caller-provided context accessors.
#[allow(clippy::too_many_arguments)]
pub fn verify_solution_with_runtime_kernels_with_state<
    T,
    FContext,
    FContextMut,
    FSimplifyWithOptions,
    FGroundEvalCandidate,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    solution: ExprId,
    context: FContext,
    context_mut: FContextMut,
    simplify_with_options: FSimplifyWithOptions,
    ground_eval_candidate: FGroundEvalCandidate,
) -> VerifyStatus
where
    FContext: Fn(&T) -> &cas_ast::Context + Copy,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context + Copy,
    FSimplifyWithOptions:
        Fn(&mut T, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId + Copy,
    FGroundEvalCandidate: Fn(
            &cas_ast::Context,
            ExprId,
            &crate::simplify_options::SimplifyOptions,
        ) -> Option<(cas_ast::Context, ExprId)>
        + Copy,
{
    let mut status = verify_solution_with_conservative_domain_simplify_with_state(
        state,
        equation,
        var,
        solution,
        |state, eq, solve_var, candidate| {
            crate::verify_substitution::substitute_equation_diff(
                context_mut(state),
                eq,
                solve_var,
                candidate,
            )
        },
        simplify_with_options,
        |state, expr| cas_math::expr_predicates::contains_variable(context(state), expr),
        |state, root| {
            crate::verification_runtime_helpers::fold_numeric_islands_with_default_guard_and_conservative_options(
                context_mut(state),
                root,
                ground_eval_candidate,
            )
        },
        |state, expr| crate::isolation_utils::is_numeric_zero(context(state), expr),
        |state, expr| cas_formatter::render_expr(context(state), expr),
    );

    attach_counterexample_hint(context(state), &mut status);
    status
}

/// Verify a full solution set using the same runtime kernels as
/// [`verify_solution_with_runtime_kernels_with_state`].
#[allow(clippy::too_many_arguments)]
pub fn verify_solution_set_with_runtime_kernels_with_state<
    T,
    FContext,
    FContextMut,
    FSimplifyWithOptions,
    FGroundEvalCandidate,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    solutions: &cas_ast::SolutionSet,
    context: FContext,
    context_mut: FContextMut,
    simplify_with_options: FSimplifyWithOptions,
    ground_eval_candidate: FGroundEvalCandidate,
) -> VerifyResult
where
    FContext: Fn(&T) -> &cas_ast::Context + Copy,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context + Copy,
    FSimplifyWithOptions:
        Fn(&mut T, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId + Copy,
    FGroundEvalCandidate: Fn(
            &cas_ast::Context,
            ExprId,
            &crate::simplify_options::SimplifyOptions,
        ) -> Option<(cas_ast::Context, ExprId)>
        + Copy,
{
    let mut result = crate::verification_flow::verify_solution_set_for_equation_with_state(
        state,
        equation,
        var,
        solutions,
        |state, equation, var, solution| {
            verify_solution_with_runtime_kernels_with_state(
                state,
                equation,
                var,
                solution,
                context,
                context_mut,
                simplify_with_options,
                ground_eval_candidate,
            )
        },
    );

    refine_non_discrete_guard_description(context(state), var, solutions, &mut result);
    result
}

fn refine_non_discrete_guard_description(
    ctx: &Context,
    var: &str,
    solutions: &SolutionSet,
    result: &mut VerifyResult,
) {
    if !matches!(
        result.summary,
        crate::verification::VerifySummary::NeedsSampling
    ) {
        return;
    }

    if let Some(guard_text) = native_guard_text(ctx, var, solutions) {
        result.guard_description = Some(format!(
            "verification requires numeric sampling (solution set matches guard {guard_text})"
        ));
    }
}

fn native_guard_text(ctx: &Context, var: &str, solutions: &SolutionSet) -> Option<String> {
    match solutions {
        SolutionSet::Continuous(interval) => native_guard_text_for_interval(ctx, var, interval),
        SolutionSet::Union(intervals) => native_guard_text_for_union(ctx, var, intervals),
        _ => None,
    }
}

fn native_guard_text_for_interval(ctx: &Context, var: &str, interval: &Interval) -> Option<String> {
    if is_zero(ctx, interval.min)
        && interval.min_type == BoundType::Open
        && is_pos_infinity(ctx, interval.max)
    {
        return Some(format!("`{var} > 0`"));
    }
    if is_zero(ctx, interval.min)
        && interval.min_type == BoundType::Closed
        && is_pos_infinity(ctx, interval.max)
    {
        return Some(format!("`{var} >= 0`"));
    }
    if is_neg_infinity(ctx, interval.min)
        && is_zero(ctx, interval.max)
        && interval.max_type == BoundType::Open
    {
        return Some(format!("`{var} < 0`"));
    }
    if is_neg_infinity(ctx, interval.min)
        && is_zero(ctx, interval.max)
        && interval.max_type == BoundType::Closed
    {
        return Some(format!("`{var} <= 0`"));
    }
    None
}

fn native_guard_text_for_union(ctx: &Context, var: &str, intervals: &[Interval]) -> Option<String> {
    if intervals.len() != 2 {
        return None;
    }

    let first = (&intervals[0], &intervals[1]);
    let second = (&intervals[1], &intervals[0]);
    if matches_nonzero_union(ctx, first.0, first.1)
        || matches_nonzero_union(ctx, second.0, second.1)
    {
        return Some(format!("`{var} != 0`"));
    }

    None
}

fn matches_nonzero_union(ctx: &Context, left: &Interval, right: &Interval) -> bool {
    is_neg_infinity(ctx, left.min)
        && left.min_type == BoundType::Open
        && is_zero(ctx, left.max)
        && left.max_type == BoundType::Open
        && is_zero(ctx, right.min)
        && right.min_type == BoundType::Open
        && is_pos_infinity(ctx, right.max)
        && right.max_type == BoundType::Open
}

fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
}

fn is_pos_infinity(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Constant(Constant::Infinity))
}

fn is_neg_infinity(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Neg(inner) => matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)),
        _ => false,
    }
}

fn attach_counterexample_hint(ctx: &Context, status: &mut VerifyStatus) {
    if let VerifyStatus::Unverifiable {
        residual,
        counterexample_hint,
        ..
    } = status
    {
        if counterexample_hint.is_none() {
            *counterexample_hint = try_counterexample_hint(ctx, *residual);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CounterexampleHintSuppressionReason {
    Log,
    Sqrt,
    InverseTrig,
}

pub fn counterexample_hint_suppression_note(
    ctx: &Context,
    residual: ExprId,
) -> Option<&'static str> {
    match counterexample_hint_suppression_reason(ctx, residual) {
        Some(CounterexampleHintSuppressionReason::Log) => {
            Some("counterexample hint omitted for branch-sensitive residual (`log/ln`)")
        }
        Some(CounterexampleHintSuppressionReason::Sqrt) => {
            Some("counterexample hint omitted for branch-sensitive residual (`sqrt`)")
        }
        Some(CounterexampleHintSuppressionReason::InverseTrig) => {
            Some("counterexample hint omitted for branch-sensitive residual (`inverse trig`)")
        }
        None => None,
    }
}

fn try_counterexample_hint(ctx: &Context, residual: ExprId) -> Option<String> {
    if counterexample_hint_suppression_reason(ctx, residual).is_some() {
        return None;
    }

    let mut vars: Vec<String> = cas_ast::collect_variables(ctx, residual)
        .into_iter()
        .collect();
    if vars.is_empty() {
        return None;
    }
    vars.sort();
    vars.dedup();

    for probe in [0.0_f64, 1.0, 2.0, -1.0] {
        let assignments: HashMap<String, f64> =
            vars.iter().cloned().map(|var| (var, probe)).collect();
        let value = match cas_math::evaluator_f64::eval_f64(ctx, residual, &assignments) {
            Some(value) if value.is_finite() => value,
            _ => continue,
        };
        if crate::equivalence::is_numeric_equiv_zero(value) {
            continue;
        }

        let probe_text = format_probe_literal(probe);
        let assignment_text = vars
            .iter()
            .map(|var| format!("{var}={probe_text}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Some(format!(
            "counterexample hint: {assignment_text} gives residual {}",
            format_probe_value(value)
        ));
    }

    None
}

fn counterexample_hint_suppression_reason(
    ctx: &Context,
    expr: ExprId,
) -> Option<CounterexampleHintSuppressionReason> {
    let half = BigRational::new(1.into(), 2.into());
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => None,
        Expr::Neg(inner) | Expr::Hold(inner) => counterexample_hint_suppression_reason(ctx, *inner),
        Expr::Pow(base, exp) => {
            if let Some(exp_val) = as_rational_const(ctx, *exp, 8) {
                if exp_val == half {
                    return Some(CounterexampleHintSuppressionReason::Sqrt);
                }
            }
            counterexample_hint_suppression_reason(ctx, *base)
                .or_else(|| counterexample_hint_suppression_reason(ctx, *exp))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            counterexample_hint_suppression_reason(ctx, *l)
                .or_else(|| counterexample_hint_suppression_reason(ctx, *r))
        }
        Expr::Function(fn_id, args) => {
            let builtin_reason = if ctx.is_builtin(*fn_id, BuiltinFn::Ln)
                || ctx.is_builtin(*fn_id, BuiltinFn::Log)
            {
                Some(CounterexampleHintSuppressionReason::Log)
            } else if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) {
                Some(CounterexampleHintSuppressionReason::Sqrt)
            } else if [
                BuiltinFn::Asin,
                BuiltinFn::Acos,
                BuiltinFn::Atan,
                BuiltinFn::Asec,
                BuiltinFn::Acsc,
                BuiltinFn::Acot,
                BuiltinFn::Arcsin,
                BuiltinFn::Arccos,
                BuiltinFn::Arctan,
                BuiltinFn::Arcsec,
                BuiltinFn::Arccsc,
                BuiltinFn::Arccot,
            ]
            .into_iter()
            .any(|builtin| ctx.is_builtin(*fn_id, builtin))
            {
                Some(CounterexampleHintSuppressionReason::InverseTrig)
            } else {
                None
            };

            builtin_reason.or_else(|| {
                args.iter()
                    .find_map(|arg| counterexample_hint_suppression_reason(ctx, *arg))
            })
        }
        Expr::Matrix { data, .. } => data
            .iter()
            .find_map(|arg| counterexample_hint_suppression_reason(ctx, *arg)),
    }
}

fn format_probe_literal(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}

fn format_probe_value(value: f64) -> String {
    let rounded = value.round();
    if (value - rounded).abs() < crate::equivalence::DEFAULT_EQUIV_NUMERIC_EPS {
        format!("{}", rounded as i64)
    } else {
        format!("{value:.6}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_guard_text_identifies_positive_half_line() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let inf = ctx.add(Expr::Constant(Constant::Infinity));
        let interval = Interval {
            min: zero,
            min_type: BoundType::Open,
            max: inf,
            max_type: BoundType::Open,
        };

        assert_eq!(
            native_guard_text_for_interval(&ctx, "x", &interval).as_deref(),
            Some("`x > 0`")
        );
    }

    #[test]
    fn native_guard_text_identifies_nonnegative_half_line() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let inf = ctx.add(Expr::Constant(Constant::Infinity));
        let interval = Interval {
            min: zero,
            min_type: BoundType::Closed,
            max: inf,
            max_type: BoundType::Open,
        };

        assert_eq!(
            native_guard_text_for_interval(&ctx, "x", &interval).as_deref(),
            Some("`x >= 0`")
        );
    }

    #[test]
    fn native_guard_text_identifies_nonzero_union() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let inf = ctx.add(Expr::Constant(Constant::Infinity));
        let neg_inf = ctx.add(Expr::Neg(inf));
        let solutions = SolutionSet::Union(vec![
            Interval {
                min: neg_inf,
                min_type: BoundType::Open,
                max: zero,
                max_type: BoundType::Open,
            },
            Interval {
                min: zero,
                min_type: BoundType::Open,
                max: inf,
                max_type: BoundType::Open,
            },
        ]);

        assert_eq!(
            native_guard_text(&ctx, "x", &solutions).as_deref(),
            Some("`x != 0`")
        );
    }

    #[test]
    fn refine_non_discrete_guard_description_rewrites_needs_sampling_message() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let inf = ctx.add(Expr::Constant(Constant::Infinity));
        let solutions = SolutionSet::Continuous(Interval {
            min: zero,
            min_type: BoundType::Open,
            max: inf,
            max_type: BoundType::Open,
        });
        let mut result = VerifyResult {
            solutions: vec![],
            summary: crate::verification::VerifySummary::NeedsSampling,
            guard_description: Some(
                "verification requires numeric sampling (continuous interval)".to_string(),
            ),
        };

        refine_non_discrete_guard_description(&ctx, "x", &solutions, &mut result);
        assert_eq!(
            result.guard_description.as_deref(),
            Some("verification requires numeric sampling (solution set matches guard `x > 0`)")
        );
    }

    #[test]
    fn try_counterexample_hint_uses_small_literal_probe() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let one = ctx.num(1);
        let residual = ctx.add(Expr::Sub(y, one));

        assert_eq!(
            try_counterexample_hint(&ctx, residual).as_deref(),
            Some("counterexample hint: y=0 gives residual -1")
        );
    }

    #[test]
    fn try_counterexample_hint_returns_none_when_probes_only_cancel() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let residual = ctx.add(Expr::Sub(y, y));

        assert_eq!(try_counterexample_hint(&ctx, residual), None);
    }

    #[test]
    fn try_counterexample_hint_returns_none_when_all_probes_are_invalid() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let zero = ctx.num(0);
        let denom = ctx.add(Expr::Sub(y, y));
        let residual = ctx.add(Expr::Div(zero, denom));

        assert_eq!(try_counterexample_hint(&ctx, residual), None);
    }

    #[test]
    fn try_counterexample_hint_suppresses_unary_log_residuals() {
        let mut ctx = Context::new();
        let residual = cas_parser::parse("ln(y) - 1", &mut ctx).expect("parse");

        assert_eq!(try_counterexample_hint(&ctx, residual), None);
        assert_eq!(
            counterexample_hint_suppression_reason(&ctx, residual),
            Some(CounterexampleHintSuppressionReason::Log)
        );
    }

    #[test]
    fn try_counterexample_hint_suppresses_sqrt_residuals() {
        let mut ctx = Context::new();
        let residual = cas_parser::parse("sqrt(y) - 1", &mut ctx).expect("parse");

        assert_eq!(try_counterexample_hint(&ctx, residual), None);
        assert_eq!(
            counterexample_hint_suppression_reason(&ctx, residual),
            Some(CounterexampleHintSuppressionReason::Sqrt)
        );
        assert_eq!(
            counterexample_hint_suppression_note(&ctx, residual),
            Some("counterexample hint omitted for branch-sensitive residual (`sqrt`)")
        );
    }

    #[test]
    fn try_counterexample_hint_suppresses_inverse_trig_residuals() {
        let mut ctx = Context::new();
        let residual = cas_parser::parse("arcsin(y) - 1", &mut ctx).expect("parse");

        assert_eq!(try_counterexample_hint(&ctx, residual), None);
        assert_eq!(
            counterexample_hint_suppression_reason(&ctx, residual),
            Some(CounterexampleHintSuppressionReason::InverseTrig)
        );
        assert_eq!(
            counterexample_hint_suppression_note(&ctx, residual),
            Some("counterexample hint omitted for branch-sensitive residual (`inverse trig`)")
        );
    }

    #[test]
    fn counterexample_hint_suppression_reason_finds_nested_sensitive_family() {
        let mut ctx = Context::new();
        let residual = cas_parser::parse("(y + 1) / sqrt(y)", &mut ctx).expect("parse");

        assert_eq!(
            counterexample_hint_suppression_reason(&ctx, residual),
            Some(CounterexampleHintSuppressionReason::Sqrt)
        );
    }
}
