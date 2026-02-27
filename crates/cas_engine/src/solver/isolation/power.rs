use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use crate::SimplifyOptions;
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_power::execute_pow_isolation_with_default_kernels_for_var_with_state;

use super::isolate;

/// Handle isolation for `Pow(b, e)`: `B^E = RHS`
#[allow(clippy::too_many_arguments)]
pub(super) fn isolate_pow(
    lhs: ExprId,
    b: ExprId,
    e: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_base_item = simplifier.collect_steps();
    let include_shortcut_item = simplifier.collect_steps();
    let include_base_one_item = simplifier.collect_steps();
    let solve_tactic_enabled = opts.domain_mode == crate::domain::DomainMode::Assume
        && opts.value_domain == crate::semantics::ValueDomain::RealOnly;
    let tactic_opts = SimplifyOptions::for_solve_tactic(opts.domain_mode);
    let mode = opts.core_domain_mode();
    let wildcard_scope = opts.wildcard_scope();
    let include_terminal_items = simplifier.collect_steps();
    let include_unsupported_items = simplifier.collect_steps();
    let include_log_item = simplifier.collect_steps();

    execute_pow_isolation_with_default_kernels_for_var_with_state(
        simplifier,
        lhs,
        b,
        e,
        rhs,
        op,
        var,
        include_base_item,
        opts.budget.max_branches >= 2,
        opts.budget.can_branch(),
        solve_tactic_enabled,
        mode,
        wildcard_scope,
        include_shortcut_item,
        include_base_one_item,
        include_terminal_items,
        include_unsupported_items,
        include_log_item,
        steps,
        |simplifier| &simplifier.context,
        |simplifier| &mut simplifier.context,
        solver_render_expr,
        |simplifier, iso_lhs, iso_rhs, iso_op| {
            isolate(iso_lhs, iso_rhs, iso_op, var, simplifier, opts, ctx)
        },
        |item| medium_step(item.description().to_string(), item.equation),
        |simplifier, expr| simplifier.simplify(expr).0,
        |_simplifier| crate::domain::clear_blocked_hints(),
        |simplifier, expr| {
            simplifier
                .simplify_with_options(expr, tactic_opts.clone())
                .0
        },
        |simplifier| simplifier.collect_steps(),
        |simplifier, tactic_base, tactic_rhs| {
            crate::solver::classify_log_solve(
                &simplifier.context,
                tactic_base,
                tactic_rhs,
                &opts,
                &ctx.domain_env,
            )
        },
        |simplifier, shortcut_rhs, shortcut_op| {
            isolate(e, shortcut_rhs, shortcut_op, var, simplifier, opts, ctx)
        },
        |item| medium_step(item.description().to_string(), item.equation),
        |item| medium_step(item.description().to_string(), item.equation),
        |local_var, message| CasError::IsolationError(local_var.to_string(), message.to_string()),
        |item| medium_step(item.description().to_string(), item.equation),
        |item| medium_step(item.description().to_string(), item.equation),
        |core_ctx, assumption| {
            let event = crate::solver::assumption_event_from_log_assumption_targets(
                core_ctx, assumption, b, rhs,
            );
            ctx.note_assumption(event);
        },
        |simplifier, equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
            .ok()
            .map(|(solutions, _)| solutions)
        },
        |core_ctx, hint| {
            let blocked_hint =
                crate::solver::domain_blocked_hint_from_log_blocked_hint(core_ctx, hint);
            crate::domain::register_blocked_hint(blocked_hint);
        },
        |message| CasError::UnsupportedInRealDomain(message.to_string()),
        |simplifier, equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
        },
        |item| medium_step(item.description().to_string(), item.equation),
    )
}
