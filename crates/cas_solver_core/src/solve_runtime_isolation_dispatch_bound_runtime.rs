//! Shared isolation-dispatch wrapper bound to [`RuntimeSolveAdapterState`].

use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

/// Dispatch isolation using the default runtime-state helpers while only
/// requiring recursive solve/isolation entrypoints plus the runtime-specific
/// positive prover and blocked-hint sink.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_isolation_with_runtime_state_and_reentrant_entrypoints_and_state<
    T,
    FSolveReentrant,
    FIsolateReentrant,
    FRegisterBlockedHint,
>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    solve_reentrant: FSolveReentrant,
    mut isolate_reentrant: FIsolateReentrant,
    register_blocked_hint: FRegisterBlockedHint,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    T: crate::proof_runtime_bound_runtime::RuntimeProofSimplifierFactory
        + crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
    FSolveReentrant: FnMut(
        &Equation,
        &str,
        &mut T,
        crate::solver_options::SolverOptions,
        &crate::solve_runtime_types::RuntimeSolveCtx,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
    FIsolateReentrant: FnMut(
        ExprId,
        ExprId,
        RelOp,
        &str,
        &mut T,
        crate::solver_options::SolverOptions,
        &crate::solve_runtime_types::RuntimeSolveCtx,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
    FRegisterBlockedHint: FnMut(crate::blocked_hint::BlockedHint),
{
    // PARAMETRIC monotone-step guard for ORDER relations: dividing by (or scaling
    // through) a VAR-FREE NON-NUMERIC constant requires its sign. A PROVEN sign
    // transforms exactly — positive keeps the direction, negative flips — and
    // recurses on the peeled side (this also bypasses the equation-only
    // linear-collect kernel that dropped the operator: `(a²+1)·x > b` returned the
    // DISCRETE boundary `{b/(a²+1)}`). An UNPROVABLE sign (a bare parameter `a`)
    // declines with the same honest message as the quadratic strategy's guard,
    // instead of emitting an unconditional ray that is wrong for half the
    // parameter space (`a·x > b` → `(b/a, ∞)`, `√x < a` → `[0, a²)`). NUMERIC
    // coefficients keep their historical routes untouched (zero churn).
    if matches!(op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        let shape = classify_parametric_monotone_step(state.runtime_context(), lhs, rhs, var);
        match shape {
            Some(ParametricMonotoneStep::ScaleBy {
                kept,
                factor,
                rhs_multiplies,
            }) => {
                let negative =
                    crate::solve_runtime_adapter_state_runtime::simplifier_is_known_negative(
                        state, factor,
                    );
                let positive = !negative
                    && matches!(
                        crate::proof_runtime_bound_runtime::prove_positive_with_runtime_proof_simplifier::<T>(
                            state.runtime_context(),
                            factor,
                            opts.value_domain,
                        ),
                        crate::domain_proof::Proof::Proven
                    );
                if positive || negative {
                    let combined = if rhs_multiplies {
                        state
                            .runtime_context_mut()
                            .add(cas_ast::Expr::Mul(rhs, factor))
                    } else {
                        state
                            .runtime_context_mut()
                            .add(cas_ast::Expr::Div(rhs, factor))
                    };
                    let new_rhs =
                        crate::solve_runtime_adapter_state_runtime::simplifier_simplify_expr(
                            state, combined,
                        );
                    let new_op = if negative {
                        crate::isolation_utils::flip_inequality(op)
                    } else {
                        op
                    };
                    return isolate_reentrant(kept, new_rhs, new_op, var, state, opts, ctx);
                }
                return Err(crate::error_model::CasError::SolverError(
                    "Inequalities with symbolic coefficients not yet supported".to_string(),
                ));
            }
            Some(ParametricMonotoneStep::UnprovableOnly { probe }) => {
                let negative =
                    crate::solve_runtime_adapter_state_runtime::simplifier_is_known_negative(
                        state, probe,
                    );
                let positive = !negative
                    && matches!(
                        crate::proof_runtime_bound_runtime::prove_positive_with_runtime_proof_simplifier::<T>(
                            state.runtime_context(),
                            probe,
                            opts.value_domain,
                        ),
                        crate::domain_proof::Proof::Proven
                    );
                if !(positive || negative) {
                    return Err(crate::error_model::CasError::SolverError(
                        "Inequalities with symbolic coefficients not yet supported".to_string(),
                    ));
                }
                // A provable sign keeps the historical route (its owners handle it).
            }
            None => {}
        }
    }
    crate::solve_runtime_isolation_dispatch_reentrant_context_runtime::dispatch_isolation_with_runtime_ctx_and_reentrant_entrypoints_and_state(
        state,
        lhs,
        rhs,
        op,
        var,
        opts,
        ctx,
        crate::solve_runtime_adapter_state_runtime::simplifier_context,
        crate::solve_runtime_adapter_state_runtime::simplifier_context_mut,
        crate::solve_runtime_adapter_state_runtime::simplifier_simplify_expr,
        crate::solve_runtime_adapter_state_runtime::simplifier_collect_steps,
        crate::solve_runtime_adapter_state_runtime::simplifier_prove_nonzero_status,
        crate::solve_runtime_adapter_state_runtime::context_render_expr,
        solve_reentrant,
        crate::solve_runtime_adapter_state_runtime::simplifier_is_known_negative,
        isolate_reentrant,
        crate::solve_runtime_adapter_state_runtime::simplifier_clear_blocked_hints,
        crate::solve_runtime_adapter_state_runtime::simplifier_simplify_with_options_expr,
        crate::proof_runtime_bound_runtime::prove_positive_with_runtime_proof_simplifier::<T>,
        register_blocked_hint,
        crate::solve_runtime_adapter_state_runtime::simplify_rhs_with_step_pairs,
        crate::solve_runtime_adapter_state_runtime::sym_name_as_string,
    )
}

/// A monotone isolation step about to run against a VAR-FREE NON-NUMERIC constant.
enum ParametricMonotoneStep {
    /// `factor · kept {op} rhs` (or `kept / factor`): peel the factor into the RHS
    /// (`rhs_multiplies` = true for the division shape).
    ScaleBy {
        kept: ExprId,
        factor: ExprId,
        rhs_multiplies: bool,
    },
    /// Shapes where an unprovable sign must decline but a provable one keeps the
    /// existing route: the constant NUMERATOR of `c/f(x)` and the var-free
    /// threshold of an even-root comparison (`√x < a`).
    UnprovableOnly { probe: ExprId },
}

/// Classify `lhs {order-op} rhs` when the next isolation step would divide by or
/// scale through a var-free constant that is NOT a plain number. Returns `None`
/// for every other shape (numeric coefficients keep their historical routes).
fn classify_parametric_monotone_step(
    ctx: &cas_ast::Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<ParametricMonotoneStep> {
    use crate::isolation_utils::contains_var;
    use cas_ast::Expr;
    let var_free_non_numeric = |e: ExprId| -> bool {
        !contains_var(ctx, e, var) && cas_math::numeric_eval::as_rational_const(ctx, e).is_none()
    };
    match ctx.get(lhs).clone() {
        Expr::Mul(l, r) => {
            let (kept, factor) = if contains_var(ctx, l, var) && !contains_var(ctx, r, var) {
                (l, r)
            } else if contains_var(ctx, r, var) && !contains_var(ctx, l, var) {
                (r, l)
            } else {
                return None;
            };
            if var_free_non_numeric(factor) {
                return Some(ParametricMonotoneStep::ScaleBy {
                    kept,
                    factor,
                    rhs_multiplies: false,
                });
            }
            None
        }
        Expr::Div(num, den) => {
            if contains_var(ctx, num, var) && var_free_non_numeric(den) {
                return Some(ParametricMonotoneStep::ScaleBy {
                    kept: num,
                    factor: den,
                    rhs_multiplies: true,
                });
            }
            if contains_var(ctx, den, var) && var_free_non_numeric(num) {
                return Some(ParametricMonotoneStep::UnprovableOnly { probe: num });
            }
            None
        }
        Expr::Pow(base, exp) => {
            // Even-root LHS (`√f {op} a`): the threshold's sign decides the set shape.
            let is_sqrt = cas_math::numeric_eval::as_rational_const(ctx, exp)
                .map(|q| q == num_rational::BigRational::new(1.into(), 2.into()))
                .unwrap_or(false);
            if is_sqrt && contains_var(ctx, base, var) && var_free_non_numeric(rhs) {
                return Some(ParametricMonotoneStep::UnprovableOnly { probe: rhs });
            }
            None
        }
        Expr::Function(fn_id, args) => {
            if args.len() == 1
                && ctx.is_builtin(fn_id, cas_ast::BuiltinFn::Sqrt)
                && contains_var(ctx, args[0], var)
                && var_free_non_numeric(rhs)
            {
                return Some(ParametricMonotoneStep::UnprovableOnly { probe: rhs });
            }
            None
        }
        _ => None,
    }
}
