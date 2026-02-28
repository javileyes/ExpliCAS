//! Post-processing: diagnostics assembly, session caching, and output construction.
//!
//! After action dispatch produces a raw result, this module assembles the
//! unified `Diagnostics`, updates session cache, and constructs the final `EvalOutput`.

use super::*;

type EngineSimplifiedUpdate = cas_session_core::eval::SimplifiedUpdate<
    crate::domain::DomainMode,
    crate::diagnostics::RequiredItem,
    crate::step::Step,
>;

fn push_step_requires_with_display_dedup(
    ctx: &cas_ast::Context,
    steps: &[crate::Step],
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    use std::collections::HashSet;

    let mut seen: HashSet<String> = HashSet::new();
    for step in steps {
        for cond in step.required_conditions() {
            let display = cond.display(ctx);
            if seen.insert(display) {
                diagnostics.push_required(
                    cond.clone(),
                    crate::diagnostics::RequireOrigin::RewriteAirbag,
                );
            }
        }
    }
}

fn push_solver_requires(
    solver_required: &[crate::implicit_domain::ImplicitCondition],
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    for cond in solver_required {
        diagnostics.push_required(
            cond.clone(),
            crate::diagnostics::RequireOrigin::EquationDerived,
        );
    }
}

fn eval_result_first_expr(result: &EvalResult) -> Option<ExprId> {
    match result {
        EvalResult::Expr(e) => Some(*e),
        EvalResult::Set(exprs) => exprs.first().copied(),
        EvalResult::SolutionSet(solution_set) => {
            use cas_ast::SolutionSet;
            match solution_set {
                SolutionSet::Discrete(vec) => vec.first().copied(),
                _ => None,
            }
        }
        _ => None,
    }
}

fn push_structural_requires(
    ctx: &cas_ast::Context,
    resolved: ExprId,
    result: &EvalResult,
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    use crate::implicit_domain::infer_implicit_domain;

    let input_domain =
        infer_implicit_domain(ctx, resolved, crate::semantics::ValueDomain::RealOnly);
    for cond in input_domain.conditions() {
        diagnostics.push_required(
            cond.clone(),
            crate::diagnostics::RequireOrigin::InputImplicit,
        );
    }

    if let Some(result_id) = eval_result_first_expr(result) {
        let output_domain =
            infer_implicit_domain(ctx, result_id, crate::semantics::ValueDomain::RealOnly);
        for cond in output_domain.conditions() {
            diagnostics.push_required(
                cond.clone(),
                crate::diagnostics::RequireOrigin::OutputImplicit,
            );
        }
    }
}

fn push_blocked_hints(
    blocked_hints: &[crate::domain::BlockedHint],
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    for hint in blocked_hints {
        diagnostics.push_blocked(hint.clone());
    }
}

fn push_assumed_from_steps(
    steps: &[crate::Step],
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    for step in steps {
        for event in step.assumption_events() {
            diagnostics.push_assumed(event.clone());
        }
    }
}

fn build_simplified_cache_update(
    ctx: &cas_ast::Context,
    result: &EvalResult,
    options: &crate::options::EvalOptions,
    diagnostics: &crate::diagnostics::Diagnostics,
    steps: &[crate::Step],
) -> Option<EngineSimplifiedUpdate> {
    match result {
        EvalResult::Expr(simplified_expr)
            if !cas_math::poly_result::is_poly_result(ctx, *simplified_expr) =>
        {
            Some(EngineSimplifiedUpdate {
                domain: options.shared.semantics.domain_mode,
                expr: *simplified_expr,
                requires: diagnostics.requires.clone(),
                steps: Some(std::sync::Arc::new(steps.to_vec())),
            })
        }
        _ => None,
    }
}

fn build_eval_diagnostics(
    ctx: &cas_ast::Context,
    resolved: ExprId,
    result: &EvalResult,
    steps: &[crate::Step],
    solver_required: &[crate::implicit_domain::ImplicitCondition],
    blocked_hints: &[crate::domain::BlockedHint],
    inherited_diagnostics: &crate::diagnostics::Diagnostics,
) -> crate::diagnostics::Diagnostics {
    let mut diagnostics = crate::diagnostics::Diagnostics::new();

    // Each source gets its proper provenance/origin classification.
    push_step_requires_with_display_dedup(ctx, steps, &mut diagnostics);
    push_solver_requires(solver_required, &mut diagnostics);
    push_structural_requires(ctx, resolved, result, &mut diagnostics);
    push_blocked_hints(blocked_hints, &mut diagnostics);
    push_assumed_from_steps(steps, &mut diagnostics);

    // Track provenance when reusing cached/session entries.
    diagnostics.inherit_requires_from(inherited_diagnostics);

    // Stable output ordering and trivial-condition filtering.
    diagnostics.dedup_and_sort(ctx);
    diagnostics
}

impl Engine {
    /// Build unified diagnostics, update session cache, and construct final `EvalOutput`.
    ///
    /// This is the post-processing stage after action dispatch has produced raw results.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_output<StoreT>(
        &mut self,
        stored_id: Option<u64>,
        parsed: ExprId,
        resolved: ExprId,
        result: EvalResult,
        domain_warnings: Vec<DomainWarning>,
        steps: Vec<crate::Step>,
        solve_steps: Vec<crate::solver::SolveStep>,
        solver_assumptions: Vec<crate::assumptions::AssumptionRecord>,
        output_scopes: Vec<cas_formatter::display_transforms::ScopeTag>,
        solver_required: Vec<crate::implicit_domain::ImplicitCondition>,
        inherited_diagnostics: crate::diagnostics::Diagnostics,
        store: &mut StoreT,
        options: &crate::options::EvalOptions,
    ) -> Result<EvalOutput, anyhow::Error>
    where
        StoreT: cas_session_core::eval::TypedEvalStore<
            crate::domain::DomainMode,
            crate::diagnostics::RequiredItem,
            crate::step::Step,
            crate::diagnostics::Diagnostics,
        >,
    {
        // Collect blocked hints from simplifier
        let blocked_hints = self.simplifier.take_blocked_hints();

        let diagnostics = build_eval_diagnostics(
            &self.simplifier.context,
            resolved,
            &result,
            &steps,
            &solver_required,
            &blocked_hints,
            &inherited_diagnostics,
        );

        // Update stored entry with final diagnostics and optional simplified cache.
        // This keeps cache write policy centralized in session-core helpers.
        let simplified_update = build_simplified_cache_update(
            &self.simplifier.context,
            &result,
            options,
            &diagnostics,
            &steps,
        );
        cas_session_core::eval::apply_post_dispatch_store_updates(
            store,
            stored_id,
            diagnostics.clone(),
            simplified_update,
        );

        // Legacy field: extract conditions from diagnostics for backward compatibility
        // Tests and some code paths still use output.required_conditions
        let required_conditions = diagnostics.required_conditions();

        // V2.9.9: Convert raw steps to display-ready steps via unified pipeline.
        // This is the ONLY place DisplayEvalSteps is constructed from raw steps.
        // The pipeline removes no-ops and prepares steps for all renderers.
        let display_steps = crate::eval_step_pipeline::to_display_steps(steps);

        Ok(EvalOutput {
            stored_id,
            parsed,
            resolved,
            result,
            domain_warnings,
            steps: display_steps,
            solve_steps,
            solver_assumptions,
            output_scopes,
            required_conditions,
            blocked_hints,
            diagnostics,
        })
    }
}
