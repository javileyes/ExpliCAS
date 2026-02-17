//! Post-processing: diagnostics assembly, session caching, and output construction.
//!
//! After action dispatch produces a raw result, this module assembles the
//! unified `Diagnostics`, updates session cache, and constructs the final `EvalOutput`.

use super::*;

impl Engine {
    /// Build unified diagnostics, update session cache, and construct final `EvalOutput`.
    ///
    /// This is the post-processing stage after action dispatch has produced raw results.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_output(
        &mut self,
        stored_id: Option<u64>,
        parsed: ExprId,
        resolved: ExprId,
        result: EvalResult,
        domain_warnings: Vec<DomainWarning>,
        steps: Vec<crate::Step>,
        solve_steps: Vec<crate::solver::SolveStep>,
        solver_assumptions: Vec<crate::assumptions::AssumptionRecord>,
        output_scopes: Vec<cas_ast::display_transforms::ScopeTag>,
        solver_required: Vec<crate::implicit_domain::ImplicitCondition>,
        inherited_diagnostics: crate::diagnostics::Diagnostics,
        store: &mut impl EvalStore,
        options: &crate::options::EvalOptions,
    ) -> Result<EvalOutput, anyhow::Error> {
        // Collect blocked hints from simplifier
        let blocked_hints = self.simplifier.take_blocked_hints();

        // V2.2+: Build unified Diagnostics with origin tracking
        // Each source gets its appropriate origin:
        // - Steps (rewrite airbag) → RewriteAirbag
        // - Solver → EquationDerived
        // - Structural inference on input → InputImplicit (via OutputImplicit for now)
        let mut diagnostics = crate::diagnostics::Diagnostics::new();

        // 1. Add requires from simplification steps → RewriteAirbag
        //    These are conditions detected when a rewrite consumed the witness
        {
            use std::collections::HashSet;
            let mut seen: HashSet<String> = HashSet::new();
            for step in &steps {
                for cond in step.required_conditions() {
                    let display = cond.display(&self.simplifier.context);
                    if seen.insert(display) {
                        diagnostics.push_required(
                            cond.clone(),
                            crate::diagnostics::RequireOrigin::RewriteAirbag,
                        );
                    }
                }
            }
        }

        // 2. Add requires from solver → EquationDerived
        //    These are conditions derived from equation structure
        for cond in &solver_required {
            diagnostics.push_required(
                cond.clone(),
                crate::diagnostics::RequireOrigin::EquationDerived,
            );
        }

        // 3. Add requires from structural inference
        //    InputImplicit: conditions visible in input (resolved) before simplification
        //    OutputImplicit: conditions visible in output (result) after simplification
        {
            use crate::implicit_domain::infer_implicit_domain;

            // InputImplicit: infer from resolved (input after ref resolution)
            let input_domain = infer_implicit_domain(
                &self.simplifier.context,
                resolved,
                crate::semantics::ValueDomain::RealOnly,
            );

            for cond in input_domain.conditions() {
                diagnostics.push_required(
                    cond.clone(),
                    crate::diagnostics::RequireOrigin::InputImplicit,
                );
            }

            // OutputImplicit: infer from result (after simplification/solving)
            // Extract ExprId from EvalResult if available
            let result_expr_id = match &result {
                EvalResult::Expr(e) => Some(*e),
                EvalResult::Set(exprs) => {
                    // For solve results (legacy), infer from first solution
                    exprs.first().copied()
                }
                EvalResult::SolutionSet(solution_set) => {
                    // For V2.0 solutions, extract first concrete value if any
                    use cas_ast::SolutionSet;
                    match solution_set {
                        SolutionSet::Discrete(vec) => vec.first().copied(),
                        _ => None,
                    }
                }
                _ => None,
            };

            if let Some(result_id) = result_expr_id {
                let output_domain = infer_implicit_domain(
                    &self.simplifier.context,
                    result_id,
                    crate::semantics::ValueDomain::RealOnly,
                );

                for cond in output_domain.conditions() {
                    diagnostics.push_required(
                        cond.clone(),
                        crate::diagnostics::RequireOrigin::OutputImplicit,
                    );
                }
            }
        }

        // Add blocked hints
        for hint in &blocked_hints {
            diagnostics.push_blocked(hint.clone());
        }

        // Add assumed events from solve steps (if any)
        for step in &steps {
            for event in step.assumption_events() {
                diagnostics.push_assumed(event.clone());
            }
        }

        // SessionPropagated: inherit requires from any referenced session entries
        // This tracks provenance when reusing #id
        diagnostics.inherit_requires_from(&inherited_diagnostics);

        // Dedup and sort for stable output (also filters trivials)
        diagnostics.dedup_and_sort(&self.simplifier.context);

        // Update stored entry with final diagnostics (for SessionPropagated tracking)
        if let Some(id) = stored_id {
            store.update_diagnostics(id, diagnostics.clone());

            // V2.15.36: Populate simplified cache for session reference caching
            // This enables `#N` to use the cached simplified result instead of re-simplifying
            if let EvalResult::Expr(simplified_expr) = &result {
                // Skip caching poly_result(id) handles.
                // The thread-local PolyStore is cleared before each evaluation,
                // so poly_result handles become dangling references in later evals.
                // By NOT caching, `#N` resolution falls back to the raw parsed
                // expression (e.g. `expand(...)`) which re-enters the full eval
                // pipeline and is correctly handled by the orchestrator's
                // eager-eval + poly_lower pre-passes.
                if !crate::poly_result::is_poly_result(&self.simplifier.context, *simplified_expr) {
                    use crate::session::{SimplifiedCache, SimplifyCacheKey};

                    let cache_key =
                        SimplifyCacheKey::from_context(options.shared.semantics.domain_mode);
                    let cache = SimplifiedCache {
                        key: cache_key,
                        expr: *simplified_expr,
                        requires: diagnostics.requires.clone(),
                        steps: Some(std::sync::Arc::new(steps.clone())),
                    };
                    store.update_simplified(id, cache);
                }
            }
        }

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
