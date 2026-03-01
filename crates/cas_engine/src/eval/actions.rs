//! Action handlers used by `Engine::eval` dispatch.
//!
//! This keeps non-simplify actions (solve/equiv/limit) out of the main
//! dispatch orchestrator so the pipeline flow stays compact.

use super::*;

impl Engine {
    /// Route one already-resolved action to its concrete handler.
    pub(super) fn dispatch_eval_action(
        &mut self,
        options: &crate::options::EvalOptions,
        action: EvalAction,
        resolved: ExprId,
        resolved_equiv_other: Option<ExprId>,
    ) -> Result<ActionResult, anyhow::Error> {
        match action {
            EvalAction::Simplify => self.eval_simplify(options, resolved),
            EvalAction::Expand => self.eval_expand(resolved),
            EvalAction::Solve { var } => self.eval_solve(options, resolved, &var),
            EvalAction::Equiv { .. } => {
                let resolved_other = resolved_equiv_other
                    .ok_or_else(|| anyhow::anyhow!("Missing resolved equivalence operand"))?;
                self.eval_equiv(resolved, resolved_other)
            }
            EvalAction::Limit { var, approach } => self.eval_limit(resolved, &var, approach),
        }
    }

    /// Handle `EvalAction::Solve`: equation construction, solver invocation.
    pub(super) fn eval_solve(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        var: &str,
    ) -> Result<ActionResult, anyhow::Error> {
        let eq_to_solve = cas_solver_core::solve_entry::equation_from_expr_or_zero(
            &mut self.simplifier.context,
            resolved,
        );

        let solver_opts = crate::api::SolverOptions::from_eval_options(options);

        let sol_result = crate::api::solve_with_display_steps(
            &eq_to_solve,
            var,
            &mut self.simplifier,
            solver_opts,
        );

        match sol_result {
            Ok((solution_set, display_steps, diagnostics)) => {
                let solve_steps = display_steps.0;

                let solver_assumptions = if options.shared.assumption_reporting
                    == crate::assumptions::AssumptionReporting::Off
                {
                    vec![]
                } else {
                    diagnostics.assumed_records.clone()
                };

                let warnings: Vec<DomainWarning> = vec![];
                let eval_res = EvalResult::SolutionSet(solution_set);
                let output_scopes = diagnostics.output_scopes;
                let solver_required = diagnostics.required;

                Ok((
                    eval_res,
                    warnings,
                    vec![],
                    solve_steps,
                    solver_assumptions,
                    output_scopes,
                    solver_required,
                ))
            }
            Err(e) => Err(anyhow::anyhow!("Solver error: {}", e)),
        }
    }

    /// Handle `EvalAction::Equiv`: resolve other expression and check equivalence.
    pub(super) fn eval_equiv(
        &mut self,
        resolved: ExprId,
        resolved_other: ExprId,
    ) -> Result<ActionResult, anyhow::Error> {
        let are_eq = self.simplifier.are_equivalent(resolved, resolved_other);
        Ok((
            EvalResult::Bool(are_eq),
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ))
    }

    /// Handle `EvalAction::Limit`: compute limit using the limits engine.
    pub(super) fn eval_limit(
        &mut self,
        resolved: ExprId,
        var: &str,
        approach: crate::limits::Approach,
    ) -> Result<ActionResult, anyhow::Error> {
        use crate::limits::{limit, LimitOptions};

        let var_id = self.simplifier.context.var(var);
        let opts = LimitOptions::default();
        let mut budget = crate::budget::Budget::preset_cli();

        match limit(
            &mut self.simplifier.context,
            resolved,
            var_id,
            approach,
            &opts,
            &mut budget,
        ) {
            Ok(result) => Ok((
                EvalResult::Expr(result.expr),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            )),
            Err(e) => Err(anyhow::anyhow!("Limit error: {}", e)),
        }
    }
}
