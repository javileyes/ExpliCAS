//! Action dispatch: `Engine.eval()` and per-action handler methods.
//!
//! This is the main entry point for evaluating requests. It handles
//! session storage, reference resolution, and dispatches to the
//! appropriate action handler (Simplify, Expand, Solve, Equiv, Limit).

use super::*;

type PreparedEvalDispatch =
    cas_session_core::eval::PreparedEvalDispatch<crate::diagnostics::Diagnostics, crate::Step>;

type StatelessEvalSession = cas_session_core::eval::StatelessEvalSession<
    crate::options::EvalOptions,
    crate::domain::DomainMode,
    crate::diagnostics::RequiredItem,
    crate::step::Step,
    crate::diagnostics::Diagnostics,
>;

fn build_cache_hit_step(
    ctx: &cas_ast::Context,
    description: String,
    before: ExprId,
    after: ExprId,
) -> crate::Step {
    let mut step = crate::Step::new(
        &description,
        "Use cached result",
        before,
        after,
        Vec::new(),
        Some(ctx),
    );
    step.importance = crate::step::ImportanceLevel::Medium;
    step.category = crate::step::StepCategory::Substitute;
    step
}

fn resolve_prepare_config(req: &EvalRequest) -> cas_session_core::eval::ResolvePrepareConfig {
    cas_session_core::eval::ResolvePrepareConfig {
        parsed: req.parsed,
        raw_input: req.raw_input.clone(),
        auto_store: req.auto_store,
        equiv_other: match &req.action {
            EvalAction::Equiv { other } => Some(*other),
            _ => None,
        },
        cache_step_max_shown: 6,
    }
}

impl Engine {
    /// Stateless eval path for APIs that do not use session refs (`#N`) or env vars.
    ///
    /// This keeps `cas_engine` usable without `cas_session` in outer orchestration layers.
    pub fn eval_stateless(
        &mut self,
        options: crate::options::EvalOptions,
        mut req: EvalRequest,
    ) -> Result<EvalOutput, anyhow::Error> {
        req.auto_store = false;
        let mut session = StatelessEvalSession::new(options);
        self.eval(&mut session, req)
    }

    /// The main entry point for evaluating requests.
    /// Handles session storage, resolution, and action dispatch.
    pub fn eval<S>(
        &mut self,
        session: &mut S,
        req: EvalRequest,
    ) -> Result<EvalOutput, anyhow::Error>
    where
        S: cas_session_core::eval::TypedEvalSession<
            crate::domain::DomainMode,
            crate::diagnostics::RequiredItem,
            crate::step::Step,
            crate::diagnostics::Diagnostics,
            crate::options::EvalOptions,
        >,
        S::Store: cas_session_core::eval::TypedEvalStore<
            crate::domain::DomainMode,
            crate::diagnostics::RequiredItem,
            crate::step::Step,
            crate::diagnostics::Diagnostics,
        >,
    {
        let prepared = cas_session_core::eval::resolve_and_prepare_dispatch(
            session,
            &mut self.simplifier.context,
            resolve_prepare_config(&req),
            build_cache_hit_step,
        )?;

        let options = session.options().clone();
        self.eval_with_parts(session, &options, req, prepared)
    }

    fn eval_with_parts<S>(
        &mut self,
        session: &mut S,
        options: &crate::options::EvalOptions,
        req: EvalRequest,
        prepared: PreparedEvalDispatch,
    ) -> Result<EvalOutput, anyhow::Error>
    where
        S: cas_session_core::eval::TypedEvalSession<
            crate::domain::DomainMode,
            crate::diagnostics::RequiredItem,
            crate::step::Step,
            crate::diagnostics::Diagnostics,
            crate::options::EvalOptions,
        >,
        S::Store: cas_session_core::eval::TypedEvalStore<
            crate::domain::DomainMode,
            crate::diagnostics::RequiredItem,
            crate::step::Step,
            crate::diagnostics::Diagnostics,
        >,
    {
        let PreparedEvalDispatch {
            stored_id,
            resolved,
            inherited_diagnostics,
            resolved_equiv_other,
            cache_hit_step,
        } = prepared;

        // 3. Dispatch Action -> produce EvalResult
        let (
            result,
            domain_warnings,
            mut steps,
            solve_steps,
            solver_assumptions,
            output_scopes,
            solver_required,
        ) = self.dispatch_eval_action(options, req.action, resolved, resolved_equiv_other)?;

        // V2.15.36: Prepend synthetic cache hit step if any refs were resolved from cache
        if let Some(step) = cache_hit_step {
            steps.insert(0, step);
        }

        // 4. Build diagnostics and finalize output
        self.build_output(
            stored_id,
            req.parsed,
            resolved,
            result,
            domain_warnings,
            steps,
            solve_steps,
            solver_assumptions,
            output_scopes,
            solver_required,
            inherited_diagnostics,
            session.store_mut(),
            options,
        )
    }
}
