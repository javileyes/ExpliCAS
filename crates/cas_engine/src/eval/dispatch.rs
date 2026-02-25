//! Action dispatch: `Engine.eval()` and per-action handler methods.
//!
//! This is the main entry point for evaluating requests. It handles
//! session storage, reference resolution, and dispatches to the
//! appropriate action handler (Simplify, Expand, Solve, Equiv, Limit).

use super::*;

struct ResolvedEvalInput {
    resolved: ExprId,
    inherited_diagnostics: crate::diagnostics::Diagnostics,
    cache_hits: Vec<u64>,
    resolved_equiv_other: Option<ExprId>,
}

#[derive(Default)]
struct NoopEvalStore;

impl EvalStore for NoopEvalStore {
    fn push_raw_expr(&mut self, _expr: ExprId, _raw_input: String) -> u64 {
        0
    }

    fn push_raw_equation(&mut self, _lhs: ExprId, _rhs: ExprId, _raw_input: String) -> u64 {
        0
    }

    fn touch_cached(&mut self, _entry_id: u64) {}

    fn update_diagnostics(&mut self, _id: u64, _diagnostics: crate::diagnostics::Diagnostics) {}

    fn update_simplified(
        &mut self,
        _id: u64,
        _domain: crate::domain::DomainMode,
        _expr: ExprId,
        _requires: Vec<crate::diagnostics::RequiredItem>,
        _steps: Option<std::sync::Arc<Vec<crate::step::Step>>>,
    ) {
    }
}

struct StatelessEvalSession {
    store: NoopEvalStore,
    options: crate::options::EvalOptions,
    profile_cache: crate::profile_cache::ProfileCache,
}

impl StatelessEvalSession {
    fn new(options: crate::options::EvalOptions) -> Self {
        Self {
            store: NoopEvalStore,
            options,
            profile_cache: crate::profile_cache::ProfileCache::new(),
        }
    }
}

impl EvalSession for StatelessEvalSession {
    type Store = NoopEvalStore;

    fn store_mut(&mut self) -> &mut Self::Store {
        &mut self.store
    }

    fn options(&self) -> &crate::options::EvalOptions {
        &self.options
    }

    fn profile_cache_mut(&mut self) -> &mut crate::profile_cache::ProfileCache {
        &mut self.profile_cache
    }

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> anyhow::Result<(ExprId, crate::diagnostics::Diagnostics, Vec<u64>)> {
        if let Some(ref_id) = first_session_ref(ctx, expr) {
            return Err(anyhow::anyhow!(
                "Session reference #{} requires stateful eval (Engine::eval with EvalSession)",
                ref_id
            ));
        }
        Ok((expr, crate::diagnostics::Diagnostics::new(), Vec::new()))
    }
}

fn first_session_ref(ctx: &cas_ast::Context, root: ExprId) -> Option<u64> {
    use cas_ast::Expr;

    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::SessionRef(ref_id) => return Some(*ref_id),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => {}
        }
    }

    None
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
    pub fn eval(
        &mut self,
        session: &mut impl EvalSession,
        req: EvalRequest,
    ) -> Result<EvalOutput, anyhow::Error> {
        // 1) Resolve parsed input (session refs + env vars) before mutably
        // borrowing session internals for store/cache updates.
        let (resolved, inherited_diagnostics, cache_hits) = session
            .resolve_all_with_diagnostics(&mut self.simplifier.context, req.parsed)
            .map_err(|e| anyhow::anyhow!("Resolution error: {}", e))?;

        // Equivalence checks need to resolve the "other" expression too.
        let resolved_equiv_other = match &req.action {
            EvalAction::Equiv { other } => Some(
                session
                    .resolve_all_with_diagnostics(&mut self.simplifier.context, *other)
                    .map_err(|e| anyhow::anyhow!("Resolution error in other: {}", e))?
                    .0,
            ),
            _ => None,
        };

        let resolved_input = ResolvedEvalInput {
            resolved,
            inherited_diagnostics,
            cache_hits,
            resolved_equiv_other,
        };

        let options = session.options().clone();
        self.eval_with_parts(session, &options, req, resolved_input)
    }

    fn eval_with_parts(
        &mut self,
        session: &mut impl EvalSession,
        options: &crate::options::EvalOptions,
        req: EvalRequest,
        resolved_input: ResolvedEvalInput,
    ) -> Result<EvalOutput, anyhow::Error> {
        let ResolvedEvalInput {
            resolved,
            inherited_diagnostics,
            cache_hits,
            resolved_equiv_other,
        } = resolved_input;

        // 2. Auto-store raw + parsed (unresolved)
        let stored_id = if req.auto_store {
            let id = if let Some((lhs, rhs)) =
                cas_ast::eq::unwrap_eq(&self.simplifier.context, req.parsed)
            {
                session
                    .store_mut()
                    .push_raw_equation(lhs, rhs, req.raw_input.clone())
            } else {
                session
                    .store_mut()
                    .push_raw_expr(req.parsed, req.raw_input.clone())
            };
            Some(id)
        } else {
            None
        };

        // V2.15.36: Build synthetic cache hit step (if any)
        let cache_hit_step =
            build_cache_hit_step(&self.simplifier.context, req.parsed, resolved, &cache_hits);

        // V2.15.36: Touch cached entries to maintain true LRU ordering
        for hit in &cache_hits {
            session.store_mut().touch_cached(*hit);
        }

        // 3. Dispatch Action -> produce EvalResult
        let (
            result,
            domain_warnings,
            mut steps,
            solve_steps,
            solver_assumptions,
            output_scopes,
            solver_required,
        ) = match req.action {
            EvalAction::Simplify => {
                let profile_cache = session.profile_cache_mut();
                self.eval_simplify(options, profile_cache, resolved)?
            }
            EvalAction::Expand => {
                // Treating Expand as Simplify for now, as Simplifier has no explicit expand mode yet exposed cleanly
                let (res, steps) = self.simplifier.simplify(resolved);
                let warnings = collect_domain_warnings(&steps);
                (
                    EvalResult::Expr(res),
                    warnings,
                    steps,
                    vec![],
                    vec![],
                    vec![],
                    vec![], // solver_required: empty for expand
                )
            }
            EvalAction::Solve { var } => self.eval_solve(options, resolved, &var)?,
            EvalAction::Equiv { .. } => {
                let resolved_other = resolved_equiv_other
                    .ok_or_else(|| anyhow::anyhow!("Missing resolved equivalence operand"))?;
                self.eval_equiv(resolved, resolved_other)?
            }
            EvalAction::Limit { var, approach } => self.eval_limit(resolved, &var, approach)?,
        };

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

    /// Handle `EvalAction::Simplify`: tool dispatch, simplification, const fold, domain classification.
    fn eval_simplify(
        &mut self,
        options: &crate::options::EvalOptions,
        profile_cache: &mut crate::profile_cache::ProfileCache,
        resolved: ExprId,
    ) -> Result<ActionResult, anyhow::Error> {
        // Determine effective context mode for this request
        let effective_opts = self.effective_options(options, resolved);

        // Get cached profile (or build once and cache)
        let profile = profile_cache.get_or_build(&effective_opts);

        // Create simplifier from cached profile
        let mut ctx_simplifier = Simplifier::from_profile(profile);
        // Transfer the context (expressions)
        ctx_simplifier.context = std::mem::take(&mut self.simplifier.context);

        // Use simplify_with_stats to respect expand_policy and other options
        let mut simplify_opts = effective_opts.to_simplify_options();

        // TOOL DISPATCHER: Detect tool functions and set appropriate goal
        // This prevents inverse rules from undoing the effect of collect/expand_log
        let (expr_to_simplify, expand_log_events) =
            if let Expr::Function(fn_id, args) = ctx_simplifier.context.get(resolved).clone() {
                match ctx_simplifier.context.sym_name(fn_id) {
                    "collect" => {
                        simplify_opts.goal = crate::semantics::NormalFormGoal::Collected;
                        (resolved, Vec::new())
                    }
                    "expand_log" if args.len() == 1 => {
                        // V2.12.14: Apply log expansion BEFORE simplification
                        // This ensures the result is simplified with goal=ExpandedLog
                        // which blocks LogContractionRule from undoing the expansion
                        simplify_opts.goal = crate::semantics::NormalFormGoal::ExpandedLog;
                        crate::rules::logarithms::expand_logs_with_assumptions(
                            &mut ctx_simplifier.context,
                            args[0],
                        )
                    }
                    _ => (resolved, Vec::new()),
                }
            } else {
                (resolved, Vec::new())
            };

        let (mut res, mut steps, _stats) =
            ctx_simplifier.simplify_with_stats(expr_to_simplify, simplify_opts);

        // Inject expand_log assumption events into the first step (or create one if needed)
        if !expand_log_events.is_empty() {
            if steps.is_empty() {
                // Create a placeholder step to hold the assumptions
                let mut step = crate::Step::new(
                    "Log expansion",
                    "expand_log",
                    resolved, // Before: the original expand_log(...) expression
                    res,      // After: the expanded result
                    Vec::new(),
                    Some(&ctx_simplifier.context),
                );
                step.meta_mut().assumption_events.extend(expand_log_events);
                steps.push(step);
            } else {
                // Add events to first step
                steps[0]
                    .meta_mut()
                    .assumption_events
                    .extend(expand_log_events);
            }
        }

        if effective_opts.const_fold == crate::const_fold::ConstFoldMode::Safe {
            let mut budget = crate::budget::Budget::preset_cli();
            // Use effective_opts to propagate value_domain to const_fold
            let cfg = crate::semantics::EvalConfig {
                value_domain: effective_opts.shared.semantics.value_domain,
                branch: effective_opts.shared.semantics.branch,
                ..Default::default()
            };
            if let Ok(fold_result) = crate::const_fold::fold_constants(
                &mut ctx_simplifier.context,
                res,
                &cfg,
                effective_opts.const_fold,
                &mut budget,
            ) {
                res = fold_result.expr;
            }
        }

        // Transfer context and blocked hints back to main simplifier
        // Hints must be transferred BEFORE context to preserve pedagogical feedback
        self.simplifier
            .extend_blocked_hints(ctx_simplifier.take_blocked_hints());
        self.simplifier.context = ctx_simplifier.context;

        // V2.14.15: Classify assumptions against global requires before collecting warnings
        // This suppresses warnings (⚠) for conditions already in Requires (ℹ)
        {
            use crate::implicit_domain::{
                classify_assumptions_in_place, infer_implicit_domain, DomainContext,
            };

            // Infer global requires from original input
            let input_domain = infer_implicit_domain(
                &self.simplifier.context,
                resolved,
                effective_opts.shared.semantics.value_domain,
            );
            let global_requires: Vec<_> = input_domain.conditions().iter().cloned().collect();
            let mut dc = DomainContext::new(global_requires);

            // Classify each step's assumption events
            for step in steps.iter_mut() {
                classify_assumptions_in_place(
                    &self.simplifier.context,
                    &mut dc,
                    &mut step.meta_mut().assumption_events,
                );
            }
        }

        // Collect domain warnings (now filtered by classification)
        let mut warnings = collect_domain_warnings(&steps);

        // Add warning if i is used in RealOnly mode
        if effective_opts.shared.semantics.value_domain == crate::semantics::ValueDomain::RealOnly
            && cas_math::numeric_eval::contains_i(&self.simplifier.context, resolved)
        {
            let i_warning = DomainWarning {
                message: "To use complex arithmetic (i² = -1), run: semantics set value complex"
                    .to_string(),
                rule_name: "Imaginary Usage Warning".to_string(),
            };
            // Only add if not already present
            if !warnings.iter().any(|w| w.message == i_warning.message) {
                warnings.push(i_warning);
            }
        }

        Ok((
            EvalResult::Expr(res),
            warnings,
            steps,
            vec![],
            vec![],
            vec![],
            vec![], // solver_required: empty for simplify
        ))
    }

    /// Handle `EvalAction::Solve`: equation construction, solver invocation.
    fn eval_solve(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        var: &str,
    ) -> Result<ActionResult, anyhow::Error> {
        // Construct proper Equation for solver
        // If resolved is "Equal(lhs, rhs)", use that.
        // Otherwise assume "resolved == 0".

        // We must peek at the resolved expression structure
        let eq_to_solve = match self.simplifier.context.get(resolved) {
            Expr::Function(fn_id, args)
                if self.simplifier.context.is_builtin(*fn_id, BuiltinFn::Equal)
                    && args.len() == 2 =>
            {
                Equation {
                    lhs: args[0],
                    rhs: args[1],
                    op: RelOp::Eq, // Assuming strict equality for Solve for now
                }
            }
            Expr::Function(fn_id, args)
                if self.simplifier.context.is_builtin(*fn_id, BuiltinFn::Less)
                    && args.len() == 2 =>
            {
                Equation {
                    lhs: args[0],
                    rhs: args[1],
                    op: RelOp::Lt,
                }
            }
            Expr::Function(fn_id, args)
                if self
                    .simplifier
                    .context
                    .is_builtin(*fn_id, BuiltinFn::Greater)
                    && args.len() == 2 =>
            {
                Equation {
                    lhs: args[0],
                    rhs: args[1],
                    op: RelOp::Gt,
                }
            }
            // Handle other ops if needed, or default to Expr = 0
            _ => {
                use num_traits::Zero;
                let zero = self
                    .simplifier
                    .context
                    .add(Expr::Number(num_rational::BigRational::zero()));
                Equation {
                    lhs: resolved,
                    rhs: zero,
                    op: RelOp::Eq,
                }
            }
        };

        // Call solver with semantic options and assumption collection
        let solver_opts = crate::solver::SolverOptions {
            value_domain: options.shared.semantics.value_domain,
            domain_mode: options.shared.semantics.domain_mode,
            assume_scope: options.shared.semantics.assume_scope,
            budget: options.budget,
            ..Default::default()
        };

        // V2.9.8: Use type-safe API that guarantees cleanup is applied
        // detailed_steps=true for Normal+ verbosity (caller's responsibility to set)
        let sol_result = crate::solver::solve_with_display_steps(
            &eq_to_solve,
            var,
            &mut self.simplifier,
            solver_opts,
        );

        match sol_result {
            Ok((solution_set, display_steps, diagnostics)) => {
                // V2.9.8: Extract Vec<SolveStep> from DisplaySolveSteps wrapper
                // Steps are guaranteed to be post-cleanup
                let solve_steps = display_steps.0;

                let solver_assumptions = if options.shared.assumption_reporting
                    == crate::assumptions::AssumptionReporting::Off
                {
                    vec![]
                } else {
                    diagnostics.assumed_records.clone()
                };

                // V2.0: Return the full SolutionSet preserving all variants
                // including Conditional for proper REPL display
                let warnings: Vec<DomainWarning> = vec![];
                let eval_res = EvalResult::SolutionSet(solution_set);
                // Output scopes are now solve-context data (no TLS collector).
                let output_scopes = diagnostics.output_scopes;

                // V2.16+: Required conditions now come in-band via SolveDiagnostics
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
    fn eval_equiv(
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
            vec![], // solver_required: empty for equiv
        ))
    }

    /// Handle `EvalAction::Limit`: compute limit using the limits engine.
    fn eval_limit(
        &mut self,
        resolved: ExprId,
        var: &str,
        approach: crate::limits::Approach,
    ) -> Result<ActionResult, anyhow::Error> {
        // Compute limit using the limits engine
        use crate::limits::{limit, LimitOptions};

        // Create variable ExprId from name
        let var_id = self.simplifier.context.var(var);

        // Default limit options
        let opts = LimitOptions::default();

        // Budget for limit computation
        let mut budget = crate::budget::Budget::preset_cli();

        match limit(
            &mut self.simplifier.context,
            resolved,
            var_id,
            approach,
            &opts,
            &mut budget,
        ) {
            Ok(result) => {
                let res_expr = result.expr;
                Ok((
                    EvalResult::Expr(res_expr),
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![], // solver_required: empty for limit
                ))
            }
            Err(e) => Err(anyhow::anyhow!("Limit error: {}", e)),
        }
    }
}
