use crate::session::{EntryKind, ResolveError};
use crate::session_state::SessionState;
use crate::Simplifier;
use cas_ast::{BuiltinFn, Equation, Expr, ExprId, RelOp};

/// The central Engine struct that wraps the core Simplifier and potentially other components.
///
/// # Example
///
/// ```
/// use cas_engine::Engine;
/// use cas_parser::parse;
///
/// let mut engine = Engine::new();
/// let expr = parse("x + x", &mut engine.simplifier.context).unwrap();
/// let (result, _steps) = engine.simplifier.simplify(expr);
///
/// // Result is simplified
/// use cas_ast::display::DisplayExpr;
/// let output = format!("{}", DisplayExpr { context: &engine.simplifier.context, id: result });
/// assert!(output.contains("x")); // Contains x
/// ```
pub struct Engine {
    pub simplifier: Simplifier,
}

impl Engine {
    /// Create a new Engine with default rules.
    ///
    /// # Example
    ///
    /// ```
    /// use cas_engine::Engine;
    ///
    /// let engine = Engine::new();
    /// // Engine is ready to simplify expressions
    /// ```
    pub fn new() -> Self {
        Self {
            simplifier: Simplifier::with_default_rules(),
        }
    }

    /// Create an Engine with a pre-populated Context (for session restoration).
    pub fn with_context(context: cas_ast::Context) -> Self {
        Self {
            simplifier: Simplifier::with_context(context),
        }
    }

    /// Determine effective options, resolving Auto modes based on expression content.
    /// - ContextMode::Auto → IntegratePrep if contains integrate(), else Standard
    /// - ComplexMode::Auto → On if contains i, else Off
    /// - expand_policy forced Off in Solve mode (anti-contamination)
    fn effective_options(
        &self,
        opts: &crate::options::EvalOptions,
        expr: ExprId,
    ) -> crate::options::EvalOptions {
        use crate::options::{ComplexMode, ContextMode};
        use crate::phase::ExpandPolicy;

        let mut effective = opts.clone();

        // Resolve ContextMode::Auto
        if opts.context_mode == ContextMode::Auto {
            if crate::helpers::contains_integral(&self.simplifier.context, expr) {
                effective.context_mode = ContextMode::IntegratePrep;
            } else {
                effective.context_mode = ContextMode::Standard;
            }
        }

        // Resolve ComplexMode::Auto
        if opts.complex_mode == ComplexMode::Auto {
            if crate::helpers::contains_i(&self.simplifier.context, expr) {
                effective.complex_mode = ComplexMode::On;
            } else {
                effective.complex_mode = ComplexMode::Off;
            }
        }

        // CRITICAL: Force expand_policy Off in Solve mode
        // Auto-expansion can interfere with equation solving strategies
        if effective.context_mode == ContextMode::Solve {
            effective.expand_policy = ExpandPolicy::Off;
        }

        effective
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub enum EvalAction {
    Simplify,
    Expand,
    // Solve for a variable.
    Solve { var: String },
    // Check equivalence between two expressions
    Equiv { other: ExprId },
}

#[derive(Clone, Debug)]
pub struct EvalRequest {
    pub raw_input: String,
    pub parsed: ExprId,
    pub kind: EntryKind,
    pub action: EvalAction,
    pub auto_store: bool,
}

#[derive(Clone, Debug)]
pub enum EvalResult {
    Expr(ExprId),
    Set(Vec<ExprId>),                  // For Solve multiple roots (legacy)
    SolutionSet(cas_ast::SolutionSet), // V2.0: Full solution set including Conditional
    Bool(bool),                        // For Equiv
    None,                              // For commands with no output
}

/// A domain assumption warning with its source rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DomainWarning {
    pub message: String,
    pub rule_name: String,
}

#[derive(Clone, Debug)]
pub struct EvalOutput {
    pub stored_id: Option<u64>,
    pub parsed: ExprId,
    pub resolved: ExprId,
    pub result: EvalResult,
    /// Domain warnings with deduplication and rule source.
    pub domain_warnings: Vec<DomainWarning>,
    pub steps: crate::step::DisplayEvalSteps,
    pub solve_steps: Vec<crate::solver::SolveStep>,
    /// Assumptions made during solver operations (for Assume mode).
    pub solver_assumptions: Vec<crate::assumptions::AssumptionRecord>,
    /// Scopes for context-aware display (e.g., QuadraticFormula -> sqrt display).
    pub output_scopes: Vec<cas_ast::display_transforms::ScopeTag>,
    /// Required conditions for validity - implicit domain constraints from input.
    /// NOT assumptions! These were already required by the input expression.
    /// Sorted and deduplicated for stable display.
    pub required_conditions: Vec<crate::implicit_domain::ImplicitCondition>,
    /// Blocked hints - operations that could not proceed in Strict/Generic mode.
    /// These suggest using Assume mode to enable certain transformations.
    pub blocked_hints: Vec<crate::domain::BlockedHint>,
    /// V2.2+: Unified diagnostics with origin tracking.
    /// Consolidates requires, assumed, and blocked in one container.
    pub diagnostics: crate::diagnostics::Diagnostics,
}

/// Collect domain warnings from steps with deduplication.
/// Collects structured assumption_events from each step.
/// Note: Only events that are NOT RequiresIntroduced become DomainWarnings (⚠️).
/// RequiresIntroduced events are displayed in steps with ℹ️ icon instead.
fn collect_domain_warnings(steps: &[crate::Step]) -> Vec<DomainWarning> {
    use std::collections::HashSet;

    let mut seen: HashSet<String> = HashSet::new();
    let mut warnings = Vec::new();

    for step in steps {
        // Collect structured assumption_events
        for event in &step.assumption_events {
            // Skip RequiresIntroduced - these show in steps with ℹ️ icon, not as ⚠️ warnings
            // Skip DerivedFromRequires - these are implied by existing requires, don't show
            if matches!(
                event.kind,
                crate::assumptions::AssumptionKind::RequiresIntroduced
                    | crate::assumptions::AssumptionKind::DerivedFromRequires
            ) {
                continue;
            }
            let msg_str = event.message.clone();
            if !seen.contains(&msg_str) {
                seen.insert(msg_str.clone());
                warnings.push(DomainWarning {
                    message: msg_str,
                    rule_name: step.rule_name.clone(),
                });
            }
        }
    }

    warnings
}

/// V2.15.36: Build a synthetic timeline step showing cache hits.
///
/// Creates a single aggregated step when `#N` references were resolved
/// from cache. This provides traceability ("Used cached result from #1, #3")
/// without repeating the full derivation steps.
fn build_cache_hit_step(
    ctx: &cas_ast::Context,
    original_expr: cas_ast::ExprId,
    resolved_expr: cas_ast::ExprId,
    cache_hits: &[crate::session::CacheHitTrace],
) -> Option<crate::Step> {
    if cache_hits.is_empty() {
        return None;
    }

    // Collect and sort entry IDs for deterministic output
    let mut ids: Vec<u64> = cache_hits.iter().map(|h| h.entry_id).collect();
    ids.sort();

    // Format the description with truncation for readability
    let shown: Vec<String> = ids.iter().take(6).map(|id| format!("#{}", id)).collect();
    let suffix = if ids.len() > 6 {
        format!(" (+{})", ids.len() - 6)
    } else {
        String::new()
    };

    let description = format!(
        "Used cached simplified result from {}{}",
        shown.join(", "),
        suffix
    );

    let mut step = crate::Step::new(
        &description,        // label
        "Use cached result", // rule_name
        original_expr,       // before: the original parsed expression with #N
        resolved_expr,       // after: with #N replaced by cached simplified result
        Vec::new(),          // child_steps
        Some(ctx),           // context for display
    );
    // V2.15.36: Set to Medium so it appears in the timeline
    step.importance = crate::step::ImportanceLevel::Medium;
    step.category = crate::step::StepCategory::Substitute;
    Some(step)
}

impl Engine {
    /// The main entry point for evaluating requests.
    /// Handles session storage, resolution, and action dispatch.
    pub fn eval(
        &mut self,
        state: &mut SessionState,
        req: EvalRequest,
    ) -> Result<EvalOutput, anyhow::Error> {
        // 1. Auto-store raw + parsed (unresolved)
        let stored_id = if req.auto_store {
            Some(state.store.push(req.kind, req.raw_input.clone()))
        } else {
            None
        };

        // 2. Resolve (state.resolve_all_with_diagnostics)
        // We resolve the parsed expression against the session state (Session refs #id and Environment vars)
        // Also captures inherited diagnostics from any referenced entries for SessionPropagated tracking
        // V2.15.36: Also returns cache hit traces for synthetic timeline step generation
        let (resolved, inherited_diagnostics, cache_hits) =
            match state.resolve_all_with_diagnostics(&mut self.simplifier.context, req.parsed) {
                Ok(r) => r,
                Err(ResolveError::CircularReference(msg)) => {
                    return Err(anyhow::anyhow!("Circular reference detected: {}", msg))
                }
                Err(e) => return Err(anyhow::anyhow!("Resolution error: {}", e)),
            };

        // V2.15.36: Build synthetic cache hit step (if any)
        let cache_hit_step =
            build_cache_hit_step(&self.simplifier.context, req.parsed, resolved, &cache_hits);

        // V2.15.36: Touch cached entries to maintain true LRU ordering
        for hit in &cache_hits {
            state.store.touch_cached(hit.entry_id);
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
                // Determine effective context mode for this request
                let effective_opts = self.effective_options(&state.options, resolved);

                // Get cached profile (or build once and cache)
                let profile = state.profile_cache.get_or_build(&effective_opts);

                // Create simplifier from cached profile
                let mut ctx_simplifier = Simplifier::from_profile(profile);
                // Transfer the context (expressions)
                ctx_simplifier.context = std::mem::take(&mut self.simplifier.context);

                // Use simplify_with_stats to respect expand_policy and other options
                let mut simplify_opts = effective_opts.to_simplify_options();

                // TOOL DISPATCHER: Detect tool functions and set appropriate goal
                // This prevents inverse rules from undoing the effect of collect/expand_log
                let (expr_to_simplify, expand_log_events) = if let Expr::Function(fn_id, args) =
                    ctx_simplifier.context.get(resolved).clone()
                {
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
                        step.assumption_events.extend(expand_log_events);
                        steps.push(step);
                    } else {
                        // Add events to first step
                        steps[0].assumption_events.extend(expand_log_events);
                    }
                }

                if effective_opts.const_fold == crate::const_fold::ConstFoldMode::Safe {
                    let mut budget = crate::budget::Budget::preset_cli();
                    // Use effective_opts to propagate value_domain to const_fold
                    let cfg = crate::semantics::EvalConfig {
                        value_domain: effective_opts.value_domain,
                        branch: effective_opts.branch,
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
                        effective_opts.value_domain,
                    );
                    let global_requires: Vec<_> =
                        input_domain.conditions().iter().cloned().collect();
                    let mut dc = DomainContext::new(global_requires);

                    // Classify each step's assumption events
                    for step in steps.iter_mut() {
                        classify_assumptions_in_place(
                            &self.simplifier.context,
                            &mut dc,
                            &mut step.assumption_events,
                        );
                    }
                }

                // Collect domain warnings (now filtered by classification)
                let mut warnings = collect_domain_warnings(&steps);

                // Add warning if i is used in RealOnly mode
                if effective_opts.value_domain == crate::semantics::ValueDomain::RealOnly
                    && crate::helpers::contains_i(&self.simplifier.context, resolved)
                {
                    let i_warning = DomainWarning {
                        message:
                            "To use complex arithmetic (i² = -1), run: semantics set value complex"
                                .to_string(),
                        rule_name: "Imaginary Usage Warning".to_string(),
                    };
                    // Only add if not already present
                    if !warnings.iter().any(|w| w.message == i_warning.message) {
                        warnings.push(i_warning);
                    }
                }

                (
                    EvalResult::Expr(res),
                    warnings,
                    steps,
                    vec![],
                    vec![],
                    vec![],
                    vec![], // solver_required: empty for simplify
                )
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
            EvalAction::Solve { var } => {
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
                    value_domain: state.options.value_domain,
                    domain_mode: state.options.domain_mode,
                    assume_scope: state.options.assume_scope,
                    budget: state.options.budget,
                    ..Default::default()
                };

                // RAII guard for assumption collection (handles nested solves safely)
                let collect_assumptions = state.options.assumption_reporting
                    != crate::assumptions::AssumptionReporting::Off;
                let assumption_guard =
                    crate::solver::SolveAssumptionsGuard::new(collect_assumptions);

                // V2.9.8: Use type-safe API that guarantees cleanup is applied
                // detailed_steps=true for Normal+ verbosity (caller's responsibility to set)
                let sol_result = crate::solver::solve_with_display_steps(
                    &eq_to_solve,
                    &var,
                    &mut self.simplifier,
                    solver_opts,
                );

                // Collect assumptions (guard restores previous collector on drop)
                let solver_assumptions = assumption_guard.finish();

                match sol_result {
                    Ok((solution_set, display_steps)) => {
                        // V2.9.8: Extract Vec<SolveStep> from DisplaySolveSteps wrapper
                        // Steps are guaranteed to be post-cleanup
                        let solve_steps = display_steps.0;

                        // V2.0: Return the full SolutionSet preserving all variants
                        // including Conditional for proper REPL display
                        let warnings: Vec<DomainWarning> = vec![];
                        let eval_res = EvalResult::SolutionSet(solution_set);
                        // Collect output scopes from solver (e.g., QuadraticFormula)
                        let output_scopes = crate::solver::take_scopes();

                        // V2.2+: Collect required conditions from solver's domain env
                        // These are the structural domain constraints proven during solve
                        // Note: Must use take_solver_required() because RAII guard clears TLS on solve exit
                        let solver_required: Vec<crate::implicit_domain::ImplicitCondition> =
                            crate::solver::take_solver_required();

                        (
                            eval_res,
                            warnings,
                            vec![],
                            solve_steps,
                            solver_assumptions,
                            output_scopes,
                            solver_required,
                        )
                    }
                    Err(e) => return Err(anyhow::anyhow!("Solver error: {}", e)),
                }
            }
            EvalAction::Equiv { other } => {
                let resolved_other = match state.resolve_all(&mut self.simplifier.context, other) {
                    Ok(r) => r,
                    Err(ResolveError::CircularReference(msg)) => {
                        return Err(anyhow::anyhow!(
                            "Circular reference detected in other: {}",
                            msg
                        ))
                    }
                    Err(e) => return Err(anyhow::anyhow!("Resolution error in other: {}", e)),
                };

                let are_eq = self.simplifier.are_equivalent(resolved, resolved_other);
                (
                    EvalResult::Bool(are_eq),
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![], // solver_required: empty for equiv
                )
            }
        };

        // V2.15.36: Prepend synthetic cache hit step if any refs were resolved from cache
        if let Some(step) = cache_hit_step {
            steps.insert(0, step);
        }

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
                for cond in &step.required_conditions {
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
            for event in &step.assumption_events {
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
            state.store.update_diagnostics(id, diagnostics.clone());

            // V2.15.36: Populate simplified cache for session reference caching
            // This enables `#N` to use the cached simplified result instead of re-simplifying
            if let EvalResult::Expr(simplified_expr) = &result {
                use crate::session::{SimplifiedCache, SimplifyCacheKey};

                let cache_key = SimplifyCacheKey::from_context(state.options.domain_mode);
                let cache = SimplifiedCache {
                    key: cache_key,
                    expr: *simplified_expr,
                    requires: diagnostics.requires.clone(),
                    steps: Some(std::sync::Arc::new(steps.clone())),
                };
                state.store.update_simplified(id, cache);
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
            parsed: req.parsed,
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
