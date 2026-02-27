pub mod check;
pub(crate) mod isolation;
pub(crate) mod linear_collect;
pub(crate) mod numeric_islands;
pub(crate) mod reciprocal_solve;
pub(crate) mod solve_core;
pub(crate) mod strategies;
pub(crate) mod strategy;
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::solve_budget::SolveBudget;
pub use cas_solver_core::verify_stats;

#[cfg(test)]
use crate::engine::Simplifier;
#[cfg(test)]
use cas_ast::SolutionSet;
use cas_ast::{Equation, ExprId};
use cas_solver_core::shared_sink::SharedSink;
use std::collections::HashSet;

pub use self::solve_core::{solve, solve_with_display_steps};

/// Solver context — threaded explicitly through the solve pipeline.
///
/// Holds per-invocation state that was formerly stored in TLS,
/// enabling clean reentrancy for recursive/nested solves.
///
/// The `required_sink` is a shared accumulator so recursive sub-solves
/// contribute conditions to the same set.
/// `solve_with_display_steps` creates one, and every recursive
/// `solve_with_ctx_and_options` / `solve_with_options` pushes into it.
#[derive(Debug, Clone)]
pub struct SolveCtx {
    /// Domain environment inferred from equation structure (per-level).
    pub domain_env: SolveDomainEnv,
    /// Solve recursion depth for this context branch.
    depth: usize,
    /// Shared accumulator for required conditions across all recursive levels.
    required_sink: SharedSink<HashSet<crate::implicit_domain::ImplicitCondition>>,
    /// Shared accumulator for solver assumption events across recursive levels.
    assumptions_sink: SharedSink<Vec<crate::assumptions::AssumptionEvent>>,
    /// Shared collector for output scope tags across recursive levels.
    output_scopes_sink: SharedSink<Vec<cas_formatter::display_transforms::ScopeTag>>,
}

impl Default for SolveCtx {
    fn default() -> Self {
        Self {
            domain_env: SolveDomainEnv::default(),
            depth: 0,
            required_sink: SharedSink::new(HashSet::new()),
            assumptions_sink: SharedSink::new(Vec::new()),
            output_scopes_sink: SharedSink::new(Vec::new()),
        }
    }
}

impl SolveCtx {
    /// Build a child context that shares accumulators and bumps solve depth.
    pub(crate) fn fork_with_domain_env_next_depth(&self, domain_env: SolveDomainEnv) -> Self {
        Self {
            domain_env,
            depth: self.depth.saturating_add(1),
            required_sink: self.required_sink.clone(),
            assumptions_sink: self.assumptions_sink.clone(),
            output_scopes_sink: self.output_scopes_sink.clone(),
        }
    }

    /// Current solve recursion depth.
    pub(crate) fn depth(&self) -> usize {
        self.depth
    }

    /// Record one required domain condition in the shared accumulator.
    pub(crate) fn note_required_condition(
        &self,
        condition: crate::implicit_domain::ImplicitCondition,
    ) {
        self.required_sink.with_mut(|required| {
            required.insert(condition);
        });
    }

    /// Snapshot all required conditions accumulated by this solve tree.
    pub(crate) fn required_conditions(&self) -> Vec<crate::implicit_domain::ImplicitCondition> {
        self.required_sink
            .with(|required| required.iter().cloned().collect())
    }

    /// Record an assumption emitted during solve.
    pub(crate) fn note_assumption(&self, event: crate::assumptions::AssumptionEvent) {
        self.assumptions_sink
            .with_mut(|assumptions| assumptions.push(event));
    }

    /// Snapshot collected solver assumptions.
    pub(crate) fn assumptions(&self) -> Vec<crate::assumptions::AssumptionEvent> {
        self.assumptions_sink
            .with(|assumptions| assumptions.clone())
    }

    /// Emit a scope tag used by output display transforms.
    pub(crate) fn emit_scope(&self, scope: cas_formatter::display_transforms::ScopeTag) {
        self.output_scopes_sink.with_mut(|scopes| {
            if !scopes.contains(&scope) {
                scopes.push(scope);
            }
        });
    }

    /// Snapshot collected output scopes.
    pub(crate) fn output_scopes(&self) -> Vec<cas_formatter::display_transforms::ScopeTag> {
        self.output_scopes_sink.with(|scopes| scopes.clone())
    }
}

/// Convert one solver-core log-domain assumption to an engine assumption event.
pub(crate) fn assumption_event_from_log_assumption_targets(
    ctx: &cas_ast::Context,
    assumption: cas_solver_core::log_domain::LogAssumption,
    base: ExprId,
    rhs: ExprId,
) -> crate::assumptions::AssumptionEvent {
    let target = cas_solver_core::log_domain::assumption_target_expr(assumption, base, rhs);
    crate::assumptions::AssumptionEvent::positive(ctx, target)
}

/// Convert one solver-core blocked-hint record to an engine assumption event.
pub(crate) fn assumption_event_from_log_blocked_hint(
    ctx: &cas_ast::Context,
    hint: cas_solver_core::solve_outcome::LogBlockedHintRecord,
) -> crate::assumptions::AssumptionEvent {
    match hint.assumption {
        cas_solver_core::log_domain::LogAssumption::PositiveBase
        | cas_solver_core::log_domain::LogAssumption::PositiveRhs => {
            crate::assumptions::AssumptionEvent::positive(ctx, hint.expr_id)
        }
    }
}

/// Convert one solver-core blocked-hint record to a domain blocked hint.
pub(crate) fn domain_blocked_hint_from_log_blocked_hint(
    ctx: &cas_ast::Context,
    hint: cas_solver_core::solve_outcome::LogBlockedHintRecord,
) -> crate::domain::BlockedHint {
    let event = assumption_event_from_log_blocked_hint(ctx, hint);
    crate::domain::BlockedHint {
        key: event.key,
        expr_id: hint.expr_id,
        rule: hint.rule.to_string(),
        suggestion: hint.suggestion,
    }
}

/// Options for solver operations, containing semantic context.
///
/// This struct passes value domain and domain mode information to the solver,
/// enabling domain-aware decisions like rejecting log operations on negative bases.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    /// The value domain (RealOnly or ComplexEnabled)
    pub value_domain: crate::semantics::ValueDomain,
    /// The domain mode (Strict, Assume, Generic)
    pub domain_mode: crate::domain::DomainMode,
    /// Scope for assumptions (only active if domain_mode=Assume)
    pub assume_scope: crate::semantics::AssumeScope,
    /// V2.0: Budget for conditional branching (anti-explosion)
    pub budget: SolveBudget,
    /// V2.9.8: If true, generate detailed step narrative (5 atomic steps).
    /// If false, generate compact narrative (3 steps for Succinct verbosity).
    pub detailed_steps: bool,
}

// =============================================================================
// Type-Safe Step Pipeline (V2.9.8)
// =============================================================================
// These newtypes enforce that renderers only consume post-processed steps.
// This eliminates bifurcation between text/timeline outputs at compile time.

/// Display-ready solve steps after didactic cleanup and narration.
/// All renderers (text, timeline, JSON) consume this type only.
#[derive(Debug, Clone)]
pub struct DisplaySolveSteps(pub Vec<SolveStep>);

impl DisplaySolveSteps {
    /// Check if there are no steps.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the number of steps.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Iterate over steps.
    pub fn iter(&self) -> std::slice::Iter<'_, SolveStep> {
        self.0.iter()
    }

    /// Get inner Vec reference.
    pub fn as_slice(&self) -> &[SolveStep] {
        &self.0
    }

    /// Consume and return inner Vec.
    pub fn into_inner(self) -> Vec<SolveStep> {
        self.0
    }
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            value_domain: crate::semantics::ValueDomain::RealOnly,
            domain_mode: crate::domain::DomainMode::Generic,
            assume_scope: crate::semantics::AssumeScope::Real,
            budget: SolveBudget::default(),
            detailed_steps: true, // V2.9.8: Default to detailed (Normal/Verbose)
        }
    }
}

impl SolverOptions {
    /// Convert engine domain mode into core solver domain mode kind.
    pub(crate) fn core_domain_mode(&self) -> cas_solver_core::log_domain::DomainModeKind {
        match self.domain_mode {
            crate::domain::DomainMode::Strict => {
                cas_solver_core::log_domain::DomainModeKind::Strict
            }
            crate::domain::DomainMode::Generic => {
                cas_solver_core::log_domain::DomainModeKind::Generic
            }
            crate::domain::DomainMode::Assume => {
                cas_solver_core::log_domain::DomainModeKind::Assume
            }
        }
    }

    /// Returns true when assume-scope allows wildcard assumptions.
    pub(crate) fn wildcard_scope(&self) -> bool {
        self.assume_scope == crate::semantics::AssumeScope::Wildcard
    }
}

/// Map engine proof classification to solver-core non-zero status.
pub(crate) fn prove_nonzero_status(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> cas_solver_core::linear_solution::NonZeroStatus {
    match crate::helpers::prove_nonzero(ctx, expr) {
        crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit => {
            cas_solver_core::linear_solution::NonZeroStatus::NonZero
        }
        crate::domain::Proof::Unknown => cas_solver_core::linear_solution::NonZeroStatus::Unknown,
        crate::domain::Proof::Disproven => cas_solver_core::linear_solution::NonZeroStatus::Zero,
    }
}

fn proof_status_from_proof(
    proof: crate::domain::Proof,
) -> cas_solver_core::log_domain::ProofStatus {
    match proof {
        crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit => {
            cas_solver_core::log_domain::ProofStatus::Proven
        }
        crate::domain::Proof::Unknown => cas_solver_core::log_domain::ProofStatus::Unknown,
        crate::domain::Proof::Disproven => cas_solver_core::log_domain::ProofStatus::Disproven,
    }
}

/// Classify whether a logarithmic solve step (`base^x = rhs`) is valid.
pub(crate) fn classify_log_solve(
    ctx: &cas_ast::Context,
    base: ExprId,
    rhs: ExprId,
    opts: &SolverOptions,
    env: &SolveDomainEnv,
) -> cas_solver_core::log_domain::LogSolveDecision {
    cas_solver_core::log_domain::classify_log_solve_with_prover(
        ctx,
        base,
        rhs,
        opts.value_domain == crate::semantics::ValueDomain::RealOnly,
        opts.core_domain_mode(),
        env.has_positive(base),
        env.has_positive(rhs),
        |core_ctx, expr| {
            proof_status_from_proof(crate::helpers::prove_positive(
                core_ctx,
                expr,
                opts.value_domain,
            ))
        },
    )
}

// =============================================================================
// Domain Environment (V2.2+)
// =============================================================================

/// Domain environment for solver operations.
///
/// Contains the "semantic ground" under which the solver operates:
/// - `required`: Constraints inferred from equation structure (e.g., sqrt(y) → y ≥ 0)
///
/// This is passed explicitly rather than via TLS for clean reentrancy and testability.
#[derive(Debug, Clone, Default)]
pub struct SolveDomainEnv {
    /// Required conditions inferred from equation structure.
    /// These are NOT assumptions - they represent the minimum domain of validity.
    pub required: crate::implicit_domain::ImplicitDomain,
}

impl SolveDomainEnv {
    /// Create a new empty environment
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a Positive condition is already in the required set
    pub fn has_positive(&self, expr: cas_ast::ExprId) -> bool {
        self.required.contains_positive(expr)
    }

    /// Check if a NonNegative condition is already in the required set
    pub fn has_nonnegative(&self, expr: cas_ast::ExprId) -> bool {
        self.required.contains_nonnegative(expr)
    }

    /// Check if a NonZero condition is already in the required set
    pub fn has_nonzero(&self, expr: cas_ast::ExprId) -> bool {
        self.required.contains_nonzero(expr)
    }

    /// Convert required conditions to a ConditionSet for use with guards
    pub fn required_as_condition_set(&self) -> cas_ast::ConditionSet {
        self.required.to_condition_set()
    }
}

/// Diagnostics collected during solve operation.
///
/// This is returned alongside solutions to provide transparency about
/// what conditions were required vs assumed during solving.
#[derive(Debug, Clone, Default)]
pub struct SolveDiagnostics {
    /// Conditions required by the equation structure (domain minimum)
    pub required: Vec<crate::implicit_domain::ImplicitCondition>,
    /// Assumptions made during solving (policy decisions)
    pub assumed: Vec<crate::assumptions::AssumptionEvent>,
    /// Deduplicated assumption summary records for external output.
    pub assumed_records: Vec<crate::assumptions::AssumptionRecord>,
    /// Output scopes for context-aware display transforms.
    pub output_scopes: Vec<cas_formatter::display_transforms::ScopeTag>,
}

impl SolveDiagnostics {
    /// Create empty diagnostics
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert to Vec for display
    pub fn required_display(&self, ctx: &cas_ast::Context) -> Vec<String> {
        self.required.iter().map(|c| c.display(ctx)).collect()
    }

    /// Convert to Vec for display
    pub fn assumed_display(&self) -> Vec<String> {
        self.assumed.iter().map(|a| a.message.clone()).collect()
    }
}

/// Educational sub-step for solver derivations (e.g., completing the square)
/// Displayed as indented in REPL and collapsible in timeline.
#[derive(Debug, Clone)]
pub struct SolveSubStep {
    /// Description of the substep (e.g., "Divide both sides by a")
    pub description: String,
    /// The equation state after this substep
    pub equation_after: Equation,
    /// Importance level for verbosity filtering
    pub importance: crate::step::ImportanceLevel,
}

impl SolveSubStep {
    /// Create a new SolveSubStep with Low importance (educational detail)
    pub fn new(description: impl Into<String>, equation_after: Equation) -> Self {
        Self {
            description: description.into(),
            equation_after,
            importance: crate::step::ImportanceLevel::Low,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SolveStep {
    pub description: String,
    pub equation_after: Equation,
    /// Importance level for step filtering (matches Step::importance system)
    /// Default: Medium (visible in Normal verbosity)
    pub importance: crate::step::ImportanceLevel,
    /// Educational sub-steps explaining the derivation (e.g., completing the square)
    /// Displayed as indented in REPL and collapsible in timeline
    pub substeps: Vec<SolveSubStep>,
}

impl SolveStep {
    /// Create a new SolveStep with default Medium importance (visible in Normal mode)
    pub fn new(description: impl Into<String>, equation_after: Equation) -> Self {
        Self {
            description: description.into(),
            equation_after,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        }
    }

    /// Create a SolveStep with Low importance (visible only in Verbose mode)
    pub fn with_low_importance(description: impl Into<String>, equation_after: Equation) -> Self {
        Self {
            description: description.into(),
            equation_after,
            importance: crate::step::ImportanceLevel::Low,
            substeps: vec![],
        }
    }

    /// Create a SolveStep with High importance (key solver steps)
    pub fn with_high_importance(description: impl Into<String>, equation_after: Equation) -> Self {
        Self {
            description: description.into(),
            equation_after,
            importance: crate::step::ImportanceLevel::High,
            substeps: vec![],
        }
    }

    /// Add substeps to this step (builder pattern)
    pub fn with_substeps(mut self, substeps: Vec<SolveSubStep>) -> Self {
        self.substeps = substeps;
        self
    }
}

pub(crate) fn medium_step(description: String, equation_after: Equation) -> SolveStep {
    SolveStep::new(description, equation_after)
}

pub(crate) fn render_expr(ctx: &cas_ast::Context, expr: ExprId) -> String {
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expr
        }
    )
}

// ---------------------------------------------------------------------------
// Recursion depth limit
// ---------------------------------------------------------------------------

/// Maximum recursion depth for solver to prevent stack overflow
pub(crate) const MAX_SOLVE_DEPTH: usize = 50;

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{BoundType, Context, RelOp};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;
    use cas_solver_core::log_domain::{LogAssumption, LogSolveDecision};

    // Helper to make equation from strings
    fn make_eq(ctx: &mut Context, lhs: &str, rhs: &str) -> Equation {
        Equation {
            lhs: parse(lhs, ctx).unwrap(),
            rhs: parse(rhs, ctx).unwrap(),
            op: RelOp::Eq,
        }
    }

    #[test]
    fn test_solve_ctx_fork_shares_required_conditions() {
        let mut context = Context::new();
        let x = parse("x", &mut context).unwrap();

        let parent = SolveCtx::default();
        parent.note_required_condition(crate::implicit_domain::ImplicitCondition::NonZero(x));

        let child = parent.fork_with_domain_env_next_depth(SolveDomainEnv::default());
        child.note_required_condition(crate::implicit_domain::ImplicitCondition::Positive(x));

        let required = parent.required_conditions();
        assert_eq!(required.len(), 2);
    }

    #[test]
    fn test_solve_ctx_fork_shares_assumptions_and_scopes() {
        let mut context = Context::new();
        let x = parse("x", &mut context).unwrap();

        let parent = SolveCtx::default();
        let child = parent.fork_with_domain_env_next_depth(SolveDomainEnv::default());
        child.note_assumption(crate::assumptions::AssumptionEvent::positive(&context, x));
        child.emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
            "QuadraticFormula",
        ));

        assert_eq!(parent.assumptions().len(), 1);
        assert_eq!(parent.output_scopes().len(), 1);
    }

    #[test]
    fn test_solve_ctx_fork_increments_depth() {
        let parent = SolveCtx::default();
        assert_eq!(parent.depth(), 0);

        let child = parent.fork_with_domain_env_next_depth(SolveDomainEnv::default());
        let grandchild = child.fork_with_domain_env_next_depth(SolveDomainEnv::default());

        assert_eq!(child.depth(), 1);
        assert_eq!(grandchild.depth(), 2);
    }

    #[test]
    fn test_solve_linear() {
        // x + 2 = 5 -> x = 3
        let mut simplifier = Simplifier::new();
        let eq = make_eq(&mut simplifier.context, "x + 2", "5");
        simplifier.set_collect_steps(true);
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 1);
            let s = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: solutions[0]
                }
            );
            assert_eq!(s, "3");
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_mul() {
        // 2 * x = 6 -> x = 6 / 2
        let mut simplifier = Simplifier::with_default_rules();
        let eq = make_eq(&mut simplifier.context, "2 * x", "6");
        simplifier.set_collect_steps(true);
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 1);
            let s = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: solutions[0]
                }
            );
            assert_eq!(s, "3");
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_pow() {
        // x^2 = 4 -> x = 4^(1/2)
        let mut simplifier = Simplifier::new();
        let eq = make_eq(&mut simplifier.context, "x^2", "4");
        simplifier.add_rule(Box::new(crate::rules::exponents::EvaluatePowerRule));
        simplifier.add_rule(Box::new(
            crate::rules::canonicalization::CanonicalizeNegationRule,
        ));
        simplifier.add_rule(Box::new(crate::rules::arithmetic::CombineConstantsRule));
        simplifier.set_collect_steps(true);
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Discrete(mut solutions) = result {
            assert_eq!(solutions.len(), 2);
            // Sort to ensure order
            solutions.sort_by(|a, b| {
                let sa = format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: *a
                    }
                );
                let sb = format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: *b
                    }
                );
                sa.cmp(&sb)
            });

            let s1 = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: solutions[0]
                }
            );
            let s2 = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: solutions[1]
                }
            );

            // We want to eventually see "-2" and "2".
            assert_eq!(s1, "-2");
            assert_eq!(s2, "2");
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_abs() {
        // |x| = 5 -> x=5, x=-5
        let mut simplifier = Simplifier::new();
        let eq = make_eq(&mut simplifier.context, "|x|", "5");
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 2);
            // Order might vary
            let s: Vec<String> = solutions
                .iter()
                .map(|e| {
                    format!(
                        "{}",
                        DisplayExpr {
                            context: &simplifier.context,
                            id: *e
                        }
                    )
                })
                .collect();
            assert!(s.contains(&"5".to_string()));
            assert!(s.contains(&"-5".to_string()));
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_inequality_flip() {
        // -2x < 10 -> x > -5
        let mut simplifier = Simplifier::with_default_rules();
        let eq = Equation {
            lhs: parse("-2*x", &mut simplifier.context).unwrap(),
            rhs: parse("10", &mut simplifier.context).unwrap(),
            op: RelOp::Lt,
        };
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Continuous(interval) = result {
            // (-5, inf)
            let s_min = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: interval.min
                }
            );
            let s_max = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: interval.max
                }
            );
            assert!(
                s_min == "-5" || s_min == "10 / -2",
                "Expected -5 or canonical form 10 / -2, got: {}",
                s_min
            );
            assert_eq!(interval.min_type, BoundType::Open);
            assert_eq!(s_max, "infinity");
        } else {
            panic!("Expected Continuous solution, got {:?}", result);
        }
    }

    #[test]
    fn test_solve_abs_inequality() {
        // |x| < 5 -> (-5, 5)
        let mut simplifier = Simplifier::new();
        let eq = Equation {
            lhs: parse("|x|", &mut simplifier.context).unwrap(),
            rhs: parse("5", &mut simplifier.context).unwrap(),
            op: RelOp::Lt,
        };
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Continuous(interval) = result {
            let s_min = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: interval.min
                }
            );
            let s_max = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: interval.max
                }
            );
            assert_eq!(s_min, "-5");
            assert_eq!(s_max, "5");
        } else {
            panic!("Expected Continuous solution, got {:?}", result);
        }
    }

    fn classify_for_test(
        ctx: &Context,
        base: ExprId,
        rhs: ExprId,
        mode: crate::domain::DomainMode,
    ) -> LogSolveDecision {
        let opts = SolverOptions {
            domain_mode: mode,
            ..Default::default()
        };
        classify_log_solve(ctx, base, rhs, &opts, &SolveDomainEnv::default())
    }

    #[test]
    fn test_log_solve_both_proven_positive_ok() {
        let mut ctx = Context::new();
        let base = ctx.num(2);
        let rhs = ctx.num(8);
        let decision = classify_for_test(&ctx, base, rhs, crate::domain::DomainMode::Generic);
        assert!(matches!(decision, LogSolveDecision::Ok));
    }

    #[test]
    fn test_log_solve_negative_rhs_empty() {
        let mut ctx = Context::new();
        let base = ctx.num(2);
        let rhs = ctx.num(-5);
        let decision = classify_for_test(&ctx, base, rhs, crate::domain::DomainMode::Generic);
        assert!(matches!(decision, LogSolveDecision::EmptySet(_)));
    }

    #[test]
    fn test_log_solve_negative_base_needs_complex() {
        let mut ctx = Context::new();
        let base = ctx.num(-2);
        let rhs = ctx.num(5);
        let decision = classify_for_test(&ctx, base, rhs, crate::domain::DomainMode::Generic);
        assert!(matches!(decision, LogSolveDecision::NeedsComplex(_)));
    }

    #[test]
    fn test_log_solve_assume_unknown_rhs_emits_assumption() {
        let mut ctx = Context::new();
        let base = ctx.num(2);
        let rhs = ctx.var("y");
        let decision = classify_for_test(&ctx, base, rhs, crate::domain::DomainMode::Assume);
        match decision {
            LogSolveDecision::OkWithAssumptions(assumptions) => {
                assert!(assumptions.contains(&LogAssumption::PositiveRhs));
            }
            _ => panic!("Expected OkWithAssumptions, got {:?}", decision),
        }
    }

    #[test]
    fn test_log_solve_generic_unknown_rhs_unsupported() {
        let mut ctx = Context::new();
        let base = ctx.num(2);
        let rhs = ctx.var("y");
        let decision = classify_for_test(&ctx, base, rhs, crate::domain::DomainMode::Generic);
        assert!(matches!(decision, LogSolveDecision::Unsupported(_, _)));
    }

    #[test]
    fn test_log_solve_neg_expr_rhs_empty() {
        let mut ctx = Context::new();
        let base = ctx.num(2);
        let five = ctx.num(5);
        let rhs = ctx.add(cas_ast::Expr::Neg(five));
        let decision = classify_for_test(&ctx, base, rhs, crate::domain::DomainMode::Generic);
        assert!(
            matches!(decision, LogSolveDecision::EmptySet(_)),
            "Expected EmptySet for Neg(5), got {:?}",
            decision
        );
    }

    #[test]
    fn test_assumption_event_from_log_assumption_targets_maps_base_and_rhs() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let rhs = ctx.var("r");

        let base_event = assumption_event_from_log_assumption_targets(
            &ctx,
            cas_solver_core::log_domain::LogAssumption::PositiveBase,
            base,
            rhs,
        );
        let rhs_event = assumption_event_from_log_assumption_targets(
            &ctx,
            cas_solver_core::log_domain::LogAssumption::PositiveRhs,
            base,
            rhs,
        );

        assert_eq!(
            base_event.expr_id,
            Some(base),
            "PositiveBase must target base expression",
        );
        assert_eq!(
            rhs_event.expr_id,
            Some(rhs),
            "PositiveRhs must target rhs expression",
        );
    }

    #[test]
    fn test_assumption_event_from_log_blocked_hint_uses_hint_expression() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let other = ctx.var("o");
        let hint = cas_solver_core::solve_outcome::LogBlockedHintRecord {
            assumption: cas_solver_core::log_domain::LogAssumption::PositiveBase,
            expr_id: base,
            rule: "Take log of both sides",
            suggestion: "use `semantics set domain assume`",
        };

        let event = assumption_event_from_log_blocked_hint(&ctx, hint);
        let expected = crate::assumptions::AssumptionEvent::positive(&ctx, base);

        assert_eq!(event.expr_id, Some(base));
        assert_eq!(event.key, expected.key);

        let rhs_hint = cas_solver_core::solve_outcome::LogBlockedHintRecord {
            assumption: cas_solver_core::log_domain::LogAssumption::PositiveRhs,
            expr_id: other,
            rule: "Take log of both sides",
            suggestion: "use `semantics set domain assume`",
        };
        let rhs_event = assumption_event_from_log_blocked_hint(&ctx, rhs_hint);
        assert_eq!(rhs_event.expr_id, Some(other));
    }

    #[test]
    fn test_domain_blocked_hint_from_log_blocked_hint_maps_payload() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let hint = cas_solver_core::solve_outcome::LogBlockedHintRecord {
            assumption: cas_solver_core::log_domain::LogAssumption::PositiveBase,
            expr_id: base,
            rule: "Take log of both sides",
            suggestion: "use `semantics set domain assume`",
        };

        let blocked = domain_blocked_hint_from_log_blocked_hint(&ctx, hint);
        let expected_key = crate::assumptions::AssumptionEvent::positive(&ctx, base).key;

        assert_eq!(blocked.key, expected_key);
        assert_eq!(blocked.expr_id, base);
        assert_eq!(blocked.rule, "Take log of both sides");
        assert_eq!(blocked.suggestion, "use `semantics set domain assume`");
    }
}
