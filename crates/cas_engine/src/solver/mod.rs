use crate::build::mul2_raw;
pub mod check;
pub mod domain_guards;
pub mod isolation;
pub mod linear_collect;
pub mod log_linear_narrator;
pub mod quadratic_steps;
pub mod reciprocal_solve;
pub mod solution_set;
pub mod step_cleanup;
pub mod strategies;
pub mod strategy;

use crate::engine::Simplifier;
use cas_ast::{Context, Equation, ExprId, SolutionSet};

pub use self::isolation::contains_var;

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

/// Raw solve steps as produced by solver strategies (internal only).
/// Contains the raw step sequence before didactic cleanup.
/// Reserved for future internal usage when solve_with_options is deprecated.
#[allow(dead_code)]
pub struct RawSolveSteps(pub(crate) Vec<SolveStep>);

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

/// V2.0 Phase 2A: Budget for conditional solution branching.
///
/// Controls how many conditional branches the solver can create,
/// preventing combinatorial explosion in complex equations.
#[derive(Debug, Clone, Copy)]
pub struct SolveBudget {
    /// Maximum number of branches that can be created (0 = no branching allowed)
    pub max_branches: usize,
    /// Maximum nesting depth for conditional solutions
    pub max_depth: usize,
}

impl Default for SolveBudget {
    fn default() -> Self {
        Self {
            max_branches: 1,
            max_depth: 2,
        }
    }
}

impl SolveBudget {
    /// No branching allowed - always return residual
    pub fn none() -> Self {
        Self {
            max_branches: 0,
            max_depth: 0,
        }
    }

    /// Check if branching is allowed
    pub fn can_branch(&self) -> bool {
        self.max_branches > 0
    }

    /// Consume one branch, returning remaining budget
    pub fn consume_branch(self) -> Self {
        Self {
            max_branches: self.max_branches.saturating_sub(1),
            ..self
        }
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

// =============================================================================
// Domain Environment (V2.2+)
// =============================================================================

/// Domain environment for solver operations.
///
/// Contains the "semantic ground" under which the solver operates:
/// - `required`: Constraints inferred from equation structure (e.g., sqrt(y) → y ≥ 0)
/// - `assumed`: Constraints assumed by policy (only in Assume mode)
///
/// This is passed explicitly rather than via TLS for clean reentrancy and testability.
#[derive(Debug, Clone, Default)]
pub struct SolveDomainEnv {
    /// Required conditions inferred from equation structure.
    /// These are NOT assumptions - they represent the minimum domain of validity.
    pub required: crate::implicit_domain::ImplicitDomain,
    /// Assumed conditions made during solving (only populated in Assume mode).
    pub assumed: Vec<crate::assumptions::AssumptionEvent>,
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

use crate::error::CasError;

use crate::solver::strategies::{
    CollectTermsStrategy, IsolationStrategy, QuadraticStrategy, RationalExponentStrategy,
    SubstitutionStrategy, UnwrapStrategy,
};
use crate::solver::strategy::SolverStrategy;

/// Maximum recursion depth for solver to prevent stack overflow
pub(crate) const MAX_SOLVE_DEPTH: usize = 50;

thread_local! {
    pub(crate) static SOLVE_DEPTH: std::cell::RefCell<usize> = const { std::cell::RefCell::new(0) };
    /// Thread-local collector for solver assumptions.
    /// Used to pass assumptions from strategies back to caller without changing return type.
    static SOLVE_ASSUMPTIONS: std::cell::RefCell<Option<crate::assumptions::AssumptionCollector>> =
        const { std::cell::RefCell::new(None) };
    /// Thread-local collector for output scopes (display context).
    /// Strategies emit scopes like "QuadraticFormula" which affect display transforms.
    static OUTPUT_SCOPES: std::cell::RefCell<Vec<cas_ast::display_transforms::ScopeTag>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Thread-local current domain environment for solver.
    /// Set by solve_with_options, consulted by classify_log_solve via get_current_domain_env().
    static CURRENT_DOMAIN_ENV: std::cell::RefCell<Option<SolveDomainEnv>> =
        const { std::cell::RefCell::new(None) };
}

/// Get the current domain environment (if set by an enclosing solve).
/// Used by classify_log_solve to check if conditions are already proven.
pub fn get_current_domain_env() -> Option<SolveDomainEnv> {
    CURRENT_DOMAIN_ENV.with(|e| e.borrow().clone())
}

/// Set the current domain environment (called by solve_with_options).
fn set_current_domain_env(env: SolveDomainEnv) {
    CURRENT_DOMAIN_ENV.with(|e| {
        *e.borrow_mut() = Some(env);
    });
}

/// Clear the current domain environment (called on solve exit).
fn clear_current_domain_env() {
    CURRENT_DOMAIN_ENV.with(|e| {
        // Before clearing, save to LAST_SOLVER_REQUIRED for eval.rs to retrieve
        if let Some(ref env) = *e.borrow() {
            LAST_SOLVER_REQUIRED.with(|last| {
                *last.borrow_mut() = env.required.conditions().iter().cloned().collect();
            });
        }
        *e.borrow_mut() = None;
    });
}

thread_local! {
    /// Thread-local storage for required conditions from the last completed solve.
    /// Set by clear_current_domain_env before clearing the main env.
    /// Retrieved by eval.rs via take_solver_required().
    static LAST_SOLVER_REQUIRED: std::cell::RefCell<Vec<crate::implicit_domain::ImplicitCondition>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Take the required conditions from the last completed solve.
/// This is called by eval.rs to get solver-derived requirements after solve completes.
pub fn take_solver_required() -> Vec<crate::implicit_domain::ImplicitCondition> {
    LAST_SOLVER_REQUIRED.with(|last| std::mem::take(&mut *last.borrow_mut()))
}

/// RAII guard for solver assumption collection.
///
/// Creates a fresh collector on creation, and on drop:
/// - Returns the collected assumptions via `finish()`
/// - Restores any previous collector (for nested solves)
///
/// # Safety against leaks/reentrancy
/// - Drop always clears or restores state
/// - Nested solves get their own collectors (previous is saved)
pub struct SolveAssumptionsGuard {
    /// The previous collector (if any) that was active before this guard
    previous: Option<crate::assumptions::AssumptionCollector>,
    /// Whether collection is enabled for this guard
    enabled: bool,
}

impl SolveAssumptionsGuard {
    /// Create a new guard that starts assumption collection.
    /// If `enabled` is false, no collection happens (passthrough).
    pub fn new(enabled: bool) -> Self {
        let previous = if enabled {
            // Take any existing collector (for nested solve case)
            let prev = SOLVE_ASSUMPTIONS.with(|c| c.borrow_mut().take());
            // Install fresh collector
            SOLVE_ASSUMPTIONS.with(|c| {
                *c.borrow_mut() = Some(crate::assumptions::AssumptionCollector::new());
            });
            prev
        } else {
            None
        };

        Self { previous, enabled }
    }

    /// Finish collection and return the collected records.
    /// This consumes the guard.
    pub fn finish(self) -> Vec<crate::assumptions::AssumptionRecord> {
        // The Drop impl will restore previous, we just need to take current
        if self.enabled {
            SOLVE_ASSUMPTIONS.with(|c| {
                c.borrow_mut()
                    .take()
                    .map(|collector| collector.finish())
                    .unwrap_or_default()
            })
        } else {
            vec![]
        }
    }
}

impl Drop for SolveAssumptionsGuard {
    fn drop(&mut self) {
        if self.enabled {
            // Restore previous collector (or None if there wasn't one)
            SOLVE_ASSUMPTIONS.with(|c| {
                *c.borrow_mut() = self.previous.take();
            });
        }
    }
}

/// Start assumption collection for solver.
/// DEPRECATED: Use SolveAssumptionsGuard for RAII safety.
/// Returns true if collection was started (false if already active).
pub fn start_assumption_collection() -> bool {
    SOLVE_ASSUMPTIONS.with(|c| {
        let mut collector = c.borrow_mut();
        if collector.is_none() {
            *collector = Some(crate::assumptions::AssumptionCollector::new());
            true
        } else {
            false
        }
    })
}

/// Finish assumption collection and return the collector.
/// DEPRECATED: Use SolveAssumptionsGuard for RAII safety.
/// Returns None if collection wasn't started.
pub fn finish_assumption_collection() -> Option<crate::assumptions::AssumptionCollector> {
    SOLVE_ASSUMPTIONS.with(|c| c.borrow_mut().take())
}

/// Note an assumption during solver operation (internal use).
pub(crate) fn note_assumption(event: crate::assumptions::AssumptionEvent) {
    SOLVE_ASSUMPTIONS.with(|c| {
        if let Some(ref mut collector) = *c.borrow_mut() {
            collector.note(event);
        }
    });
}

/// Emit a scope tag during solver operation for display transforms.
/// Called by strategies like QuadraticFormula to mark the result context.
pub fn emit_scope(scope: cas_ast::display_transforms::ScopeTag) {
    OUTPUT_SCOPES.with(|s| {
        let mut scopes = s.borrow_mut();
        // Avoid duplicates
        if !scopes.contains(&scope) {
            scopes.push(scope);
        }
    });
}

/// Take all emitted scopes, clearing the TLS collector.
/// Called after solve to get scopes for EvalOutput.
pub fn take_scopes() -> Vec<cas_ast::display_transforms::ScopeTag> {
    OUTPUT_SCOPES.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

/// Clear emitted scopes without returning them.
pub fn clear_scopes() {
    OUTPUT_SCOPES.with(|s| s.borrow_mut().clear());
}

/// Guard that decrements depth on drop
pub(crate) struct DepthGuard;

impl Drop for DepthGuard {
    fn drop(&mut self) {
        SOLVE_DEPTH.with(|d| {
            *d.borrow_mut() -= 1;
        });
    }
}

/// Solve with default options (for backward compatibility with tests).
/// Uses RealOnly domain and Generic mode.
pub fn solve(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    solve_with_options(eq, var, simplifier, SolverOptions::default())
}

/// V2.9.8: Solve with type-enforced display-ready steps.
///
/// This is the PREFERRED entry point for display-facing code (REPL, timeline, JSON API).
/// Returns `DisplaySolveSteps` which enforces that all renderers consume post-processed
/// steps, eliminating bifurcation between text/timeline outputs at compile time.
///
/// The cleanup is applied automatically based on `opts.detailed_steps`:
/// - `true` → 5 atomic sub-steps for Normal/Verbose verbosity
/// - `false` → 3 compact steps for Succinct verbosity
pub fn solve_with_display_steps(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, DisplaySolveSteps), CasError> {
    let (solution_set, raw_steps) = solve_with_options(eq, var, simplifier, opts)?;

    // Apply didactic cleanup using opts.detailed_steps
    let cleaned =
        step_cleanup::cleanup_solve_steps(&mut simplifier.context, raw_steps, opts.detailed_steps);

    Ok((solution_set, DisplaySolveSteps(cleaned)))
}

/// Internal: Solve an equation with explicit semantic options.
///
/// **For display-facing code**, use [`solve_with_display_steps`] instead.
/// This function returns raw steps that need cleanup before display.
///
/// `opts` contains ValueDomain and DomainMode which control:
/// - Whether log operations are valid (RealOnly requires positive arguments)
/// - Whether to emit assumptions or reject operations
pub(crate) fn solve_with_options(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // Check and increment recursion depth
    let current_depth = SOLVE_DEPTH.with(|d| {
        let mut depth = d.borrow_mut();
        *depth += 1;
        *depth
    });

    // Create guard to decrement on exit
    let _guard = DepthGuard;

    if current_depth > MAX_SOLVE_DEPTH {
        return Err(CasError::SolverError(
            "Maximum solver recursion depth exceeded. The equation may be too complex.".to_string(),
        ));
    }

    // 1. Check if variable exists in equation
    if !contains_var(&simplifier.context, eq.lhs, var)
        && !contains_var(&simplifier.context, eq.rhs, var)
    {
        return Err(CasError::VariableNotFound(var.to_string()));
    }

    // V2.1 Issue #10: Extract denominators containing the variable BEFORE simplification
    // These will be used to create NonZero guards in the final result
    let mut domain_exclusions: std::collections::HashSet<ExprId> = std::collections::HashSet::new();
    domain_exclusions.extend(extract_denominators_with_var(
        &simplifier.context,
        eq.lhs,
        var,
    ));
    domain_exclusions.extend(extract_denominators_with_var(
        &simplifier.context,
        eq.rhs,
        var,
    ));
    let domain_exclusions: Vec<ExprId> = domain_exclusions.into_iter().collect();

    // V2.2+: Build SolveDomainEnv with global required conditions
    // This is the "semantic ground" for all solver decisions
    let mut domain_env = SolveDomainEnv::new();
    {
        use crate::implicit_domain::{derive_requires_from_equation, infer_implicit_domain};

        // 1. Infer implicit domain from both sides
        let lhs_domain = infer_implicit_domain(&simplifier.context, eq.lhs, opts.value_domain);
        let rhs_domain = infer_implicit_domain(&simplifier.context, eq.rhs, opts.value_domain);
        domain_env.required.extend(&lhs_domain);
        domain_env.required.extend(&rhs_domain);

        // 2. Derive additional requires from equation equality
        // e.g., 2^x = sqrt(y) → since 2^x > 0, we have sqrt(y) > 0, so y > 0
        let derived = derive_requires_from_equation(
            &simplifier.context,
            eq.lhs,
            eq.rhs,
            &domain_env.required,
            opts.value_domain,
        );
        for cond in derived {
            domain_env.required.conditions_mut().insert(cond);
        }
    }

    // V2.2+: Set domain_env in TLS for strategies to access via get_current_domain_env()
    set_current_domain_env(domain_env);

    // RAII guard to clear env on exit (handles all return paths)
    struct EnvGuard;
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            clear_current_domain_env();
        }
    }
    let _env_guard = EnvGuard;

    // EARLY CHECK: Handle rational exponent equations BEFORE simplification
    // This prevents x^(3/2) from being simplified to |x|*sqrt(x) which causes loops
    if eq.op == cas_ast::RelOp::Eq {
        if let Some(result) = try_solve_rational_exponent(eq, var, simplifier) {
            // Wrap result with domain guards if needed
            return wrap_with_domain_guards(result, &domain_exclusions, simplifier);
        }
    }

    // 2. Simplify both sides BEFORE applying strategies
    // This is crucial for equations like "1/3*x + 1/2*x = 5"
    // which need to be simplified to "5/6*x = 5" before isolation
    let mut simplified_eq = eq.clone();

    // Simplify LHS if it contains the variable
    if contains_var(&simplifier.context, eq.lhs, var) {
        // SolveSafety: use prepass to avoid conditional rules corrupting solution set
        let sim_lhs = simplifier.simplify_for_solve(eq.lhs);
        simplified_eq.lhs = sim_lhs;

        // After simplification, try to recompose a^x/b^x -> (a/b)^x
        // This fixes cases where (a/b)^x was expanded to a^x/b^x during simplify,
        // which would leave 'x' on both sides after isolation attempts.
        if let Some(recomposed) =
            isolation::try_recompose_pow_quotient(&mut simplifier.context, sim_lhs)
        {
            simplified_eq.lhs = recomposed;
        }
    }

    // Simplify RHS if it contains the variable
    if contains_var(&simplifier.context, eq.rhs, var) {
        // SolveSafety: use prepass to avoid conditional rules corrupting solution set
        let sim_rhs = simplifier.simplify_for_solve(eq.rhs);
        simplified_eq.rhs = sim_rhs;

        // Also try recomposition on RHS
        if let Some(recomposed) =
            isolation::try_recompose_pow_quotient(&mut simplifier.context, sim_rhs)
        {
            simplified_eq.rhs = recomposed;
        }
    }

    // CRITICAL: After simplification, check for identities and contradictions
    // Do this by moving everything to one side: LHS - RHS
    let difference = simplifier
        .context
        .add(cas_ast::Expr::Sub(simplified_eq.lhs, simplified_eq.rhs));
    // SolveSafety: use prepass for identity/contradiction check
    let diff_simplified = simplifier.simplify_for_solve(difference);

    // Check if the difference has NO variable
    if !contains_var(&simplifier.context, diff_simplified, var) {
        // Variable disappeared - this is either an identity, contradiction, or parameter-dependent
        use cas_ast::Expr;
        match simplifier.context.get(diff_simplified) {
            Expr::Number(n) => {
                use num_traits::Zero;
                if n.is_zero() {
                    // 0 = 0: Identity, all real numbers
                    return Ok((SolutionSet::AllReals, vec![]));
                } else {
                    // c = 0 where c ≠ 0: Contradiction, no solution
                    return Ok((SolutionSet::Empty, vec![]));
                }
            }
            _ => {
                // Variable was eliminated during simplification (e.g., x/x = 1)
                // The equation is now a constraint on OTHER variables.
                // Example: (x*y)/x = 0 simplifies to y = 0
                // Solution: x can be any value (AllReals) when the constraint holds,
                // EXCEPT values that make denominators zero.
                let steps = if simplifier.collect_steps() {
                    vec![SolveStep {
                        description: format!(
                            "Variable '{}' canceled during simplification. Solution depends on constraint: {} = 0",
                            var,
                            cas_ast::DisplayExpr { context: &simplifier.context, id: diff_simplified }
                        ),
                        equation_after: Equation {
                            lhs: diff_simplified,
                            rhs: simplifier.context.add(Expr::Number(num_rational::BigRational::from_integer(0.into()))),
                            op: cas_ast::RelOp::Eq,
                        },
                        importance: crate::step::ImportanceLevel::Medium, substeps: vec![],
                    }]
                } else {
                    vec![]
                };

                // V2.1 Issue #10: Apply domain guards if any denominators contained the variable
                return wrap_with_domain_guards(
                    Ok((SolutionSet::AllReals, steps)),
                    &domain_exclusions,
                    simplifier,
                );
            }
        }
    }

    // 3. Define strategies
    // In a real app, these might be configured in Simplifier or passed in.
    let strategies: Vec<Box<dyn SolverStrategy>> = vec![
        Box::new(RationalExponentStrategy), // Must run BEFORE UnwrapStrategy to avoid loops
        Box::new(SubstitutionStrategy),
        Box::new(UnwrapStrategy),
        Box::new(QuadraticStrategy),
        Box::new(CollectTermsStrategy), // Must run before IsolationStrategy
        Box::new(IsolationStrategy),
    ];

    // 4. Try strategies on the simplified equation
    for strategy in strategies {
        if let Some(res) = strategy.apply(&simplified_eq, var, simplifier, &opts) {
            match res {
                Ok((result, steps)) => {
                    // Verify solutions if Discrete
                    if let SolutionSet::Discrete(sols) = result {
                        if !strategy.should_verify() {
                            return Ok((SolutionSet::Discrete(sols), steps));
                        }
                        let mut valid_sols = Vec::new();
                        for sol in sols {
                            // Skip verification for symbolic solutions (containing functions/variables)
                            // These can't be verified by substitution. Examples: ln(c/d)/ln(a/b)
                            // Only verify pure numeric solutions to catch extraneous roots.
                            if is_symbolic_expr(&simplifier.context, sol) {
                                // Trust symbolic solutions - can't verify
                                valid_sols.push(sol);
                            } else {
                                // CRITICAL: Verify against ORIGINAL equation, not simplified
                                // This ensures we reject solutions that cause division by zero
                                // in the original equation, even if they work in the simplified form
                                if verify_solution(eq, var, sol, simplifier) {
                                    valid_sols.push(sol);
                                }
                            }
                        }
                        return Ok((SolutionSet::Discrete(valid_sols), steps));
                    }
                    return Ok((result, steps));
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
    }

    Err(CasError::SolverError(
        "No strategy could solve this equation.".to_string(),
    ))
}

/// Try to solve equations with rational exponents like x^(3/2) = 8
/// by converting to x^3 = 64 (raising both sides to power q)
fn try_solve_rational_exponent(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    use cas_ast::Expr;
    use strategies::match_rational_power;

    // Check if RHS contains the variable (we only handle simple cases)
    if contains_var(&simplifier.context, eq.rhs, var) {
        return None;
    }

    // Try to match x^(p/q) on LHS
    let (base, p, q) = match_rational_power(&simplifier.context, eq.lhs, var)?;

    let mut steps = Vec::new();

    // Build new equation: base^p = rhs^q
    let p_expr = simplifier.context.num(p);
    let q_expr = simplifier.context.num(q);

    let new_lhs = simplifier.context.add(Expr::Pow(base, p_expr));
    let new_rhs = simplifier.context.add(Expr::Pow(eq.rhs, q_expr));

    // Simplify RHS (no variable, safe to simplify)
    let (sim_rhs, _) = simplifier.simplify(new_rhs);

    // DON'T simplify LHS yet - it might contain x^p which we want to solve

    let new_eq = Equation {
        lhs: new_lhs,
        rhs: sim_rhs,
        op: cas_ast::RelOp::Eq,
    };

    if simplifier.collect_steps() {
        steps.push(SolveStep {
            description: format!(
                "Raise both sides to power {} to eliminate rational exponent",
                q
            ),
            equation_after: new_eq.clone(),
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // Recursively solve (this will go through the full solve pipeline)
    match solve(&new_eq, var, simplifier) {
        Ok((set, mut sub_steps)) => {
            steps.append(&mut sub_steps);

            // Verify solutions against original equation (handles extraneous roots)
            if let SolutionSet::Discrete(sols) = set {
                let mut valid_sols = Vec::new();
                for sol in sols {
                    if verify_solution(eq, var, sol, simplifier) {
                        valid_sols.push(sol);
                    }
                }
                Some(Ok((SolutionSet::Discrete(valid_sols), steps)))
            } else {
                Some(Ok((set, steps)))
            }
        }
        Err(e) => Some(Err(e)),
    }
}

fn verify_solution(eq: &Equation, var: &str, sol: ExprId, simplifier: &mut Simplifier) -> bool {
    // 1. Substitute
    let lhs_sub = substitute(&mut simplifier.context, eq.lhs, var, sol);
    let rhs_sub = substitute(&mut simplifier.context, eq.rhs, var, sol);

    // 2. Simplify
    let (lhs_sim, _) = simplifier.simplify(lhs_sub);
    let (rhs_sim, _) = simplifier.simplify(rhs_sub);

    // 3. Check equality
    simplifier.are_equivalent(lhs_sim, rhs_sim)
}

/// Check if an expression is "symbolic" (contains functions or variables).
/// Symbolic expressions cannot be verified by substitution because they don't
/// simplify to pure numbers. Examples: ln(c/d)/ln(a/b), x + a, sqrt(y)
fn is_symbolic_expr(ctx: &Context, expr: ExprId) -> bool {
    use cas_ast::Expr;
    match ctx.get(expr) {
        Expr::Number(_) => false,
        Expr::Constant(_) => true, // Pi, E, etc are symbolic
        Expr::Variable(_) => true,
        Expr::Function(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            is_symbolic_expr(ctx, *l) || is_symbolic_expr(ctx, *r)
        }
        Expr::Neg(e) => is_symbolic_expr(ctx, *e),
        _ => true, // Default: treat as symbolic
    }
}

fn substitute(ctx: &mut Context, expr: ExprId, var: &str, val: ExprId) -> ExprId {
    use cas_ast::Expr;
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => val,
        Expr::Add(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Sub(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                mul2_raw(ctx, nl, nr)
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Pow(b, e) => {
            let nb = substitute(ctx, b, var, val);
            let ne = substitute(ctx, e, var, val);
            if nb != b || ne != e {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }
        Expr::Neg(e) => {
            let ne = substitute(ctx, e, var, val);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for arg in args {
                let new_arg = substitute(ctx, arg, var, val);
                if new_arg != arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }
        _ => expr,
    }
}

/// V2.1 Issue #10: Extract all denominators from an expression that contain the given variable.
///
/// Used to detect domain restrictions when solving equations with fractions.
/// Returns a list of ExprIds that appear as denominators and contain the variable.
///
/// Example: `(x*y)/x` returns `[x]` (the denominator x contains var "x")
fn extract_denominators_with_var(ctx: &Context, expr: ExprId, var: &str) -> Vec<ExprId> {
    use std::collections::HashSet;
    let mut denoms_set: HashSet<ExprId> = HashSet::new();
    collect_denominators_into_set(ctx, expr, var, &mut denoms_set);
    denoms_set.into_iter().collect()
}

/// Helper to recursively collect denominators into a HashSet
fn collect_denominators_into_set(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    denoms: &mut std::collections::HashSet<ExprId>,
) {
    use cas_ast::Expr;
    match ctx.get(expr) {
        Expr::Div(num, denom) => {
            // Check if denominator contains the variable
            if contains_var(ctx, *denom, var) {
                denoms.insert(*denom);
            }
            // Also check for nested divisions in numerator and denominator
            collect_denominators_into_set(ctx, *num, var, denoms);
            collect_denominators_into_set(ctx, *denom, var, denoms);
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
            collect_denominators_into_set(ctx, *l, var, denoms);
            collect_denominators_into_set(ctx, *r, var, denoms);
        }
        Expr::Neg(e) => {
            collect_denominators_into_set(ctx, *e, var, denoms);
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_denominators_into_set(ctx, *arg, var, denoms);
            }
        }
        _ => {}
    }
}

/// V2.1 Issue #10: Wrap a solve result with domain guards for denominators.
///
/// If there are domain exclusions (denominators that must be non-zero),
/// this wraps the result in a Conditional with NonZero guards.
fn wrap_with_domain_guards(
    result: Result<(SolutionSet, Vec<SolveStep>), CasError>,
    exclusions: &[ExprId],
    _simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // If no exclusions, return as-is
    if exclusions.is_empty() {
        return result;
    }

    let (solution_set, steps) = result?;

    // Build the NonZero guard condition set
    let mut guard = cas_ast::ConditionSet::empty();
    for &denom in exclusions {
        guard.push(cas_ast::ConditionPredicate::NonZero(denom));
    }

    // Wrap in Conditional: [guard -> solution, otherwise -> Empty (undefined)]
    let cases = vec![
        cas_ast::Case::new(guard, solution_set),
        cas_ast::Case::new(cas_ast::ConditionSet::empty(), SolutionSet::Empty),
    ];

    Ok((SolutionSet::Conditional(cases).simplify(), steps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{BoundType, Context, DisplayExpr, RelOp};
    use cas_parser::parse;

    // Helper to make equation from strings
    fn make_eq(ctx: &mut Context, lhs: &str, rhs: &str) -> Equation {
        Equation {
            lhs: parse(lhs, ctx).unwrap(),
            rhs: parse(rhs, ctx).unwrap(),
            op: RelOp::Eq,
        }
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
}
