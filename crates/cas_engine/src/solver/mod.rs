pub mod check;
pub(crate) mod domain_guards;
pub(crate) mod isolation;
pub(crate) mod linear_collect;
pub(crate) mod log_linear_narrator;
pub(crate) mod numeric_islands;
pub(crate) mod proof_bridge;
pub(crate) mod quadratic_steps;
pub(crate) mod reciprocal_solve;
pub(crate) mod runtime_tls;
pub(crate) mod solve_core;
pub(crate) mod step_cleanup;
pub(crate) mod strategies;
pub(crate) mod strategy;
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::verify_stats;
pub(crate) use runtime_tls::{emit_scope, note_assumption, take_scopes, SolveAssumptionsGuard};

#[cfg(test)]
use crate::engine::Simplifier;
use cas_ast::Equation;
#[cfg(test)]
use cas_ast::SolutionSet;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

pub use self::solve_core::{solve, solve_with_display_steps};

/// Solver context — threaded explicitly through the solve pipeline.
///
/// Holds per-invocation state that was formerly stored in TLS,
/// enabling clean reentrancy for recursive/nested solves.
///
/// The `required_sink` is a shared accumulator (`Rc<RefCell<…>>`)
/// so that recursive sub-solves contribute conditions to the same set.
/// `solve_with_display_steps` creates one, and every recursive
/// `solve_with_ctx` / `solve_with_options` pushes into it.
#[derive(Debug, Clone)]
pub struct SolveCtx {
    /// Domain environment inferred from equation structure (per-level).
    pub domain_env: SolveDomainEnv,
    /// Shared accumulator for required conditions across all recursive levels.
    pub(crate) required_sink: Rc<RefCell<HashSet<crate::implicit_domain::ImplicitCondition>>>,
}

impl Default for SolveCtx {
    fn default() -> Self {
        Self {
            domain_env: SolveDomainEnv::default(),
            required_sink: Rc::new(RefCell::new(HashSet::new())),
        }
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

// ---------------------------------------------------------------------------
// Recursion depth limit
// ---------------------------------------------------------------------------

/// Maximum recursion depth for solver to prevent stack overflow
pub(crate) const MAX_SOLVE_DEPTH: usize = 50;

// ---------------------------------------------------------------------------
// Legacy thread-local state: diagnostic/UI ONLY
// ---------------------------------------------------------------------------
// These TLS cells are retained for recursion safety and display concerns.
// They must NOT carry solver-semantic information (e.g. domain conditions).
// Domain conditions are fully SolveCtx-threaded and returned in SolveDiagnostics.
//
// Allowlisted cells:
//   SOLVE_DEPTH        – recursion depth guard (prevents stack overflow)
//   SOLVE_ASSUMPTIONS  – per-solve assumption collector (runtime_tls.rs)
//   OUTPUT_SCOPES      – display scope tags emitted by strategies (runtime_tls.rs)
thread_local! {
    pub(crate) static SOLVE_DEPTH: std::cell::RefCell<usize> = const { std::cell::RefCell::new(0) };
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

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{BoundType, Context, RelOp};
    use cas_formatter::DisplayExpr;
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
