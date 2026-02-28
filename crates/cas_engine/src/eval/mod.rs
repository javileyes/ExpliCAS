//! Evaluation engine: `Engine.eval()` entry point and supporting types.
//!
//! This module is the top-level orchestrator that takes a parsed expression,
//! resolves session references, dispatches to the appropriate action handler
//! (simplify, solve, expand, equiv, limit), assembles diagnostics, and
//! returns the final `EvalOutput`.

mod actions;
mod diagnostics;
mod dispatch;
mod simplify_action;

/// Result type for individual action handlers in `Engine.eval()`.
///
/// Fields: (result, domain_warnings, steps, solve_steps, solver_assumptions, output_scopes, solver_required)
pub(crate) type ActionResult = (
    EvalResult,
    Vec<DomainWarning>,
    Vec<crate::Step>,
    Vec<crate::solver::SolveStep>,
    Vec<crate::assumptions::AssumptionRecord>,
    Vec<cas_formatter::display_transforms::ScopeTag>,
    Vec<crate::implicit_domain::ImplicitCondition>,
);

use crate::Simplifier;
use cas_ast::{Expr, ExprId};

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
/// use cas_formatter::DisplayExpr;
/// let output = format!("{}", DisplayExpr { context: &engine.simplifier.context, id: result });
/// assert!(output.contains("x")); // Contains x
/// ```
pub struct Engine {
    pub simplifier: Simplifier,
    profile_cache: crate::profile_cache::ProfileCache,
}

/// Session contracts are shared from `cas_session_core` so session state
/// can evolve outside `cas_engine` while preserving the eval API surface.
pub use cas_session_core::eval::{EvalSession, EvalStore};

impl Engine {
    /// Create an Engine from a configured `Simplifier`.
    pub fn with_simplifier(simplifier: Simplifier) -> Self {
        Self {
            simplifier,
            profile_cache: crate::profile_cache::ProfileCache::new(),
        }
    }

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
        Self::with_simplifier(Simplifier::with_default_rules())
    }

    /// Create an Engine with a pre-populated Context (for session restoration).
    pub fn with_context(context: cas_ast::Context) -> Self {
        Self::with_simplifier(Simplifier::with_context(context))
    }

    /// Number of cached rule profiles currently held by this engine.
    pub fn profile_cache_len(&self) -> usize {
        self.profile_cache.len()
    }

    /// Clear cached rule profiles.
    pub fn clear_profile_cache(&mut self) {
        self.profile_cache.clear();
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
        if opts.shared.context_mode == ContextMode::Auto {
            if cas_math::numeric_eval::contains_integral(&self.simplifier.context, expr) {
                effective.shared.context_mode = ContextMode::IntegratePrep;
            } else {
                effective.shared.context_mode = ContextMode::Standard;
            }
        }

        // Resolve ComplexMode::Auto
        if opts.complex_mode == ComplexMode::Auto {
            if cas_math::numeric_eval::contains_i(&self.simplifier.context, expr) {
                effective.complex_mode = ComplexMode::On;
            } else {
                effective.complex_mode = ComplexMode::Off;
            }
        }

        // CRITICAL: Force expand_policy Off in Solve mode
        // Auto-expansion can interfere with equation solving strategies
        if effective.shared.context_mode == ContextMode::Solve {
            effective.shared.expand_policy = ExpandPolicy::Off;
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
    Solve {
        var: String,
    },
    // Check equivalence between two expressions
    Equiv {
        other: ExprId,
    },
    // Compute limit as variable approaches a value
    Limit {
        var: String,
        approach: crate::limits::Approach,
    },
}

#[derive(Clone, Debug)]
pub struct EvalRequest {
    pub raw_input: String,
    pub parsed: ExprId,
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
    pub output_scopes: Vec<cas_formatter::display_transforms::ScopeTag>,
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
pub(crate) fn collect_domain_warnings(steps: &[crate::Step]) -> Vec<DomainWarning> {
    cas_session_core::eval::collect_warnings_with(
        steps,
        |step| step.assumption_events().to_vec(),
        |event| {
            matches!(
                event.kind,
                crate::assumptions::AssumptionKind::RequiresIntroduced
                    | crate::assumptions::AssumptionKind::DerivedFromRequires
            )
        },
        |event| event.message.clone(),
        |step| step.rule_name.clone(),
        |message, rule_name| DomainWarning { message, rule_name },
    )
}
