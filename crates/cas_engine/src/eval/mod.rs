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
    Vec<crate::api::SolveStep>,
    Vec<crate::AssumptionRecord>,
    Vec<cas_formatter::display_transforms::ScopeTag>,
    Vec<crate::ImplicitCondition>,
);

use crate::Simplifier;
use cas_ast::{Context, ExprId};
use cas_math::prove_nonzero::prove_nonzero_depth_with;
use cas_math::prove_sign::{prove_nonnegative_depth_with, prove_positive_depth_with};
use cas_math::tri_proof::TriProof;

const WARNING_SIGN_PROOF_DEPTH: usize = 12;

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
        Self::with_simplifier(Simplifier::from_profile_with_context(
            crate::profile_cache::default_rule_profile(),
            cas_ast::Context::new(),
        ))
    }

    /// Create an Engine with a pre-populated Context (for session restoration).
    pub fn with_context(context: cas_ast::Context) -> Self {
        Self::with_simplifier(Simplifier::from_profile_with_context(
            crate::profile_cache::default_rule_profile(),
            context,
        ))
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

pub use cas_solver_core::eval_models::{EvalAction, EvalRequest, EvalResult};
pub use cas_solver_core::eval_output_model::EvalOutput;

pub type DomainWarning = cas_solver_core::domain_warning::DomainWarning;

/// Collect domain warnings from steps with deduplication.
/// Collects structured assumption_events from each step.
/// Note: Only events that are NOT RequiresIntroduced become DomainWarnings (⚠️).
/// RequiresIntroduced events are displayed in steps with ℹ️ icon instead.
fn warning_event_is_intrinsically_true_in_reals(
    ctx: &Context,
    event: &crate::AssumptionEvent,
) -> bool {
    let Some(expr) = event.expr_id else {
        return false;
    };

    match event.key {
        crate::AssumptionKey::Positive { .. } => prove_positive_depth_with(
            ctx,
            expr,
            WARNING_SIGN_PROOF_DEPTH,
            true,
            |_ctx, _expr, _depth| TriProof::Unknown,
        )
        .is_proven(),
        crate::AssumptionKey::NonNegative { .. } => prove_nonnegative_depth_with(
            ctx,
            expr,
            WARNING_SIGN_PROOF_DEPTH,
            true,
            |_ctx, _expr, _depth| TriProof::Unknown,
        )
        .is_proven(),
        crate::AssumptionKey::NonZero { .. } => prove_nonzero_depth_with(
            ctx,
            expr,
            WARNING_SIGN_PROOF_DEPTH,
            |_ctx, _expr| TriProof::Unknown,
            |_ctx, _expr| None,
        )
        .is_proven(),
        _ => false,
    }
}

fn required_condition_matches_warning_event(
    ctx: &Context,
    required: &crate::ImplicitCondition,
    event: &crate::AssumptionEvent,
) -> bool {
    match (required, &event.key) {
        (
            crate::ImplicitCondition::Positive(expr),
            crate::AssumptionKey::Positive { expr_fingerprint },
        )
        | (
            crate::ImplicitCondition::NonNegative(expr),
            crate::AssumptionKey::NonNegative { expr_fingerprint },
        )
        | (
            crate::ImplicitCondition::NonZero(expr),
            crate::AssumptionKey::NonZero { expr_fingerprint },
        ) => crate::expr_fingerprint(ctx, *expr) == *expr_fingerprint,
        _ => false,
    }
}

fn warning_event_is_redundant_with_required_condition(
    ctx: &Context,
    step: &crate::Step,
    event: &crate::AssumptionEvent,
) -> bool {
    matches!(event.kind, crate::AssumptionKind::HeuristicAssumption)
        && step
            .required_conditions()
            .iter()
            .any(|required| required_condition_matches_warning_event(ctx, required, event))
}

pub(crate) fn collect_domain_warnings(
    ctx: &Context,
    value_domain: crate::semantics::ValueDomain,
    result: ExprId,
    steps: &[crate::Step],
) -> Vec<DomainWarning> {
    use std::collections::HashSet;

    let real_only = value_domain == crate::semantics::ValueDomain::RealOnly;
    let result_domain = crate::infer_implicit_domain(ctx, result, value_domain);
    let mut seen: HashSet<String> = HashSet::new();
    let mut warnings = Vec::new();

    for step in steps {
        for event in step.assumption_events() {
            let should_skip = matches!(
                event.kind,
                crate::AssumptionKind::RequiresIntroduced
                    | crate::AssumptionKind::DerivedFromRequires
            ) || (real_only
                && warning_event_is_intrinsically_true_in_reals(ctx, event))
                || warning_event_is_redundant_with_required_condition(ctx, step, event);
            let result_domain_already_requires_warning =
                matches!(event.kind, crate::AssumptionKind::HeuristicAssumption)
                    && result_domain.conditions().iter().any(|required| {
                        required_condition_matches_warning_event(ctx, required, event)
                    });
            if should_skip || result_domain_already_requires_warning {
                continue;
            }

            let message = event.message.clone();
            if seen.insert(message.clone()) {
                warnings.push(DomainWarning {
                    message,
                    rule_name: step.rule_name.to_string(),
                });
            }
        }
    }

    warnings
}
