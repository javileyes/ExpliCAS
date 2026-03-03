//! REPL core state owned by session/application layer.
//!
//! `cas_cli` uses this type as a thin UI shell over session + solver orchestration.

use crate::SessionState;
use cas_solver::{Engine, EvalOptions, PipelineStats, Simplifier, SimplifyOptions};

/// Core REPL state without terminal I/O concerns.
pub struct ReplCore {
    /// The high-level Engine instance (wraps Simplifier)
    engine: Engine,
    /// Options controlling the simplification pipeline (phases, budgets)
    simplify_options: SimplifyOptions,
    /// When true, show pipeline/engine diagnostics after simplification
    debug_mode: bool,
    /// Last pipeline stats for diagnostics
    last_stats: Option<PipelineStats>,
    /// When true, always track health metrics (independent of debug)
    health_enabled: bool,
    /// Last health report string for `health` command
    last_health_report: Option<String>,
    /// Session state (store + env)
    state: SessionState,
}

impl ReplCore {
    /// Create a new ReplCore with a pre-configured simplifier.
    pub fn with_simplifier(simplifier: Simplifier) -> Self {
        Self {
            engine: Engine::with_simplifier(simplifier),
            simplify_options: SimplifyOptions::default(),
            debug_mode: false,
            last_stats: None,
            health_enabled: false,
            last_health_report: None,
            state: SessionState::new(),
        }
    }

    /// Create with default rules (for testing or simple use).
    pub fn new() -> Self {
        let simplifier = Simplifier::with_default_rules();
        Self::with_simplifier(simplifier)
    }

    /// Run a closure with mutable access to engine + session state.
    pub fn with_engine_and_state<R>(
        &mut self,
        f: impl FnOnce(&mut Engine, &mut SessionState) -> R,
    ) -> R {
        f(&mut self.engine, &mut self.state)
    }

    /// Run a closure with mutable access to the inner simplifier.
    pub fn with_simplifier_mut<R>(&mut self, f: impl FnOnce(&mut Simplifier) -> R) -> R {
        f(&mut self.engine.simplifier)
    }

    /// Borrow session state + simplifier mutably in one operation.
    pub(crate) fn with_state_and_simplifier_mut<R>(
        &mut self,
        f: impl FnOnce(&mut SessionState, &mut Simplifier) -> R,
    ) -> R {
        f(&mut self.state, &mut self.engine.simplifier)
    }

    /// Borrow the current solver engine mutably.
    pub(crate) fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Borrow the session state.
    pub(crate) fn state(&self) -> &SessionState {
        &self.state
    }

    /// Borrow the session state mutably.
    pub(crate) fn state_mut(&mut self) -> &mut SessionState {
        &mut self.state
    }

    /// Borrow simplify pipeline options.
    pub(crate) fn simplify_options(&self) -> &SimplifyOptions {
        &self.simplify_options
    }

    /// Borrow simplify options + eval options mutably in one operation.
    pub(crate) fn with_simplify_and_eval_options_mut<R>(
        &mut self,
        f: impl FnOnce(&mut SimplifyOptions, &mut EvalOptions) -> R,
    ) -> R {
        f(&mut self.simplify_options, self.state.options_mut())
    }

    /// Borrow eval options from session state.
    pub(crate) fn eval_options(&self) -> &EvalOptions {
        self.state.options()
    }

    /// Borrow eval options from session state mutably.
    pub(crate) fn eval_options_mut(&mut self) -> &mut EvalOptions {
        self.state.options_mut()
    }

    /// Borrow the inner simplifier.
    pub(crate) fn simplifier(&self) -> &Simplifier {
        &self.engine.simplifier
    }

    /// Borrow the inner simplifier mutably.
    pub(crate) fn simplifier_mut(&mut self) -> &mut Simplifier {
        &mut self.engine.simplifier
    }

    /// Replace the inner simplifier.
    pub(crate) fn set_simplifier(&mut self, simplifier: Simplifier) {
        self.engine.simplifier = simplifier;
    }

    /// Rebuild simplifier from current eval profile.
    pub(crate) fn rebuild_simplifier_from_profile(&mut self) {
        self.engine.simplifier = Simplifier::with_profile(self.state.options());
    }

    /// Clear session state history/bindings.
    pub(crate) fn clear_state(&mut self) {
        self.state.clear();
    }

    /// Return whether debug mode is enabled.
    pub(crate) fn debug_mode(&self) -> bool {
        self.debug_mode
    }

    /// Set debug mode flag.
    pub(crate) fn set_debug_mode(&mut self, value: bool) {
        self.debug_mode = value;
    }

    /// Return whether health tracking is enabled.
    pub(crate) fn health_enabled(&self) -> bool {
        self.health_enabled
    }

    /// Set health tracking flag.
    pub(crate) fn set_health_enabled(&mut self, value: bool) {
        self.health_enabled = value;
    }

    /// Borrow latest pipeline stats.
    pub(crate) fn last_stats(&self) -> Option<&PipelineStats> {
        self.last_stats.as_ref()
    }

    /// Replace latest pipeline stats.
    pub(crate) fn set_last_stats(&mut self, value: Option<PipelineStats>) {
        self.last_stats = value;
    }

    /// Borrow latest health report text.
    pub(crate) fn last_health_report(&self) -> Option<&str> {
        self.last_health_report.as_deref()
    }

    /// Replace latest health report text.
    pub(crate) fn set_last_health_report(&mut self, value: Option<String>) {
        self.last_health_report = value;
    }

    /// Clear latest health report text.
    pub(crate) fn clear_last_health_report(&mut self) {
        self.last_health_report = None;
    }

    /// Clear engine profile cache.
    pub(crate) fn clear_profile_cache(&mut self) {
        self.engine.clear_profile_cache();
    }

    /// Number of cached engine profiles.
    pub(crate) fn profile_cache_len(&self) -> usize {
        self.engine.profile_cache_len()
    }
}

impl Default for ReplCore {
    fn default() -> Self {
        Self::new()
    }
}
