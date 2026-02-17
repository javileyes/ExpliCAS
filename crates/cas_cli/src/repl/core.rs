//! Core REPL logic without I/O.
//!
//! ReplCore contains the pure computational logic of the REPL,
//! returning structured messages instead of printing directly.

use cas_session::SessionState;
use cas_solver::{Engine, PipelineStats, Simplifier, SimplifyOptions};

/// Core REPL logic - pure computation without I/O.
///
/// This struct handles the computational aspects of the REPL,
/// returning `ReplReply` messages instead of printing directly.
/// The outer `Repl` handles actual I/O.
pub struct ReplCore {
    /// The high-level Engine instance (wraps Simplifier)
    pub engine: Engine,
    /// Options controlling the simplification pipeline (phases, budgets)
    pub simplify_options: SimplifyOptions,
    /// When true, show pipeline/engine diagnostics after simplification
    pub debug_mode: bool,
    /// Last pipeline stats for diagnostics
    pub last_stats: Option<PipelineStats>,
    /// When true, always track health metrics (independent of debug)
    pub health_enabled: bool,
    /// Last health report string for `health` command
    pub last_health_report: Option<String>,
    /// Session state (store + env)
    pub state: SessionState,
}

impl ReplCore {
    /// Create a new ReplCore with a pre-configured simplifier.
    ///
    /// Note: The simplifier should be configured by the caller (Repl::new)
    /// since rule configuration depends on CasConfig which is UI-level.
    pub fn with_simplifier(simplifier: Simplifier) -> Self {
        Self {
            engine: Engine { simplifier },
            simplify_options: SimplifyOptions::default(),
            debug_mode: false,
            last_stats: None,
            health_enabled: false,
            last_health_report: None,
            state: SessionState::new(),
        }
    }

    /// Create with default rules (for testing or simple use)
    pub fn new() -> Self {
        let simplifier = Simplifier::with_default_rules();
        Self::with_simplifier(simplifier)
    }

    /// Get verbosity-aware formatted message
    /// (verbosity is passed in since it lives in Repl, not ReplCore)
    pub fn format_output(
        &self,
        expr: cas_ast::ExprId,
        style_prefs: &cas_ast::StylePreferences,
    ) -> String {
        use cas_formatter::DisplayExprStyled;
        format!(
            "{}",
            DisplayExprStyled::new(&self.engine.simplifier.context, expr, style_prefs)
        )
    }
}

impl Default for ReplCore {
    fn default() -> Self {
        Self::new()
    }
}
