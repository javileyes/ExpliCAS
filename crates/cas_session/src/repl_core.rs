//! REPL core state owned by session/application layer.
//!
//! `cas_cli` uses this type as a thin UI shell over session + solver orchestration.

mod accessors;
mod constructors;
mod runtime_state;

use crate::SessionState;
use cas_solver::{Engine, PipelineStats, SimplifyOptions};

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
