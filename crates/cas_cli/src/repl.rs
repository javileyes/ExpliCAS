use cas_session::CasConfig;
use rustyline::error::ReadlineError;

use crate::completer::CasHelper;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    None,
    Succinct, // Compact: same filtering as Normal but 1 line per step
    Normal,
    Verbose,
}

pub struct Repl {
    /// Core logic - pure computation without I/O
    pub core: ReplCore,
    /// Output verbosity level (UI concern)
    verbosity: Verbosity,
    /// CLI configuration (loaded from file, applies rules to core)
    config: CasConfig,
}

impl Default for Repl {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Repl implementation - split across modules for maintainability
// =============================================================================
// We intentionally keep Repl's implementation split by concern, but *without*
// include!(), so the compiler can track boundaries and tooling/navigation works
// better. Each file defines `impl Repl { ... }` for its feature area.
//
// File contents:
//   init.rs             - Constructor and configuration sync
//   dispatch.rs         - Command dispatch (routing in cas_session)
//   help.rs             - Help system and documentation
//   commands_analysis.rs - Analysis commands (equiv, subst, timeline, visualize, explain)
//   commands_health.rs  - Health diagnostics command
//   commands_session.rs  - Session environment (let, vars, clear, reset, cache)
//   commands_semantics.rs - Semantics commands (semantics, presets)
//   semantics.rs        - Semantic analysis commands
//   commands_algebra.rs - Algebra commands (factor, expand, etc.)
//   commands_solve.rs   - Solve command and equation handling
//   show_steps.rs       - Step-by-step output formatting
//   eval.rs             - Expression evaluation
//   simplify.rs         - Simplification pipeline
//   rationalize.rs      - Rationalization commands
//   limit.rs            - Limit computation
// =============================================================================

mod commands_algebra;
mod commands_analysis;
mod commands_config;
mod commands_health;
mod commands_semantics;
mod commands_session;
mod commands_solve;
mod commands_system;
mod dispatch;
mod eval;
mod general_help;
mod help;
mod help_command;
mod help_topics;
mod init;
mod limit;
pub mod output;
mod panic_guard;
mod rationalize;
mod semantics;
mod show_steps;
mod simplify;
pub mod wire;

#[cfg(test)]
mod core_tests;

// Re-export core types for external use
pub use cas_session::ReplCore;
pub use cas_solver::session_api::options::SetDisplayMode;
pub use output::{reply_output, CoreResult, ReplMsg, ReplReply, ReplReplyExt, UiDelta};
