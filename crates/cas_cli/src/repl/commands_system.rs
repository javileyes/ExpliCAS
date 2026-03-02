//! Linear system solving commands.
//!
//! Syntax: solve_system(eq1; eq2; x; y)
//! Example: solve_system(x+y=3; x-y=1; x; y) → { x = 2, y = 1 }

use super::output::{reply_output, ReplReply};
use super::Repl;

impl Repl {
    /// Core: handle_solve_system_core (returns ReplReply, no I/O)
    pub(crate) fn handle_solve_system_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_linear_system_command_line(
            &mut self.core.engine.simplifier.context,
            line,
        ) {
            Ok(message) => reply_output(message),
            Err(message) => reply_output(message),
        }
    }
}
