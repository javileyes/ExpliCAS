//! Linear system solving commands.
//!
//! Syntax: solve_system(eq1; eq2; x; y)
//! Example: solve_system(x+y=3; x-y=1; x; y) → { x = 2, y = 1 }

use super::output::{reply_output, ReplReply};
use super::Repl;

impl Repl {
    /// Wrapper: handle_solve_system (prints result)
    pub(crate) fn handle_solve_system(&mut self, line: &str) {
        let reply = self.handle_solve_system_core(line);
        self.print_reply(reply);
    }

    /// Core: handle_solve_system_core (returns ReplReply, no I/O)
    fn handle_solve_system_core(&mut self, line: &str) -> ReplReply {
        let invocation = cas_solver::parse_linear_system_invocation_input(line);

        let eval_out = {
            let ctx = &mut self.core.engine.simplifier.context;
            cas_solver::evaluate_linear_system_command_input(ctx, &invocation.spec)
        };

        match eval_out {
            Ok(out) => reply_output(cas_solver::format_linear_system_result_message(
                &mut self.core.engine.simplifier.context,
                &out,
            )),
            Err(e) => reply_output(cas_solver::format_linear_system_command_error_message(&e)),
        }
    }
}
