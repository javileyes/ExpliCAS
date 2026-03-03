use super::output::ReplReply;
use super::*;

impl Repl {
    /// Main command dispatch - calls core and prints result.
    ///
    /// # Panic Fence (Defense in Depth)
    /// Wraps command execution with `catch_unwind` to prevent internal panics
    /// from crashing the REPL session. On panic:
    /// - Generates a unique error_id for correlation
    /// - Displays a user-friendly error message with error_id
    /// - Logs details to stderr (controlled by EXPLICAS_PANIC_REPORT)
    /// - Session remains alive for continued use
    pub fn handle_command(&mut self, line: &str) {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        // Wrap in catch_unwind to prevent panics from killing the session
        // AssertUnwindSafe is needed because &mut self is not UnwindSafe
        let result = catch_unwind(AssertUnwindSafe(|| self.handle_command_core(line)));

        match result {
            Ok(reply) => self.print_reply(reply),
            Err(panic_info) => {
                let panic_msg = super::panic_guard::panic_payload_to_message(panic_info.as_ref());

                // Generate short error_id for correlation (hash of timestamp + message)
                let error_id = super::panic_guard::generate_short_error_id(&panic_msg);

                // Log to stderr if EXPLICAS_PANIC_REPORT is set (for debugging)
                // This respects the repl IO lint by only logging when explicitly enabled
                if std::env::var("EXPLICAS_PANIC_REPORT").is_ok() {
                    // Use init.rs print_reply with Error variant which goes to stderr
                    let version = env!("CARGO_PKG_VERSION");
                    let log_msg = super::panic_guard::format_panic_report_message(
                        &error_id, version, line, &panic_msg,
                    );
                    // Log via ReplMsg::Debug to get it to output
                    self.print_reply(vec![ReplMsg::Debug(log_msg)]);
                }

                // Return user-friendly error that doesn't kill the session
                self.print_reply(vec![ReplMsg::Error(
                    super::panic_guard::format_user_panic_message(&error_id, &panic_msg),
                )]);
            }
        }
    }

    /// Core command dispatch - returns structured messages, no I/O.
    /// This is the heart of ReplCore logic.
    pub fn handle_command_core(&mut self, line: &str) -> ReplReply {
        // Preprocess: Convert function-style commands to command-style
        // simplify(...) -> simplify ...
        // solve(...) -> solve ...
        let line = self.preprocess_function_syntax(line);

        match cas_session::parse_repl_command_input(&line) {
            cas_session::ReplCommandInput::Help(line) => self.handle_help_core(line),
            cas_session::ReplCommandInput::Let(rest) => self.handle_let_command_core(rest),
            cas_session::ReplCommandInput::Assignment { name, expr, lazy } => {
                self.handle_assignment_core(name, expr, lazy)
            }
            cas_session::ReplCommandInput::Vars => self.handle_vars_command_core(),
            cas_session::ReplCommandInput::Clear(line) => self.handle_clear_command_core(line),
            cas_session::ReplCommandInput::Reset => self.handle_reset_command_core(),
            cas_session::ReplCommandInput::ResetFull => self.handle_reset_full_command_core(),
            cas_session::ReplCommandInput::Cache(line) => self.handle_cache_command_core(line),
            cas_session::ReplCommandInput::Semantics(line) => self.handle_semantics_core(line),
            cas_session::ReplCommandInput::Context(line) => self.handle_context_command_core(line),
            cas_session::ReplCommandInput::Steps(line) => self.handle_steps_command_core(line),
            cas_session::ReplCommandInput::Autoexpand(line) => {
                self.handle_autoexpand_command_core(line)
            }
            cas_session::ReplCommandInput::Budget(line) => self.handle_budget_command_core(line),
            cas_session::ReplCommandInput::History => self.handle_history_command_core(),
            cas_session::ReplCommandInput::Show(rest) => self.handle_show_command_core(rest),
            cas_session::ReplCommandInput::Del(rest) => self.handle_del_command_core(rest),
            cas_session::ReplCommandInput::Set(line) => {
                let result = self.handle_set_command_core(line);
                self.finalize_core_result(result)
            }
            cas_session::ReplCommandInput::Equiv(line) => self.handle_equiv_core(line),
            cas_session::ReplCommandInput::Subst(line) => {
                self.handle_subst_core(line, self.verbosity)
            }
            cas_session::ReplCommandInput::SolveSystem(line) => self.handle_solve_system_core(line),
            cas_session::ReplCommandInput::Solve(line) => {
                self.handle_solve_core(line, self.verbosity)
            }
            cas_session::ReplCommandInput::Simplify(line) => {
                self.handle_full_simplify_core(line, self.verbosity)
            }
            cas_session::ReplCommandInput::Config(line) => self.handle_config_core(line),
            cas_session::ReplCommandInput::Timeline(line) => self.handle_timeline_core(line),
            cas_session::ReplCommandInput::Visualize(line) => self.handle_visualize_core(line),
            cas_session::ReplCommandInput::Explain(line) => self.handle_explain_core(line),
            cas_session::ReplCommandInput::Det(line) => self.handle_det_core(line, self.verbosity),
            cas_session::ReplCommandInput::Transpose(line) => {
                self.handle_transpose_core(line, self.verbosity)
            }
            cas_session::ReplCommandInput::Trace(line) => {
                self.handle_trace_core(line, self.verbosity)
            }
            cas_session::ReplCommandInput::Telescope(line) => self.handle_telescope_core(line),
            cas_session::ReplCommandInput::Weierstrass(line) => self.handle_weierstrass_core(line),
            cas_session::ReplCommandInput::ExpandLog(line) => self.handle_expand_log_core(line),
            cas_session::ReplCommandInput::Expand(line) => self.handle_expand_core(line),
            cas_session::ReplCommandInput::Rationalize(line) => self.handle_rationalize_core(line),
            cas_session::ReplCommandInput::Limit(line) => self.handle_limit_core(line),
            cas_session::ReplCommandInput::Profile(line) => self.handle_profile_command_core(line),
            cas_session::ReplCommandInput::Health(line) => self.handle_health_core(line),
            cas_session::ReplCommandInput::Eval(line) => self.handle_eval_core(line),
        }
    }
}
