use super::output::ReplReply;
use super::*;

impl Repl {
    /// Core command dispatch - returns structured messages, no I/O.
    pub fn handle_command_core(&mut self, line: &str) -> ReplReply {
        let line = self.preprocess_function_syntax(line);

        match cas_solver::session_api::repl::parse_repl_command_input(&line) {
            cas_solver::session_api::repl::ReplCommandInput::Help(line) => {
                self.handle_help_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Let(rest) => {
                self.handle_let_command_core(rest)
            }
            cas_solver::session_api::repl::ReplCommandInput::Assignment { name, expr, lazy } => {
                self.handle_assignment_core(name, expr, lazy)
            }
            cas_solver::session_api::repl::ReplCommandInput::Vars => {
                self.handle_vars_command_core()
            }
            cas_solver::session_api::repl::ReplCommandInput::Clear(line) => {
                self.handle_clear_command_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Reset => {
                self.handle_reset_command_core()
            }
            cas_solver::session_api::repl::ReplCommandInput::ResetFull => {
                self.handle_reset_full_command_core()
            }
            cas_solver::session_api::repl::ReplCommandInput::Cache(line) => {
                self.handle_cache_command_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Semantics(line) => {
                self.handle_semantics_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Context(line) => {
                self.handle_context_command_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Steps(line) => {
                self.handle_steps_command_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Autoexpand(line) => {
                self.handle_autoexpand_command_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Budget(line) => {
                self.handle_budget_command_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::History => {
                self.handle_history_command_core()
            }
            cas_solver::session_api::repl::ReplCommandInput::Show(rest) => {
                self.handle_show_command_core(rest)
            }
            cas_solver::session_api::repl::ReplCommandInput::Del(rest) => {
                self.handle_del_command_core(rest)
            }
            cas_solver::session_api::repl::ReplCommandInput::Set(line) => {
                let result = self.handle_set_command_core(line);
                self.finalize_core_result(result)
            }
            cas_solver::session_api::repl::ReplCommandInput::Equiv(line) => {
                self.handle_equiv_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Subst(line) => {
                self.handle_subst_core(line, self.verbosity)
            }
            cas_solver::session_api::repl::ReplCommandInput::SolveSystem(line) => {
                self.handle_solve_system_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Solve(line) => {
                self.handle_solve_core(line, self.verbosity)
            }
            cas_solver::session_api::repl::ReplCommandInput::Simplify(line) => {
                self.handle_full_simplify_core(line, self.verbosity)
            }
            cas_solver::session_api::repl::ReplCommandInput::Config(line) => {
                self.handle_config_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Timeline(line) => {
                self.handle_timeline_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Visualize(line) => {
                self.handle_visualize_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Explain(line) => {
                self.handle_explain_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Det(line) => {
                self.handle_det_core(line, self.verbosity)
            }
            cas_solver::session_api::repl::ReplCommandInput::Transpose(line) => {
                self.handle_transpose_core(line, self.verbosity)
            }
            cas_solver::session_api::repl::ReplCommandInput::Trace(line) => {
                self.handle_trace_core(line, self.verbosity)
            }
            cas_solver::session_api::repl::ReplCommandInput::Telescope(line) => {
                self.handle_telescope_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Weierstrass(line) => {
                self.handle_weierstrass_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::ExpandLog(line) => {
                self.handle_expand_log_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Expand(line) => {
                self.handle_expand_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Rationalize(line) => {
                self.handle_rationalize_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Limit(line) => {
                self.handle_limit_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Profile(line) => {
                self.handle_profile_command_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Health(line) => {
                self.handle_health_core(line)
            }
            cas_solver::session_api::repl::ReplCommandInput::Eval(line) => {
                self.handle_eval_core(line)
            }
        }
    }
}
