use super::*;

impl Repl {
    pub(crate) fn handle_det(&mut self, line: &str) {
        let reply = self.handle_det_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_det_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let rest = cas_solver::extract_unary_command_tail(line, "det");

        let result = cas_solver::evaluate_unary_function_input(
            &mut self.core.engine.simplifier,
            "det",
            rest,
        );
        match result {
            Ok(out) => {
                let render_config = cas_solver::unary_render_config_for_display_mode(
                    "det",
                    Self::set_display_mode_from_verbosity(verbosity),
                    true,
                );
                let mut lines = cas_solver::format_unary_function_eval_lines(
                    &self.core.engine.simplifier.context,
                    rest,
                    &out,
                    render_config,
                );
                clean_result_line(&mut lines);
                reply_output(lines.join("\n"))
            }
            Err(e) => reply_output(cas_solver::format_unary_function_eval_error_message(&e)),
        }
    }

    pub(crate) fn handle_transpose(&mut self, line: &str) {
        let reply = self.handle_transpose_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_transpose_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let rest = cas_solver::extract_unary_command_tail(line, "transpose");

        let result = cas_solver::evaluate_unary_function_input(
            &mut self.core.engine.simplifier,
            "transpose",
            rest,
        );
        match result {
            Ok(out) => {
                let render_config = cas_solver::unary_render_config_for_display_mode(
                    "transpose",
                    Self::set_display_mode_from_verbosity(verbosity),
                    false,
                );
                let lines = cas_solver::format_unary_function_eval_lines(
                    &self.core.engine.simplifier.context,
                    rest,
                    &out,
                    render_config,
                );
                reply_output(lines.join("\n"))
            }
            Err(e) => reply_output(cas_solver::format_unary_function_eval_error_message(&e)),
        }
    }

    pub(crate) fn handle_trace(&mut self, line: &str) {
        let reply = self.handle_trace_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_trace_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let rest = cas_solver::extract_unary_command_tail(line, "trace");

        let result = cas_solver::evaluate_unary_function_input(
            &mut self.core.engine.simplifier,
            "trace",
            rest,
        );
        match result {
            Ok(out) => {
                let render_config = cas_solver::unary_render_config_for_display_mode(
                    "trace",
                    Self::set_display_mode_from_verbosity(verbosity),
                    false,
                );
                let mut lines = cas_solver::format_unary_function_eval_lines(
                    &self.core.engine.simplifier.context,
                    rest,
                    &out,
                    render_config,
                );
                clean_result_line(&mut lines);
                reply_output(lines.join("\n"))
            }
            Err(e) => reply_output(cas_solver::format_unary_function_eval_error_message(&e)),
        }
    }

    /// Handle the 'telescope' command for proving telescoping identities like Dirichlet kernel
    pub(crate) fn handle_telescope(&mut self, line: &str) {
        let reply = self.handle_telescope_core(line);
        self.print_reply(reply);
    }

    fn handle_telescope_core(&mut self, line: &str) -> ReplReply {
        let rest = match cas_solver::parse_telescope_command_input(line) {
            cas_solver::TelescopeCommandInput::MissingInput => {
                return reply_output(cas_solver::telescope_usage_message());
            }
            cas_solver::TelescopeCommandInput::Expr(rest) => rest,
        };

        match cas_solver::evaluate_telescope_input(&mut self.core.engine.simplifier, rest) {
            Ok(out) => reply_output(cas_solver::format_telescope_eval_lines(rest, &out).join("\n")),
            Err(e) => reply_output(cas_solver::format_transform_eval_error_message(&e)),
        }
    }

    /// Handle the 'expand' command for aggressive polynomial expansion
    /// Uses the engine `expand()` path which distributes without educational guards
    pub(crate) fn handle_expand(&mut self, line: &str) {
        let rest = match cas_solver::parse_expand_command_input(line) {
            cas_solver::ExpandCommandInput::MissingInput => {
                self.print_reply(reply_output(cas_solver::expand_usage_message()));
                return;
            }
            cas_solver::ExpandCommandInput::Expr(rest) => rest,
        };

        // V2.14.34: Delegate to normal line processing with expand() function wrapper
        // This ensures steps are shown, consistent with using expand() as a function
        let wrapped = cas_solver::wrap_expand_eval_expression(rest);
        self.handle_eval(&wrapped);
    }

    /// Handle the 'expand_log' command for explicit logarithm expansion
    /// Expands ln(xy) → ln(x) + ln(y), ln(x/y) → ln(x) - ln(y), ln(x^n) → n*ln(x)
    pub(crate) fn handle_expand_log(&mut self, line: &str) {
        let reply = self.handle_expand_log_core(line);
        self.print_reply(reply);
    }

    fn handle_expand_log_core(&mut self, line: &str) -> ReplReply {
        let rest = match cas_solver::parse_expand_log_command_input(line) {
            cas_solver::ExpandLogCommandInput::MissingInput => {
                return reply_output(cas_solver::expand_log_usage_message());
            }
            cas_solver::ExpandLogCommandInput::Expr(rest) => rest,
        };

        match cas_solver::evaluate_expand_log_input(&mut self.core.engine.simplifier, rest) {
            Ok(out) => {
                let mut lines = cas_solver::format_expand_log_eval_lines(
                    &self.core.engine.simplifier.context,
                    &out,
                );
                clean_result_line(&mut lines);
                reply_output(lines.join("\n"))
            }
            Err(e) => reply_output(cas_solver::format_transform_eval_error_message(&e)),
        }
    }
}
