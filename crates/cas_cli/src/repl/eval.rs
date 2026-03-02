use super::*;

impl Repl {
    /// Core: handle eval command, returns ReplReply (no I/O)
    pub(crate) fn handle_eval_core(&mut self, line: &str) -> ReplReply {
        use cas_formatter::root_style::ParseStyleSignals;

        let mut reply: ReplReply = vec![];

        let style_signals = ParseStyleSignals::from_input_string(line);
        match cas_solver::evaluate_eval_command_output(
            &mut self.core.engine,
            &mut self.core.state,
            line,
            &style_signals,
            self.core.debug_mode,
        ) {
            Ok(out) => {
                if let Some(line) = out.stored_entry_line {
                    reply.push(ReplMsg::output(line));
                }

                for line in out.metadata.warning_lines {
                    reply.push(ReplMsg::warn(line));
                }

                for line in out.metadata.requires_lines {
                    reply.push(ReplMsg::info(line));
                }

                if !out.steps.is_empty() || self.verbosity != Verbosity::None {
                    let steps_reply = self.show_simplification_steps_core(
                        out.resolved_expr,
                        &out.steps,
                        style_signals.clone(),
                    );
                    reply.extend(steps_reply);
                }

                if let Some(result_line) = out.result_line {
                    reply.push(ReplMsg::output(result_line.line));
                    if result_line.terminal {
                        return reply;
                    }
                }

                for line in out.metadata.hint_lines {
                    reply.push(ReplMsg::info(line));
                }
                for line in out.metadata.assumption_lines {
                    reply.push(ReplMsg::info(line));
                }
            }
            Err(cas_solver::EvalCommandError::Parse(e)) => reply.push(ReplMsg::error(
                super::error_render::render_parse_error(line, &e),
            )),
            Err(cas_solver::EvalCommandError::Eval(message)) => reply.push(ReplMsg::error(message)),
        }

        reply
    }
}
