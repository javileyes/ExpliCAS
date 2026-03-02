use super::*;

impl Repl {
    /// Core: handle eval command, returns ReplReply (no I/O)
    pub(crate) fn handle_eval_core(&mut self, line: &str) -> ReplReply {
        let mut reply: ReplReply = vec![];

        match cas_solver::evaluate_eval_command_output(
            &mut self.core.engine,
            &mut self.core.state,
            line,
            self.core.debug_mode,
        ) {
            Ok(out) => {
                let plan = cas_solver::build_eval_command_render_plan(
                    out,
                    self.verbosity == Verbosity::None,
                );

                for message in plan.pre_messages {
                    match message.kind {
                        cas_solver::EvalDisplayMessageKind::Output => {
                            reply.push(ReplMsg::output(message.text))
                        }
                        cas_solver::EvalDisplayMessageKind::Warn => {
                            reply.push(ReplMsg::warn(message.text))
                        }
                        cas_solver::EvalDisplayMessageKind::Info => {
                            reply.push(ReplMsg::info(message.text))
                        }
                    }
                }

                if plan.render_steps {
                    let steps_reply = self.show_simplification_steps_core(
                        plan.resolved_expr,
                        &plan.steps,
                        plan.style_signals.clone(),
                    );
                    reply.extend(steps_reply);
                }

                if let Some(result_message) = plan.result_message {
                    reply.push(ReplMsg::output(result_message.text));
                    if plan.result_terminal {
                        return reply;
                    }
                }

                for message in plan.post_messages {
                    reply.push(ReplMsg::info(message.text));
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
