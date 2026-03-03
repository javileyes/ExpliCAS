use super::*;

impl Repl {
    /// Core: handle eval command, returns ReplReply (no I/O)
    pub(crate) fn handle_eval_core(&mut self, line: &str) -> ReplReply {
        let mut reply: ReplReply = vec![];

        match cas_session::evaluate_eval_command_render_plan_on_repl_core(
            &mut self.core,
            line,
            self.verbosity == Verbosity::None,
        ) {
            Ok(plan) => {
                for message in plan.pre_messages {
                    match message.kind {
                        cas_session::EvalDisplayMessageKind::Output => {
                            reply.push(ReplMsg::output(message.text))
                        }
                        cas_session::EvalDisplayMessageKind::Warn => {
                            reply.push(ReplMsg::warn(message.text))
                        }
                        cas_session::EvalDisplayMessageKind::Info => {
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
            Err(message) => reply.push(ReplMsg::error(message)),
        }

        reply
    }
}
