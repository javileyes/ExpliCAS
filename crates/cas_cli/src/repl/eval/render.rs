use super::super::*;

impl Repl {
    pub(crate) fn render_eval_plan_to_reply(
        &mut self,
        plan: cas_solver::session_api::types::EvalCommandRenderPlan,
    ) -> ReplReply {
        let mut reply: ReplReply = vec![];

        for message in plan.pre_messages {
            match message.kind {
                cas_session::eval_api::EvalDisplayMessageKind::Output => {
                    reply.push(ReplMsg::output(message.text))
                }
                cas_session::eval_api::EvalDisplayMessageKind::Warn => {
                    reply.push(ReplMsg::warn(message.text))
                }
                cas_session::eval_api::EvalDisplayMessageKind::Info => {
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

        reply
    }
}
