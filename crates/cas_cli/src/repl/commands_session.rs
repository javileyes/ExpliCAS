use super::*;

impl Repl {
    // ========== SESSION ENVIRONMENT HANDLERS ==========

    pub(crate) fn handle_budget_command_core(&mut self, line: &str) -> ReplReply {
        reply_output(
            cas_session::evaluate_solve_budget_command_message_on_repl_core(&mut self.core, line),
        )
    }

    /// Handle "let <name> = <expr>" (eager) or "let <name> := <expr>" (lazy) command
    pub(crate) fn handle_let_command_core(&mut self, rest: &str) -> ReplReply {
        match cas_session::evaluate_let_assignment_command_message_on_repl_core(
            &mut self.core,
            rest,
        ) {
            Ok(message) => reply_output(message),
            Err(message) => reply_output(message),
        }
    }

    /// Handle variable assignment (from "let" or ":=")
    /// - eager=false (=): evaluate then store (unwrap __hold)
    /// - eager=true (:=): store formula without evaluating
    pub(crate) fn handle_assignment_core(
        &mut self,
        name: &str,
        expr_str: &str,
        lazy: bool,
    ) -> ReplReply {
        match cas_session::evaluate_assignment_command_message_on_repl_core(
            &mut self.core,
            name,
            expr_str,
            lazy,
        ) {
            Ok(message) => reply_output(message),
            Err(message) => reply_output(message),
        }
    }

    /// Handle "vars" command - list all variable bindings
    /// Core logic for "vars" command
    pub(crate) fn handle_vars_command_core(&self) -> ReplReply {
        reply_output(cas_session::evaluate_vars_command_message_on_repl_core(
            &self.core,
        ))
    }

    /// Handle "history" or "list" command - show session history
    pub(crate) fn handle_history_command_core(&self) -> ReplReply {
        reply_output(cas_session::evaluate_history_command_message_on_repl_core(
            &self.core,
        ))
    }

    /// Handle "show #id" command - show details of a specific entry
    pub(crate) fn handle_show_command_core(&mut self, line: &str) -> ReplReply {
        match cas_session::evaluate_show_command_lines_on_repl_core(&mut self.core, line) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    /// Handle "del #id [#id...]" command - delete session entries
    pub(crate) fn handle_del_command_core(&mut self, line: &str) -> ReplReply {
        reply_output(
            cas_session::evaluate_delete_history_command_message_on_repl_core(&mut self.core, line),
        )
    }

    /// Handle "clear" or "clear <names>" command
    /// Core logic for "clear" command
    pub(crate) fn handle_clear_command_core(&mut self, line: &str) -> ReplReply {
        let lines = cas_session::evaluate_clear_command_lines_on_repl_core(&mut self.core, line);
        let mut reply = ReplReply::new();
        for line in lines {
            reply.push(ReplMsg::output(line));
        }
        reply
    }

    /// Handle "reset" command - reset entire session
    /// Keeps access to both core and config.
    pub(crate) fn handle_reset_command_core(&mut self) -> ReplReply {
        cas_session::reset_repl_core_with_config(&mut self.core, &self.config);

        reply_output("Session reset. Environment and context cleared.")
    }

    /// Handle "reset full" command - reset session AND clear profile cache
    pub(crate) fn handle_reset_full_command_core(&mut self) -> ReplReply {
        cas_session::reset_repl_core_full_with_config(&mut self.core, &self.config);
        reply_output(
            "Session reset. Environment and context cleared.\n\
             Profile cache cleared (will rebuild on next eval).",
        )
    }

    /// Handle "cache" command - show status or clear cache
    /// Core logic for "cache" command
    pub(crate) fn handle_cache_command_core(&mut self, line: &str) -> ReplReply {
        let mut reply = ReplReply::new();
        for line in
            cas_session::evaluate_profile_cache_command_lines_on_repl_core(&mut self.core, line)
        {
            reply.push(ReplMsg::output(line));
        }
        reply
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_output(reply: &ReplReply) -> &str {
        for msg in reply {
            if let ReplMsg::Output(text) = msg {
                return text.as_str();
            }
        }
        panic!("reply should contain output");
    }

    #[test]
    fn cache_status_uses_engine_profile_cache() {
        let mut repl = Repl::new();

        if let Err(err) = cas_session::evaluate_eval_command_render_plan_on_repl_core(
            &mut repl.core,
            "x + x",
            false,
        ) {
            panic!("eval failed: {err:?}");
        }

        let status = repl.handle_cache_command_core("cache status");
        let output = first_output(&status);
        assert!(
            output.contains("Profile Cache: 1 profiles cached"),
            "unexpected cache status output: {output}"
        );
    }

    #[test]
    fn cache_clear_empties_engine_profile_cache() {
        let mut repl = Repl::new();

        if let Err(err) = cas_session::evaluate_eval_command_render_plan_on_repl_core(
            &mut repl.core,
            "x + x",
            false,
        ) {
            panic!("eval failed: {err:?}");
        }
        assert_eq!(cas_session::profile_cache_len_on_repl_core(&repl.core), 1);

        let clear = repl.handle_cache_command_core("cache clear");
        assert!(first_output(&clear).contains("Profile cache cleared"));
        assert_eq!(cas_session::profile_cache_len_on_repl_core(&repl.core), 0);
    }
}
