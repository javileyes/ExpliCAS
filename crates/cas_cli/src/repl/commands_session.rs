use super::*;

#[derive(Debug, Clone, Copy)]
struct EvalMetadataSectionLabels<'a> {
    required_header: &'a str,
    assumed_header: &'a str,
    blocked_header: &'a str,
    line_prefix: &'a str,
}

fn history_eval_metadata_section_labels() -> EvalMetadataSectionLabels<'static> {
    EvalMetadataSectionLabels {
        required_header: "  ℹ️ Requires:",
        assumed_header: "  ⚠ Assumed:",
        blocked_header: "  🚫 Blocked:",
        line_prefix: "    - ",
    }
}

fn format_eval_metadata_sections(
    ctx: &cas_ast::Context,
    required_conditions: &[cas_solver::ImplicitCondition],
    domain_warnings: &[cas_solver::DomainWarning],
    blocked_hints: &[cas_solver::BlockedHint],
    labels: EvalMetadataSectionLabels<'_>,
) -> Vec<String> {
    let mut lines = Vec::new();

    if !required_conditions.is_empty() {
        lines.push(labels.required_header.to_string());
        lines.extend(cas_solver::format_required_condition_lines(
            ctx,
            required_conditions,
            labels.line_prefix,
        ));
    }

    if !domain_warnings.is_empty() {
        lines.push(labels.assumed_header.to_string());
        lines.extend(cas_solver::format_domain_warning_lines(
            domain_warnings,
            false,
            labels.line_prefix,
        ));
    }

    if !blocked_hints.is_empty() {
        lines.push(labels.blocked_header.to_string());
        lines.extend(cas_solver::format_blocked_hint_lines(
            blocked_hints,
            labels.line_prefix,
        ));
    }

    lines
}

impl Repl {
    // ========== SESSION ENVIRONMENT HANDLERS ==========

    pub(crate) fn handle_budget_command_core(&mut self, line: &str) -> ReplReply {
        let result = cas_session::apply_solve_budget_command(&mut self.core.state, line);
        reply_output(cas_session::format_solve_budget_command_message(&result))
    }

    /// Handle "let <name> = <expr>" (eager) or "let <name> := <expr>" (lazy) command
    pub(crate) fn handle_let_command_core(&mut self, rest: &str) -> ReplReply {
        match cas_session::evaluate_let_assignment_command_message_with_simplifier(
            &mut self.core.state,
            &mut self.core.engine.simplifier,
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
        match cas_session::evaluate_assignment_command_message_with_simplifier(
            &mut self.core.state,
            &mut self.core.engine.simplifier,
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
        let lines = cas_session::evaluate_vars_command_lines_with_context(
            &self.core.state,
            &self.core.engine.simplifier.context,
        );
        reply_output(lines.join("\n"))
    }

    /// Handle "history" or "list" command - show session history
    pub(crate) fn handle_history_command_core(&self) -> ReplReply {
        let lines = cas_session::evaluate_history_command_lines_with_context(
            &self.core.state,
            &self.core.engine.simplifier.context,
        );
        reply_output(lines.join("\n"))
    }

    /// Handle "show #id" command - show details of a specific entry
    pub(crate) fn handle_show_command_core(&mut self, line: &str) -> ReplReply {
        let inspection = match cas_session::inspect_history_entry_input(
            &mut self.core.state,
            &mut self.core.engine,
            line,
        ) {
            Ok(inspection) => inspection,
            Err(error) => {
                return reply_output(cas_session::format_inspect_history_entry_error_message(
                    &error,
                ))
            }
        };

        let lines = cas_session::format_show_history_command_lines_with_context(
            &inspection,
            &self.core.engine.simplifier.context,
            |context, expr_info| {
                format_eval_metadata_sections(
                    context,
                    &expr_info.required_conditions,
                    &expr_info.domain_warnings,
                    &expr_info.blocked_hints,
                    history_eval_metadata_section_labels(),
                )
            },
        );
        reply_output(lines.join("\n"))
    }

    /// Handle "del #id [#id...]" command - delete session entries
    pub(crate) fn handle_del_command_core(&mut self, line: &str) -> ReplReply {
        reply_output(cas_session::evaluate_delete_history_command_message(
            &mut self.core.state,
            line,
        ))
    }

    /// Handle "clear" or "clear <names>" command
    /// Core logic for "clear" command
    pub(crate) fn handle_clear_command_core(&mut self, line: &str) -> ReplReply {
        let lines = cas_session::evaluate_clear_command_lines(&mut self.core.state, line);
        let mut reply = ReplReply::new();
        for line in lines {
            reply.push(ReplMsg::output(line));
        }
        reply
    }

    /// Handle "reset" command - reset entire session
    /// Keeps access to both core and config.
    pub(crate) fn handle_reset_command_core(&mut self) -> ReplReply {
        // Clear session state (history + env)
        self.core.state.clear();

        // Rebuild full configured simplifier portfolio (same source as Repl::new()).
        self.rebuild_engine_simplifier_from_config();

        // Sync config
        self.sync_config_to_simplifier();

        // Reset options
        self.core.debug_mode = false;
        self.core.last_stats = None;
        self.core.health_enabled = false;
        self.core.last_health_report = None;

        reply_output("Session reset. Environment and context cleared.")
    }

    /// Handle "reset full" command - reset session AND clear profile cache
    pub(crate) fn handle_reset_full_command_core(&mut self) -> ReplReply {
        let mut reply = self.handle_reset_command_core();
        self.core.engine.clear_profile_cache();
        reply.push(ReplMsg::output(
            "Profile cache cleared (will rebuild on next eval).",
        ));
        reply
    }

    /// Handle "cache" command - show status or clear cache
    /// Core logic for "cache" command
    pub(crate) fn handle_cache_command_core(&mut self, line: &str) -> ReplReply {
        let mut reply = ReplReply::new();
        let cache_result = cas_session::apply_profile_cache_command(&mut self.core.engine, line);
        for line in cas_session::format_profile_cache_command_lines(&cache_result) {
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

        if let Err(err) = crate::commands::eval_command::evaluate_eval_command_output(
            &mut repl.core.engine,
            &mut repl.core.state,
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

        if let Err(err) = crate::commands::eval_command::evaluate_eval_command_output(
            &mut repl.core.engine,
            &mut repl.core.state,
            "x + x",
            false,
        ) {
            panic!("eval failed: {err:?}");
        }
        assert_eq!(repl.core.engine.profile_cache_len(), 1);

        let clear = repl.handle_cache_command_core("cache clear");
        assert!(first_output(&clear).contains("Profile cache cleared"));
        assert_eq!(repl.core.engine.profile_cache_len(), 0);
    }
}
