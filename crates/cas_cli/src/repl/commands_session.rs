use super::*;

impl Repl {
    // ========== SESSION ENVIRONMENT HANDLERS ==========

    pub(crate) fn handle_budget_command_core(&mut self, line: &str) -> ReplReply {
        let result = cas_session::apply_solve_budget_command(&mut self.core.state, line);
        reply_output(cas_session::format_solve_budget_command_message(&result))
    }

    /// Handle "let <name> = <expr>" (eager) or "let <name> := <expr>" (lazy) command
    pub(crate) fn handle_let_command_core(&mut self, rest: &str) -> ReplReply {
        match cas_session::parse_let_assignment_input(rest) {
            Ok(parsed) => self.handle_assignment_core(parsed.name, parsed.expr, parsed.lazy),
            Err(e) => reply_output(cas_session::format_let_assignment_parse_error_message(&e)),
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
        match cas_session::apply_assignment(
            &mut self.core.state,
            &mut self.core.engine.simplifier,
            name,
            expr_str,
            lazy,
        ) {
            Ok(result) => {
                let display = cas_formatter::DisplayExpr {
                    context: &self.core.engine.simplifier.context,
                    id: result,
                };
                reply_output(cas_session::format_assignment_success_message(
                    name,
                    &display.to_string(),
                    lazy,
                ))
            }
            Err(e) => reply_output(cas_session::format_assignment_error_message(&e)),
        }
    }

    /// Handle "vars" command - list all variable bindings
    /// Core logic for "vars" command
    pub(crate) fn handle_vars_command_core(&self) -> ReplReply {
        let bindings = cas_session::binding_overview_entries(&self.core.state);
        if bindings.is_empty() {
            reply_output(cas_session::vars_empty_message())
        } else {
            let lines = cas_session::format_binding_overview_lines(&bindings, |id| {
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &self.core.engine.simplifier.context,
                        id
                    }
                )
            });
            reply_output(lines.join("\n"))
        }
    }

    /// Handle "history" or "list" command - show session history
    pub(crate) fn handle_history_command_core(&self) -> ReplReply {
        let entries = cas_session::history_overview_entries(&self.core.state);
        if entries.is_empty() {
            return reply_output(cas_session::history_empty_message());
        }

        let lines = cas_session::format_history_overview_lines(&entries, |id| {
            format!(
                "{}",
                DisplayExpr {
                    context: &self.core.engine.simplifier.context,
                    id
                }
            )
        });
        reply_output(lines.join("\n"))
    }

    /// Handle "show #id" command - show details of a specific entry
    pub(crate) fn handle_show_command_core(&mut self, line: &str) -> ReplReply {
        match cas_session::inspect_history_entry_input(
            &mut self.core.state,
            &mut self.core.engine,
            line,
        ) {
            Ok(inspection) => {
                let mut lines =
                    cas_session::format_history_entry_inspection_lines(&inspection, |id| {
                        format!(
                            "{}",
                            DisplayExpr {
                                context: &self.core.engine.simplifier.context,
                                id
                            }
                        )
                    });
                if let cas_session::HistoryEntryDetails::Expr(expr_info) = &inspection.details {
                    lines.extend(cas_solver::format_eval_metadata_sections(
                        &self.core.engine.simplifier.context,
                        &expr_info.required_conditions,
                        &expr_info.domain_warnings,
                        &expr_info.blocked_hints,
                        cas_solver::history_eval_metadata_section_labels(),
                    ));
                }
                reply_output(lines.join("\n"))
            }
            Err(e) => reply_output(cas_session::format_inspect_history_entry_error_message(&e)),
        }
    }

    /// Handle "del #id [#id...]" command - delete session entries
    pub(crate) fn handle_del_command_core(&mut self, line: &str) -> ReplReply {
        match cas_session::delete_history_entries(&mut self.core.state, line) {
            Ok(result) => reply_output(cas_session::format_delete_history_result_message(&result)),
            Err(e) => reply_output(cas_session::format_delete_history_error_message(&e)),
        }
    }

    /// Handle "clear" or "clear <names>" command
    /// Core logic for "clear" command
    pub(crate) fn handle_clear_command_core(&mut self, line: &str) -> ReplReply {
        let result = cas_session::clear_bindings_command(&mut self.core.state, line);
        let mut reply = ReplReply::new();
        for line in cas_session::format_clear_bindings_result_lines(&result) {
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
        let result = cas_solver::apply_profile_cache_command(&mut self.core.engine, line);
        let mut reply = ReplReply::new();
        for line in cas_solver::format_profile_cache_command_lines(&result) {
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

        let parsed = match cas_parser::parse("x + x", &mut repl.core.engine.simplifier.context) {
            Ok(id) => id,
            Err(err) => panic!("parse expression failed: {err}"),
        };
        let req = cas_solver::EvalRequest {
            raw_input: "x + x".to_string(),
            parsed,
            action: cas_solver::EvalAction::Simplify,
            auto_store: false,
        };
        if let Err(err) = repl
            .core
            .engine
            .eval_stateless(cas_solver::EvalOptions::default(), req)
        {
            panic!("eval failed: {err}");
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

        let parsed = match cas_parser::parse("x + x", &mut repl.core.engine.simplifier.context) {
            Ok(id) => id,
            Err(err) => panic!("parse expression failed: {err}"),
        };
        let req = cas_solver::EvalRequest {
            raw_input: "x + x".to_string(),
            parsed,
            action: cas_solver::EvalAction::Simplify,
            auto_store: false,
        };
        if let Err(err) = repl
            .core
            .engine
            .eval_stateless(cas_solver::EvalOptions::default(), req)
        {
            panic!("eval failed: {err}");
        }
        assert_eq!(repl.core.engine.profile_cache_len(), 1);

        let clear = repl.handle_cache_command_core("cache clear");
        assert!(first_output(&clear).contains("Profile cache cleared"));
        assert_eq!(repl.core.engine.profile_cache_len(), 0);
    }
}
