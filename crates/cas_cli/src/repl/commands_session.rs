use super::*;

impl Repl {
    // ========== SESSION ENVIRONMENT HANDLERS ==========

    /// Handle "let <name> = <expr>" (eager) or "let <name> := <expr>" (lazy) command
    pub(crate) fn handle_let_command(&mut self, rest: &str) {
        let reply = self.handle_let_command_core(rest);
        self.print_reply(reply);
    }

    fn handle_let_command_core(&mut self, rest: &str) -> ReplReply {
        match cas_session::parse_let_assignment_input(rest) {
            Ok(parsed) => self.handle_assignment_core(parsed.name, parsed.expr, parsed.lazy),
            Err(cas_session::LetAssignmentParseError::MissingAssignmentOperator) => reply_output(
                "Usage: let <name> = <expr>   (eager - evaluates)\n\
                        let <name> := <expr>  (lazy - stores formula)\n\
                 Example: let a = expand((1+x)^3)",
            ),
        }
    }

    /// Handle variable assignment (from "let" or ":=")
    /// - eager=false (=): evaluate then store (unwrap __hold)
    /// - eager=true (:=): store formula without evaluating
    pub(crate) fn handle_assignment(&mut self, name: &str, expr_str: &str, lazy: bool) {
        let reply = self.handle_assignment_core(name, expr_str, lazy);
        self.print_reply(reply);
    }

    fn handle_assignment_core(&mut self, name: &str, expr_str: &str, lazy: bool) -> ReplReply {
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
                if lazy {
                    reply_output(format!("{} := {}", name, display))
                } else {
                    reply_output(format!("{} = {}", name, display))
                }
            }
            Err(cas_session::AssignmentError::EmptyName) => {
                reply_output("Error: Variable name cannot be empty")
            }
            Err(cas_session::AssignmentError::InvalidNameStart) => {
                reply_output("Error: Variable name must start with a letter or underscore")
            }
            Err(cas_session::AssignmentError::ReservedName(name)) => reply_output(format!(
                "Error: '{}' is a reserved name and cannot be assigned",
                name
            )),
            Err(cas_session::AssignmentError::Parse(e)) => {
                reply_output(format!("Parse error: {}", e))
            }
        }
    }

    /// Handle "vars" command - list all variable bindings
    pub(crate) fn handle_vars_command(&self) {
        let reply = self.handle_vars_command_core();
        self.print_reply(reply);
    }

    /// Core logic for "vars" command
    fn handle_vars_command_core(&self) -> ReplReply {
        let bindings = self.core.state.bindings();
        if bindings.is_empty() {
            reply_output("No variables defined.")
        } else {
            let mut lines = vec!["Variables:".to_string()];
            for (name, expr_id) in bindings {
                let display = cas_formatter::DisplayExpr {
                    context: &self.core.engine.simplifier.context,
                    id: expr_id,
                };
                lines.push(format!("  {} = {}", name, display));
            }
            reply_output(lines.join("\n"))
        }
    }

    /// Handle "clear" or "clear <names>" command
    pub(crate) fn handle_clear_command(&mut self, line: &str) {
        let reply = self.handle_clear_command_core(line);
        self.print_reply(reply);
    }

    /// Core logic for "clear" command
    fn handle_clear_command_core(&mut self, line: &str) -> ReplReply {
        let mut reply = ReplReply::new();
        if line == "clear" {
            // Clear all
            let count = self.core.state.binding_count();
            self.core.state.clear_bindings();
            if count == 0 {
                reply.push(ReplMsg::output("No variables to clear."));
            } else {
                reply.push(ReplMsg::output(format!("Cleared {} variable(s).", count)));
            }
        } else {
            // Clear specific variables
            let names: Vec<&str> = line[6..].split_whitespace().collect();
            let mut cleared = 0;
            for name in names {
                if self.core.state.unset_binding(name) {
                    cleared += 1;
                } else {
                    reply.push(ReplMsg::output(format!(
                        "Warning: '{}' was not defined",
                        name
                    )));
                }
            }
            if cleared > 0 {
                reply.push(ReplMsg::output(format!("Cleared {} variable(s).", cleared)));
            }
        }
        reply
    }

    /// Handle "reset" command - reset entire session
    pub(crate) fn handle_reset_command(&mut self) {
        let reply = self.handle_reset_command_impl();
        self.print_reply(reply);
    }

    /// Implementation of reset - keeps access to both core and config
    fn handle_reset_command_impl(&mut self) -> ReplReply {
        // Clear session state (history + env)
        self.core.state.clear();

        // Reset simplifier with new context
        self.core.engine.simplifier = Simplifier::with_default_rules();

        // Re-register custom rules (same as in new())
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(cas_solver::rules::functions::AbsSquaredRule));
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(EvaluateTrigRule));
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(PythagoreanIdentityRule));
        if self.config.trig_angle_sum {
            self.core
                .engine
                .simplifier
                .add_rule(Box::new(AngleIdentityRule));
        }
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(TanToSinCosRule));
        if self.config.trig_double_angle {
            self.core
                .engine
                .simplifier
                .add_rule(Box::new(DoubleAngleRule));
        }
        if self.config.canonicalize_trig_square {
            self.core.engine.simplifier.add_rule(Box::new(
                cas_solver::rules::trigonometry::CanonicalizeTrigSquareRule,
            ));
        }
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(EvaluateLogRule));
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(ExponentialLogRule));
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(SimplifyFractionRule));
        self.core.engine.simplifier.add_rule(Box::new(ExpandRule));
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(cas_solver::rules::algebra::ConservativeExpandRule));

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
    pub(crate) fn handle_reset_full_command(&mut self) {
        // First do normal reset (which prints its message)
        let reset_reply = self.handle_reset_command_impl();
        self.print_reply(reset_reply);

        // Also clear profile cache
        self.core.engine.clear_profile_cache();

        self.print_reply(reply_output(
            "Profile cache cleared (will rebuild on next eval).",
        ));
    }

    /// Handle "cache" command - show status or clear cache
    pub(crate) fn handle_cache_command(&mut self, line: &str) {
        let reply = self.handle_cache_command_core(line);
        self.print_reply(reply);
    }

    /// Core logic for "cache" command
    fn handle_cache_command_core(&mut self, line: &str) -> ReplReply {
        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1).copied() {
            None | Some("status") => {
                // Show cache status
                let count = self.core.engine.profile_cache_len();
                let mut lines = vec![format!("Profile Cache: {} profiles cached", count)];
                if count == 0 {
                    lines.push("  (empty - profiles will be built on first eval)".to_string());
                } else {
                    lines.push("  (profiles are reused across evaluations)".to_string());
                }
                reply_output(lines.join("\n"))
            }
            Some("clear") => {
                self.core.engine.clear_profile_cache();
                reply_output("Profile cache cleared.")
            }
            Some(cmd) => {
                let mut reply = ReplReply::new();
                reply.push(ReplMsg::output(format!("Unknown cache command: {}", cmd)));
                reply.push(ReplMsg::output("Usage: cache [status|clear]"));
                reply
            }
        }
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
