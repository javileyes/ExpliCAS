use super::output::ReplReply;
use super::*;

impl Repl {
    /// Main command dispatch - calls core and prints result.
    ///
    /// # Panic Fence (Defense in Depth)
    /// Wraps command execution with `catch_unwind` to prevent internal panics
    /// from crashing the REPL session. On panic:
    /// - Generates a unique error_id for correlation
    /// - Displays a user-friendly error message with error_id
    /// - Logs details to stderr (controlled by EXPLICAS_PANIC_REPORT)
    /// - Session remains alive for continued use
    pub fn handle_command(&mut self, line: &str) {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        // Wrap in catch_unwind to prevent panics from killing the session
        // AssertUnwindSafe is needed because &mut self is not UnwindSafe
        let result = catch_unwind(AssertUnwindSafe(|| self.handle_command_core(line)));

        match result {
            Ok(reply) => self.print_reply(reply),
            Err(panic_info) => {
                let panic_msg = cas_solver::panic_payload_to_message(panic_info.as_ref());

                // Generate short error_id for correlation (hash of timestamp + message)
                let error_id = cas_solver::generate_short_error_id(&panic_msg);

                // Log to stderr if EXPLICAS_PANIC_REPORT is set (for debugging)
                // This respects the repl IO lint by only logging when explicitly enabled
                if std::env::var("EXPLICAS_PANIC_REPORT").is_ok() {
                    // Use init.rs print_reply with Error variant which goes to stderr
                    let version = env!("CARGO_PKG_VERSION");
                    let log_msg = cas_solver::format_panic_report_message(
                        &error_id, version, line, &panic_msg,
                    );
                    // Log via ReplMsg::Debug to get it to output
                    self.print_reply(vec![ReplMsg::Debug(log_msg)]);
                }

                // Return user-friendly error that doesn't kill the session
                self.print_reply(vec![ReplMsg::Error(cas_solver::format_user_panic_message(
                    &error_id, &panic_msg,
                ))]);
            }
        }
    }

    /// Core command dispatch - returns structured messages, no I/O.
    /// This is the heart of ReplCore logic.
    pub fn handle_command_core(&mut self, line: &str) -> ReplReply {
        // Preprocess: Convert function-style commands to command-style
        // simplify(...) -> simplify ...
        // solve(...) -> solve ...
        let line = self.preprocess_function_syntax(line);

        match cas_solver::parse_repl_command_input(&line) {
            cas_solver::ReplCommandInput::Help(line) => self.handle_help_core(line),
            cas_solver::ReplCommandInput::Let(rest) => self.handle_let_command_core(rest),
            cas_solver::ReplCommandInput::Assignment { name, expr, lazy } => {
                self.handle_assignment_core(name, expr, lazy)
            }
            cas_solver::ReplCommandInput::Vars => self.handle_vars_command_core(),
            cas_solver::ReplCommandInput::Clear(line) => self.handle_clear_command_core(line),
            cas_solver::ReplCommandInput::Reset => self.handle_reset_command_core(),
            cas_solver::ReplCommandInput::ResetFull => self.handle_reset_full_command_core(),
            cas_solver::ReplCommandInput::Cache(line) => self.handle_cache_command_core(line),
            cas_solver::ReplCommandInput::Semantics(line) => self.handle_semantics_core(line),
            cas_solver::ReplCommandInput::Context(line) => self.handle_context_command_core(line),
            cas_solver::ReplCommandInput::Steps(line) => self.handle_steps_command_core(line),
            cas_solver::ReplCommandInput::Autoexpand(line) => {
                self.handle_autoexpand_command_core(line)
            }
            cas_solver::ReplCommandInput::Budget(line) => self.handle_budget_command_core(line),
            cas_solver::ReplCommandInput::History => self.handle_history_command_core(),
            cas_solver::ReplCommandInput::Show(rest) => self.handle_show_command_core(rest),
            cas_solver::ReplCommandInput::Del(rest) => self.handle_del_command_core(rest),
            cas_solver::ReplCommandInput::Set(line) => {
                let result = self.handle_set_command_core(line);
                self.finalize_core_result(result)
            }
            cas_solver::ReplCommandInput::Equiv(line) => {
                self.handle_equiv(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Subst(line) => {
                self.handle_subst(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::SolveSystem(line) => {
                self.handle_solve_system(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Solve(line) => {
                self.handle_solve(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Simplify(line) => {
                self.handle_full_simplify(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Config(line) => {
                self.handle_config(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Timeline(line) => {
                self.handle_timeline(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Visualize(line) => {
                self.handle_visualize(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Explain(line) => {
                self.handle_explain(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Det(line) => {
                self.handle_det(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Transpose(line) => {
                self.handle_transpose(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Trace(line) => {
                self.handle_trace(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Telescope(line) => {
                self.handle_telescope(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Weierstrass(line) => {
                self.handle_weierstrass(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::ExpandLog(line) => {
                self.handle_expand_log(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Expand(line) => {
                self.handle_expand(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Rationalize(line) => {
                self.handle_rationalize(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Limit(line) => {
                self.handle_limit(line);
                ReplReply::new()
            }
            cas_solver::ReplCommandInput::Profile(line) => reply_output(
                cas_solver::apply_profile_command(&mut self.core.engine.simplifier, line),
            ),
            cas_solver::ReplCommandInput::Health(line) => {
                match cas_solver::parse_health_command_input(line) {
                    cas_solver::HealthCommandInput::ShowLast => {
                        let report_text = self
                            .core
                            .last_health_report
                            .as_ref()
                            .map(ToString::to_string);
                        let lines = cas_solver::format_health_report_lines(
                            self.core.last_stats.as_ref(),
                            report_text.as_deref(),
                        );
                        reply_output(lines.join("\n"))
                    }
                    cas_solver::HealthCommandInput::SetEnabled { enabled } => {
                        if enabled {
                            self.core.health_enabled = true;
                            reply_output(cas_solver::health_enable_message())
                        } else {
                            self.core.health_enabled = false;
                            reply_output(cas_solver::health_disable_message())
                        }
                    }
                    cas_solver::HealthCommandInput::Clear => {
                        self.core.engine.simplifier.profiler.clear_run();
                        self.core.last_health_report = None;
                        reply_output(cas_solver::health_clear_message())
                    }
                    cas_solver::HealthCommandInput::Status(status) => {
                        if status.list_only {
                            reply_output(crate::health_suite::list_cases())
                        } else {
                            let category_names = crate::health_suite::category_names().join(", ");
                            let category_filter = match cas_solver::resolve_health_category_filter(
                                &status,
                                &category_names,
                                |raw| raw.parse::<crate::health_suite::Category>(),
                            ) {
                                Ok(filter) => filter,
                                Err(message) => return reply_output(message),
                            };

                            let cat_msg = category_filter
                                .as_ref()
                                .map_or("all".to_string(), ToString::to_string);
                            let mut lines =
                                vec![cas_solver::format_health_status_running_message(&cat_msg)];

                            let results = crate::health_suite::run_suite_filtered(
                                &mut self.core.engine.simplifier,
                                category_filter,
                            );
                            let report = crate::health_suite::format_report_filtered(
                                &results,
                                category_filter,
                            );
                            lines.push(report);

                            let (_passed, failed) = crate::health_suite::count_results(&results);
                            if failed > 0 {
                                lines.push(format!(
                                    "\n⚠ {} tests failed. Check Transform rules for churn.",
                                    failed
                                ));
                            }
                            reply_output(lines.join("\n"))
                        }
                    }
                    cas_solver::HealthCommandInput::Invalid => {
                        reply_output(cas_solver::format_health_usage_message(
                            &crate::health_suite::category_names().join(", "),
                        ))
                    }
                }
            }
            cas_solver::ReplCommandInput::Eval(line) => {
                self.handle_eval(line);
                ReplReply::new()
            }
        }
    }

    pub(crate) fn handle_config(&mut self, line: &str) {
        let reply = self.handle_config_core(line);
        self.print_reply(reply);
    }

    fn handle_config_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::parse_config_command_input(line) {
            cas_solver::ConfigCommandInput::List => reply_output(
                cas_solver::format_simplifier_toggle_config(self.config_as_solver_toggle()),
            ),
            cas_solver::ConfigCommandInput::Save => match self.config.save() {
                Ok(_) => reply_output("Configuration saved to cas_config.toml"),
                Err(e) => reply_output(format!("Error saving configuration: {}", e)),
            },
            cas_solver::ConfigCommandInput::Restore => {
                self.config = CasConfig::restore();
                self.sync_config_to_simplifier();
                reply_output("Configuration restored to defaults.")
            }
            cas_solver::ConfigCommandInput::SetRule { rule, enable } => {
                let mut toggles = self.config_as_solver_toggle();
                match cas_solver::set_simplifier_toggle_rule(&mut toggles, &rule, enable) {
                    Ok(()) => {
                        self.set_config_from_solver_toggle(toggles);
                        self.sync_config_to_simplifier();
                        reply_output(format!("Rule '{}' set to {}.", rule, enable))
                    }
                    Err(error) => reply_output(error),
                }
            }
            cas_solver::ConfigCommandInput::MissingRuleArg { action } => {
                reply_output(cas_solver::config_rule_usage_message(&action))
            }
            cas_solver::ConfigCommandInput::InvalidUsage => {
                reply_output(cas_solver::config_usage_message())
            }
            cas_solver::ConfigCommandInput::UnknownSubcommand { subcommand } => {
                reply_output(cas_solver::config_unknown_subcommand_message(&subcommand))
            }
        }
    }
}
