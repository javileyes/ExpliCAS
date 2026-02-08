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
                // Extract panic message if possible
                let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };

                // Generate short error_id for correlation (hash of timestamp + message)
                let error_id = Self::generate_error_id(&panic_msg);

                // Log to stderr if EXPLICAS_PANIC_REPORT is set (for debugging)
                // This respects the repl IO lint by only logging when explicitly enabled
                if std::env::var("EXPLICAS_PANIC_REPORT").is_ok() {
                    // Use init.rs print_reply with Error variant which goes to stderr
                    let version = env!("CARGO_PKG_VERSION");
                    let log_msg = format!(
                        "[PANIC_REPORT] id={} version={} command={:?} panic={}",
                        error_id, version, line, panic_msg
                    );
                    // Log via ReplMsg::Debug to get it to output
                    self.print_reply(vec![ReplMsg::Debug(log_msg)]);
                }

                // Return user-friendly error that doesn't kill the session
                self.print_reply(vec![ReplMsg::Error(format!(
                    "Internal error (id: {}): {}\n\n\
                     The session is still active. You can continue working.\n\
                     Please report this issue with the error id if it persists.",
                    error_id, panic_msg
                ))]);
            }
        }
    }

    /// Generate a short error ID for crash correlation.
    /// Uses a simple hash of timestamp + message for uniqueness.
    fn generate_error_id(msg: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};

        let mut hasher = DefaultHasher::new();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        timestamp.hash(&mut hasher);
        msg.hash(&mut hasher);
        let hash = hasher.finish();

        // Return short hex (6 chars = 24 bits = ~16M unique IDs)
        format!("{:06X}", (hash & 0xFFFFFF) as u32)
    }

    /// Core command dispatch - returns structured messages, no I/O.
    /// This is the heart of ReplCore logic.
    pub fn handle_command_core(&mut self, line: &str) -> ReplReply {
        let reply = ReplReply::new();

        // Preprocess: Convert function-style commands to command-style
        // simplify(...) -> simplify ...
        // solve(...) -> solve ...
        let line = self.preprocess_function_syntax(line);

        // Check for "help" command
        if line.starts_with("help") {
            self.handle_help(&line);
            return reply; // TODO: migrate handle_help to return ReplReply
        }

        // ========== SESSION ENVIRONMENT COMMANDS ==========

        // "let <name> = <expr>" - assign variable
        if let Some(rest) = line.strip_prefix("let ") {
            self.handle_let_command(rest);
            return reply;
        }

        // "<name> := <expr>" - alternative assignment syntax
        if let Some(idx) = line.find(":=") {
            let name = line[..idx].trim();
            let expr_str = line[idx + 2..].trim();
            if !name.is_empty() && !expr_str.is_empty() {
                self.handle_assignment(name, expr_str, true); // := is lazy
                return reply;
            }
        }

        // "vars" - list all variables
        if line == "vars" {
            self.handle_vars_command();
            return reply;
        }

        // "clear" or "clear <names>" - clear variables
        if line == "clear" || line.starts_with("clear ") {
            self.handle_clear_command(&line);
            return reply;
        }

        // "reset" - reset entire session
        if line == "reset" {
            self.handle_reset_command();
            return reply;
        }

        // "reset full" - reset everything including profile cache
        if line == "reset full" {
            self.handle_reset_full_command();
            return reply;
        }

        // "cache clear" - clear only profile cache
        if line == "cache clear" || line == "cache" {
            self.handle_cache_command(&line);
            return reply;
        }

        // "semantics" - unified semantic settings (domain, value, branch, inv_trig, const_fold)
        if line == "semantics" || line.starts_with("semantics ") {
            self.handle_semantics(&line);
            return reply;
        }

        // "context" - show/switch context mode (auto, standard, solve, integrate)
        if line == "context" || line.starts_with("context ") {
            self.handle_context_command(&line);
            return reply;
        }

        // "steps" - show/switch steps collection mode (on, off, compact)
        if line == "steps" || line.starts_with("steps ") {
            self.handle_steps_command(&line);
            return reply;
        }

        // "autoexpand" - show/switch auto-expand policy (on, off)
        if line == "autoexpand" || line.starts_with("autoexpand ") {
            self.handle_autoexpand_command(&line);
            return reply;
        }

        // "budget" - V2.0: Control Conditional branching budget for solve
        if line == "budget" || line.starts_with("budget ") {
            self.handle_budget_command(&line);
            return reply;
        }

        // "history" or "list" - show session history
        if line == "history" || line == "list" {
            self.handle_history_command();
            return reply;
        }

        // "show #id" - show a specific session entry
        if let Some(rest) = line.strip_prefix("show ") {
            self.handle_show_command(rest);
            return reply;
        }

        // "del #id [#id...]" - delete session entries
        if let Some(rest) = line.strip_prefix("del ") {
            self.handle_del_command(rest);
            return reply;
        }

        // ========== END SESSION ENVIRONMENT COMMANDS ==========

        // Check for "set" command (pipeline options)
        if line.starts_with("set ") {
            self.handle_set_command(&line);
            return reply;
        }

        // Check for "help" command (duplicate check in original code?)
        if line == "help" {
            self.print_general_help();
            return reply;
        }

        // Check for "equiv" command
        if line.starts_with("equiv ") {
            self.handle_equiv(&line);
            return reply;
        }

        // Check for "subst" command
        if line.starts_with("subst ") {
            self.handle_subst(&line);
            return reply;
        }

        // Check for "solve_system" command (must be before "solve" to avoid prefix collision)
        if line.starts_with("solve_system") {
            self.handle_solve_system(&line);
            return reply;
        }

        // Check for "solve" command
        if line.starts_with("solve ") {
            self.handle_solve(&line);
            return reply;
        }

        // Check for "simplify" command
        if line.starts_with("simplify ") {
            self.handle_full_simplify(&line);
            return reply;
        }

        // Check for "config" command
        if line.starts_with("config ") {
            self.handle_config(&line);
            return reply;
        }

        // Check for "timeline" command
        if line.starts_with("timeline ") {
            self.handle_timeline(&line);
            return reply;
        }

        // Check for "visualize" command
        if line.starts_with("visualize ") {
            self.handle_visualize(&line);
            return reply;
        }

        // Check for "explain" command
        if line.starts_with("explain ") {
            self.handle_explain(&line);
            return reply;
        }

        // Check for "det" command
        if line.starts_with("det ") {
            self.handle_det(&line);
            return reply;
        }

        // Check for "transpose" command
        if line.starts_with("transpose ") {
            self.handle_transpose(&line);
            return reply;
        }

        // Check for "trace" command
        if line.starts_with("trace ") {
            self.handle_trace(&line);
            return reply;
        }

        // Check for "telescope" command - for proving telescoping identities
        if line.starts_with("telescope ") {
            self.handle_telescope(&line);
            return reply;
        }

        // Check for "weierstrass" command - Weierstrass substitution (t = tan(x/2))
        if line.starts_with("weierstrass ") {
            self.handle_weierstrass(&line);
            return reply;
        }

        // Check for "expand_log" command - explicit logarithm expansion
        // MUST come before "expand" check due to prefix matching
        if line.starts_with("expand_log ") || line == "expand_log" {
            self.handle_expand_log(&line);
            return reply;
        }

        // Check for "expand" command - aggressive expansion/distribution
        if line.starts_with("expand ") {
            self.handle_expand(&line);
            return reply;
        }

        // Check for "rationalize" command - rationalize denominators with surds
        if line.starts_with("rationalize ") {
            self.handle_rationalize(&line);
            return reply;
        }

        // Check for "limit" command - compute limits at infinity
        if line.starts_with("limit ") {
            self.handle_limit(&line);
            return reply;
        }

        // Check for "profile" commands
        if line.starts_with("profile") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 1 {
                // Just "profile" - show report
                return reply_output(self.core.engine.simplifier.profiler.report());
            } else {
                match parts[1] {
                    "enable" => {
                        self.core.engine.simplifier.profiler.enable();
                        return reply_output("Profiler enabled.");
                    }
                    "disable" => {
                        self.core.engine.simplifier.profiler.disable();
                        return reply_output("Profiler disabled.");
                    }
                    "clear" => {
                        self.core.engine.simplifier.profiler.clear();
                        return reply_output("Profiler statistics cleared.");
                    }
                    _ => return reply_output("Usage: profile [enable|disable|clear]"),
                }
            }
        }

        // Check for "health" commands
        if line.starts_with("health") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 1 {
                // Just "health" - show last report
                let mut lines: Vec<String> = Vec::new();

                // First show any cycles detected
                if let Some(ref stats) = self.core.last_stats {
                    let cycles: Vec<_> = [
                        (&stats.core.cycle, "Core"),
                        (&stats.transform.cycle, "Transform"),
                        (&stats.rationalize.cycle, "Rationalize"),
                        (&stats.post_cleanup.cycle, "PostCleanup"),
                    ]
                    .iter()
                    .filter_map(|(c, name)| c.as_ref().map(|info| (*name, info)))
                    .collect();

                    for (phase_name, cycle) in &cycles {
                        lines.push(format!(
                            "⚠ Cycle detected in {}: period={} at rewrite={} (stopped early)",
                            phase_name, cycle.period, cycle.at_step
                        ));
                    }
                    if !cycles.is_empty() {
                        lines.push(String::new());
                    }
                }

                if let Some(ref report) = self.core.last_health_report {
                    lines.push(report.to_string());
                } else {
                    lines.push("No health report available.".to_string());
                    lines.push("Run a simplification first (health is captured when debug mode or health mode is on).".to_string());
                    lines.push("Enable with: health on".to_string());
                }
                return reply_output(lines.join("\n"));
            } else {
                match parts[1] {
                    "on" | "enable" => {
                        self.core.health_enabled = true;
                        return reply_output(
                            "Health tracking ENABLED (metrics captured after each simplify)",
                        );
                    }
                    "off" | "disable" => {
                        self.core.health_enabled = false;
                        return reply_output("Health tracking DISABLED");
                    }
                    "reset" | "clear" => {
                        self.core.engine.simplifier.profiler.clear_run();
                        self.core.last_health_report = None;
                        return reply_output("Health statistics cleared.");
                    }
                    "status" => {
                        // Parse options: status [--list | --category <cat>]
                        let opts: Vec<&str> = parts.iter().skip(2).copied().collect();

                        if opts.contains(&"--list") || opts.contains(&"-l") {
                            // List available cases
                            return reply_output(crate::health_suite::list_cases());
                        }

                        // Check for --category
                        let category_filter = if let Some(idx) =
                            opts.iter().position(|&x| x == "--category" || x == "-c")
                        {
                            if let Some(cat_str) = opts.get(idx + 1) {
                                if *cat_str == "all" {
                                    None
                                } else {
                                    match cat_str.parse::<crate::health_suite::Category>() {
                                        Ok(cat) => Some(cat),
                                        Err(e) => {
                                            return reply_output(format!(
                                                "Error: {}\nAvailable categories: {}",
                                                e,
                                                crate::health_suite::category_names().join(", ")
                                            ));
                                        }
                                    }
                                }
                            } else {
                                return reply_output(format!(
                                    "Error: --category requires an argument\nAvailable categories: {}",
                                    crate::health_suite::category_names().join(", ")
                                ));
                            }
                        } else {
                            None // Run all
                        };

                        // Run the health status suite
                        let cat_msg = category_filter.map_or("all".to_string(), |c| c.to_string());
                        let mut lines = vec![format!(
                            "Running health status suite [category={}]...\n",
                            cat_msg
                        )];

                        let results = crate::health_suite::run_suite_filtered(
                            &mut self.core.engine.simplifier,
                            category_filter,
                        );
                        let report =
                            crate::health_suite::format_report_filtered(&results, category_filter);
                        lines.push(report);

                        let (_passed, failed) = crate::health_suite::count_results(&results);
                        if failed > 0 {
                            lines.push(format!(
                                "\n⚠ {} tests failed. Check Transform rules for churn.",
                                failed
                            ));
                        }
                        return reply_output(lines.join("\n"));
                    }
                    _ => {
                        return reply_output(format!(
                            "Usage: health [on|off|reset|status]\n\
                             \n\
                                   health               Show last health report\n\
                                   health on            Enable health tracking\n\
                                   health off           Disable health tracking\n\
                                   health reset         Clear health statistics\n\
                                   health status        Run diagnostic test suite\n\
                                   health status --list List available test cases\n\
                                   health status --category <cat>  Run only category\n\
                                                        Categories: {}",
                            crate::health_suite::category_names().join(", ")
                        ));
                    }
                }
            }
        }

        self.handle_eval(&line);
        reply
    }

    pub(crate) fn handle_config(&mut self, line: &str) {
        let reply = self.handle_config_core(line);
        self.print_reply(reply);
    }

    fn handle_config_core(&mut self, line: &str) -> ReplReply {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return reply_output("Usage: config <list|enable|disable|save|restore> [rule]");
        }

        match parts[1] {
            "list" => {
                let config_str = format!(
                    "Current Configuration:\n\
                       distribute: {}\n\
                       expand_binomials: {}\n\
                       distribute_constants: {}\n\
                       factor_difference_squares: {}\n\
                       root_denesting: {}\n\
                       trig_double_angle: {}\n\
                       trig_angle_sum: {}\n\
                       log_split_exponents: {}\n\
                       rationalize_denominator: {}\n\
                       canonicalize_trig_square: {}\n\
                       auto_factor: {}",
                    self.config.distribute,
                    self.config.expand_binomials,
                    self.config.distribute_constants,
                    self.config.factor_difference_squares,
                    self.config.root_denesting,
                    self.config.trig_double_angle,
                    self.config.trig_angle_sum,
                    self.config.log_split_exponents,
                    self.config.rationalize_denominator,
                    self.config.canonicalize_trig_square,
                    self.config.auto_factor
                );
                reply_output(config_str)
            }
            "save" => match self.config.save() {
                Ok(_) => reply_output("Configuration saved to cas_config.toml"),
                Err(e) => reply_output(format!("Error saving configuration: {}", e)),
            },
            "restore" => {
                self.config = CasConfig::restore();
                self.sync_config_to_simplifier();
                reply_output("Configuration restored to defaults.")
            }
            "enable" | "disable" => {
                if parts.len() < 3 {
                    return reply_output(format!("Usage: config {} <rule>", parts[1]));
                }
                let rule = parts[2];
                let enable = parts[1] == "enable";

                match rule {
                    "distribute" => self.config.distribute = enable,
                    "expand_binomials" => self.config.expand_binomials = enable,
                    "distribute_constants" => self.config.distribute_constants = enable,
                    "factor_difference_squares" => self.config.factor_difference_squares = enable,
                    "root_denesting" => self.config.root_denesting = enable,
                    "trig_double_angle" => self.config.trig_double_angle = enable,
                    "trig_angle_sum" => self.config.trig_angle_sum = enable,
                    "log_split_exponents" => self.config.log_split_exponents = enable,
                    "rationalize_denominator" => self.config.rationalize_denominator = enable,
                    "canonicalize_trig_square" => self.config.canonicalize_trig_square = enable,
                    "auto_factor" => self.config.auto_factor = enable,
                    _ => return reply_output(format!("Unknown rule: {}", rule)),
                }

                self.sync_config_to_simplifier();
                reply_output(format!("Rule '{}' set to {}.", rule, enable))
            }
            _ => reply_output(format!("Unknown config command: {}", parts[1])),
        }
    }
}
