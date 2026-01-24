use super::output::ReplReply;
use super::*;

impl Repl {
    /// Main command dispatch - calls core and prints result.
    pub fn handle_command(&mut self, line: &str) {
        let reply = self.handle_command_core(line);
        self.print_reply(reply);
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

        // "mode" - DEPRECATED: redirect to semantics
        if line == "mode" || line.starts_with("mode ") {
            println!("⚠️  The 'mode' command is deprecated.");
            println!("Use 'semantics set inv_trig strict|principal' instead.");
            println!("Run 'semantics' to see current settings.");
            return reply;
        }

        // "context" - show/switch context mode (auto, standard, solve, integrate)
        if line == "context" || line.starts_with("context ") {
            self.handle_context_command(&line);
            return reply;
        }

        // "complex" - DEPRECATED: redirect to semantics
        if line == "complex" || line.starts_with("complex ") {
            println!("⚠️  The 'complex' command is deprecated.");
            println!("Use 'semantics set value real|complex' instead.");
            println!("Run 'semantics' to see current settings.");
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
                println!("{}", self.core.engine.simplifier.profiler.report());
            } else {
                match parts[1] {
                    "enable" => {
                        self.core.engine.simplifier.profiler.enable();
                        println!("Profiler enabled.");
                    }
                    "disable" => {
                        self.core.engine.simplifier.profiler.disable();
                        println!("Profiler disabled.");
                    }
                    "clear" => {
                        self.core.engine.simplifier.profiler.clear();
                        println!("Profiler statistics cleared.");
                    }
                    _ => println!("Usage: profile [enable|disable|clear]"),
                }
            }
            return reply;
        }

        // Check for "health" commands
        if line.starts_with("health") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 1 {
                // Just "health" - show last report
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
                        println!(
                            "⚠ Cycle detected in {}: period={} at rewrite={} (stopped early)",
                            phase_name, cycle.period, cycle.at_step
                        );
                    }
                    if !cycles.is_empty() {
                        println!();
                    }
                }

                if let Some(ref report) = self.core.last_health_report {
                    println!("{}", report);
                } else {
                    println!("No health report available.");
                    println!("Run a simplification first (health is captured when debug mode or health mode is on).");
                    println!("Enable with: health on");
                }
            } else {
                match parts[1] {
                    "on" | "enable" => {
                        self.core.health_enabled = true;
                        println!("Health tracking ENABLED (metrics captured after each simplify)");
                    }
                    "off" | "disable" => {
                        self.core.health_enabled = false;
                        println!("Health tracking DISABLED");
                    }
                    "reset" | "clear" => {
                        self.core.engine.simplifier.profiler.clear_run();
                        self.core.last_health_report = None;
                        println!("Health statistics cleared.");
                    }
                    "status" => {
                        // Parse options: status [--list | --category <cat>]
                        let opts: Vec<&str> = parts.iter().skip(2).copied().collect();

                        if opts.contains(&"--list") || opts.contains(&"-l") {
                            // List available cases
                            println!("{}", crate::health_suite::list_cases());
                            return reply;
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
                                            println!("Error: {}", e);
                                            println!(
                                                "Available categories: {}",
                                                crate::health_suite::category_names().join(", ")
                                            );
                                            return reply;
                                        }
                                    }
                                }
                            } else {
                                println!("Error: --category requires an argument");
                                println!(
                                    "Available categories: {}",
                                    crate::health_suite::category_names().join(", ")
                                );
                                return reply;
                            }
                        } else {
                            None // Run all
                        };

                        // Run the health status suite
                        let cat_msg = category_filter.map_or("all".to_string(), |c| c.to_string());
                        println!("Running health status suite [category={}]...\n", cat_msg);

                        let results = crate::health_suite::run_suite_filtered(
                            &mut self.core.engine.simplifier,
                            category_filter,
                        );
                        let report =
                            crate::health_suite::format_report_filtered(&results, category_filter);
                        println!("{}", report);

                        let (_passed, failed) = crate::health_suite::count_results(&results);
                        if failed > 0 {
                            println!(
                                "\n⚠ {} tests failed. Check Transform rules for churn.",
                                failed
                            );
                        }
                    }
                    _ => {
                        println!("Usage: health [on|off|reset|status]");
                        println!("       health               Show last health report");
                        println!("       health on            Enable health tracking");
                        println!("       health off           Disable health tracking");
                        println!("       health reset         Clear health statistics");
                        println!("       health status        Run diagnostic test suite");
                        println!("       health status --list List available test cases");
                        println!("       health status --category <cat>  Run only category");
                        println!(
                            "                            Categories: {}",
                            crate::health_suite::category_names().join(", ")
                        );
                    }
                }
            }
            return reply;
        }

        self.handle_eval(&line);
        reply
    }

    pub(crate) fn handle_config(&mut self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            println!("Usage: config <list|enable|disable|save|restore> [rule]");
            return;
        }

        match parts[1] {
            "list" => {
                println!("Current Configuration:");
                println!("  distribute: {}", self.config.distribute);
                println!("  expand_binomials: {}", self.config.expand_binomials);
                println!(
                    "  distribute_constants: {}",
                    self.config.distribute_constants
                );
                println!(
                    "  factor_difference_squares: {}",
                    self.config.factor_difference_squares
                );
                println!("  root_denesting: {}", self.config.root_denesting);
                println!("  trig_double_angle: {}", self.config.trig_double_angle);
                println!("  trig_angle_sum: {}", self.config.trig_angle_sum);
                println!("  log_split_exponents: {}", self.config.log_split_exponents);
                println!(
                    "  rationalize_denominator: {}",
                    self.config.rationalize_denominator
                );
                println!(
                    "  canonicalize_trig_square: {}",
                    self.config.canonicalize_trig_square
                );
                println!("  auto_factor: {}", self.config.auto_factor);
            }
            "save" => match self.config.save() {
                Ok(_) => println!("Configuration saved to cas_config.toml"),
                Err(e) => println!("Error saving configuration: {}", e),
            },
            "restore" => {
                self.config = CasConfig::restore();
                self.sync_config_to_simplifier();
                println!("Configuration restored to defaults.");
            }
            "enable" | "disable" => {
                if parts.len() < 3 {
                    println!("Usage: config {} <rule>", parts[1]);
                    return;
                }
                let rule = parts[2];
                let enable = parts[1] == "enable";

                let mut changed = true;
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
                    _ => {
                        println!("Unknown rule: {}", rule);
                        changed = false;
                    }
                }

                if changed {
                    self.sync_config_to_simplifier();
                    println!("Rule '{}' set to {}.", rule, enable);
                }
            }
            _ => println!("Unknown config command: {}", parts[1]),
        }
    }
}
