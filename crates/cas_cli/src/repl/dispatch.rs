impl Repl {
    pub fn handle_command(&mut self, line: &str) {
        // Preprocess: Convert function-style commands to command-style
        // simplify(...) -> simplify ...
        // solve(...) -> solve ...
        let line = self.preprocess_function_syntax(line);

        // Check for "help" command
        if line.starts_with("help") {
            self.handle_help(&line);
            return;
        }

        // ========== SESSION ENVIRONMENT COMMANDS ==========

        // "let <name> = <expr>" - assign variable
        if let Some(rest) = line.strip_prefix("let ") {
            self.handle_let_command(rest);
            return;
        }

        // "<name> := <expr>" - alternative assignment syntax
        if let Some(idx) = line.find(":=") {
            let name = line[..idx].trim();
            let expr_str = line[idx + 2..].trim();
            if !name.is_empty() && !expr_str.is_empty() {
                self.handle_assignment(name, expr_str, true); // := is lazy
                return;
            }
        }

        // "vars" - list all variables
        if line == "vars" {
            self.handle_vars_command();
            return;
        }

        // "clear" or "clear <names>" - clear variables
        if line == "clear" || line.starts_with("clear ") {
            self.handle_clear_command(&line);
            return;
        }

        // "reset" - reset entire session
        if line == "reset" {
            self.handle_reset_command();
            return;
        }

        // "reset full" - reset everything including profile cache
        if line == "reset full" {
            self.handle_reset_full_command();
            return;
        }

        // "cache clear" - clear only profile cache
        if line == "cache clear" || line == "cache" {
            self.handle_cache_command(&line);
            return;
        }

        // "semantics" - unified semantic settings (domain, value, branch, inv_trig, const_fold)
        if line == "semantics" || line.starts_with("semantics ") {
            self.handle_semantics(&line);
            return;
        }

        // "mode" - DEPRECATED: redirect to semantics
        if line == "mode" || line.starts_with("mode ") {
            println!("⚠️  The 'mode' command is deprecated.");
            println!("Use 'semantics set inv_trig strict|principal' instead.");
            println!("Run 'semantics' to see current settings.");
            return;
        }

        // "context" - show/switch context mode (auto, standard, solve, integrate)
        if line == "context" || line.starts_with("context ") {
            self.handle_context_command(&line);
            return;
        }

        // "complex" - DEPRECATED: redirect to semantics
        if line == "complex" || line.starts_with("complex ") {
            println!("⚠️  The 'complex' command is deprecated.");
            println!("Use 'semantics set value real|complex' instead.");
            println!("Run 'semantics' to see current settings.");
            return;
        }

        // "steps" - show/switch steps collection mode (on, off, compact)
        if line == "steps" || line.starts_with("steps ") {
            self.handle_steps_command(&line);
            return;
        }

        // "autoexpand" - show/switch auto-expand policy (on, off)
        if line == "autoexpand" || line.starts_with("autoexpand ") {
            self.handle_autoexpand_command(&line);
            return;
        }

        // "budget" - V2.0: Control Conditional branching budget for solve
        if line == "budget" || line.starts_with("budget ") {
            self.handle_budget_command(&line);
            return;
        }

        // "history" or "list" - show session history
        if line == "history" || line == "list" {
            self.handle_history_command();
            return;
        }

        // "show #id" - show a specific session entry
        if let Some(rest) = line.strip_prefix("show ") {
            self.handle_show_command(rest);
            return;
        }

        // "del #id [#id...]" - delete session entries
        if let Some(rest) = line.strip_prefix("del ") {
            self.handle_del_command(rest);
            return;
        }

        // ========== END SESSION ENVIRONMENT COMMANDS ==========

        // Check for "set" command (pipeline options)
        if line.starts_with("set ") {
            self.handle_set_command(&line);
            return;
        }

        // Check for "help" command (duplicate check in original code?)
        if line == "help" {
            self.print_general_help();
            return;
        }

        // Check for "equiv" command
        if line.starts_with("equiv ") {
            self.handle_equiv(&line);
            return;
        }

        // Check for "subst" command
        if line.starts_with("subst ") {
            self.handle_subst(&line);
            return;
        }

        // Check for "solve" command
        if line.starts_with("solve ") {
            self.handle_solve(&line);
            return;
        }

        // Check for "simplify" command
        if line.starts_with("simplify ") {
            self.handle_full_simplify(&line);
            return;
        }

        // Check for "config" command
        if line.starts_with("config ") {
            self.handle_config(&line);
            return;
        }

        // Check for "timeline" command
        if line.starts_with("timeline ") {
            self.handle_timeline(&line);
            return;
        }

        // Check for "visualize" command
        if line.starts_with("visualize ") {
            self.handle_visualize(&line);
            return;
        }

        // Check for "explain" command
        if line.starts_with("explain ") {
            self.handle_explain(&line);
            return;
        }

        // Check for "det" command
        if line.starts_with("det ") {
            self.handle_det(&line);
            return;
        }

        // Check for "transpose" command
        if line.starts_with("transpose ") {
            self.handle_transpose(&line);
            return;
        }

        // Check for "trace" command
        if line.starts_with("trace ") {
            self.handle_trace(&line);
            return;
        }

        // Check for "telescope" command - for proving telescoping identities
        if line.starts_with("telescope ") {
            self.handle_telescope(&line);
            return;
        }

        // Check for "weierstrass" command - Weierstrass substitution (t = tan(x/2))
        if line.starts_with("weierstrass ") {
            self.handle_weierstrass(&line);
            return;
        }

        // Check for "expand_log" command - explicit logarithm expansion
        // MUST come before "expand" check due to prefix matching
        if line.starts_with("expand_log ") || line == "expand_log" {
            self.handle_expand_log(&line);
            return;
        }

        // Check for "expand" command - aggressive expansion/distribution
        if line.starts_with("expand ") {
            self.handle_expand(&line);
            return;
        }

        // Check for "rationalize" command - rationalize denominators with surds
        if line.starts_with("rationalize ") {
            self.handle_rationalize(&line);
            return;
        }

        // Check for "limit" command - compute limits at infinity
        if line.starts_with("limit ") {
            self.handle_limit(&line);
            return;
        }

        // Check for "profile" commands
        if line.starts_with("profile") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 1 {
                // Just "profile" - show report
                println!("{}", self.engine.simplifier.profiler.report());
            } else {
                match parts[1] {
                    "enable" => {
                        self.engine.simplifier.profiler.enable();
                        println!("Profiler enabled.");
                    }
                    "disable" => {
                        self.engine.simplifier.profiler.disable();
                        println!("Profiler disabled.");
                    }
                    "clear" => {
                        self.engine.simplifier.profiler.clear();
                        println!("Profiler statistics cleared.");
                    }
                    _ => println!("Usage: profile [enable|disable|clear]"),
                }
            }
            return;
        }

        // Check for "health" commands
        if line.starts_with("health") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 1 {
                // Just "health" - show last report
                // First show any cycles detected
                if let Some(ref stats) = self.last_stats {
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

                if let Some(ref report) = self.last_health_report {
                    println!("{}", report);
                } else {
                    println!("No health report available.");
                    println!("Run a simplification first (health is captured when debug mode or health mode is on).");
                    println!("Enable with: health on");
                }
            } else {
                match parts[1] {
                    "on" | "enable" => {
                        self.health_enabled = true;
                        println!("Health tracking ENABLED (metrics captured after each simplify)");
                    }
                    "off" | "disable" => {
                        self.health_enabled = false;
                        println!("Health tracking DISABLED");
                    }
                    "reset" | "clear" => {
                        self.engine.simplifier.profiler.clear_run();
                        self.last_health_report = None;
                        println!("Health statistics cleared.");
                    }
                    "status" => {
                        // Parse options: status [--list | --category <cat>]
                        let opts: Vec<&str> = parts.iter().skip(2).copied().collect();

                        if opts.contains(&"--list") || opts.contains(&"-l") {
                            // List available cases
                            println!("{}", crate::health_suite::list_cases());
                            return;
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
                                            return;
                                        }
                                    }
                                }
                            } else {
                                println!("Error: --category requires an argument");
                                println!(
                                    "Available categories: {}",
                                    crate::health_suite::category_names().join(", ")
                                );
                                return;
                            }
                        } else {
                            None // Run all
                        };

                        // Run the health status suite
                        let cat_msg = category_filter.map_or("all".to_string(), |c| c.to_string());
                        println!("Running health status suite [category={}]...\n", cat_msg);

                        let results = crate::health_suite::run_suite_filtered(
                            &mut self.engine.simplifier,
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
            return;
        }

        self.handle_eval(&line);
    }

    fn handle_config(&mut self, line: &str) {
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
