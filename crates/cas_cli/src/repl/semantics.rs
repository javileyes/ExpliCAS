impl Repl {
    fn parse_semantics_set(&mut self, args: &[&str]) {
        if args.is_empty() {
            println!("Usage: semantics set <axis> <value>");
            println!("  or:  semantics set <axis>=<value> ...");
            return;
        }

        let mut i = 0;
        while i < args.len() {
            let arg = args[i];

            // Check for key=value format
            if let Some((key, value)) = arg.split_once('=') {
                if !self.set_semantic_axis(key, value) {
                    return;
                }
                i += 1;
            } else {
                // key value format
                if i + 1 >= args.len() {
                    println!("ERROR: Missing value for axis '{}'", arg);
                    return;
                }

                // Special case: "solve check on|off" is a 3-part axis
                if arg == "solve" && args.get(i + 1) == Some(&"check") && i + 2 < args.len() {
                    let on_off = args[i + 2];
                    match on_off {
                        "on" => {
                            self.state.options.check_solutions = true;
                            println!("Solve check: ON (solutions will be verified)");
                        }
                        "off" => {
                            self.state.options.check_solutions = false;
                            println!("Solve check: OFF");
                        }
                        _ => {
                            println!("ERROR: Invalid value '{}' for 'solve check'", on_off);
                            println!("Allowed: on, off");
                            return;
                        }
                    }
                    i += 3;
                    continue;
                }

                let value = args[i + 1];
                if !self.set_semantic_axis(arg, value) {
                    return;
                }
                i += 2;
            }
        }

        self.sync_config_to_simplifier();
        self.print_semantics();
    }

    fn set_semantic_axis(&mut self, axis: &str, value: &str) -> bool {
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        match axis {
            "domain" => match value {
                "strict" => {
                    self.simplify_options.domain = DomainMode::Strict;
                    self.state.options.domain_mode = DomainMode::Strict;
                }
                "generic" => {
                    self.simplify_options.domain = DomainMode::Generic;
                    self.state.options.domain_mode = DomainMode::Generic;
                }
                "assume" => {
                    self.simplify_options.domain = DomainMode::Assume;
                    self.state.options.domain_mode = DomainMode::Assume;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'domain'", value);
                    println!("Allowed: strict, generic, assume");
                    return false;
                }
            },
            "value" => match value {
                "real" => {
                    self.simplify_options.value_domain = ValueDomain::RealOnly;
                    self.state.options.value_domain = ValueDomain::RealOnly;
                }
                "complex" => {
                    self.simplify_options.value_domain = ValueDomain::ComplexEnabled;
                    self.state.options.value_domain = ValueDomain::ComplexEnabled;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'value'", value);
                    println!("Allowed: real, complex");
                    return false;
                }
            },
            "branch" => match value {
                "principal" => {
                    self.simplify_options.branch = BranchPolicy::Principal;
                    self.state.options.branch = BranchPolicy::Principal;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'branch'", value);
                    println!("Allowed: principal");
                    return false;
                }
            },
            "inv_trig" => match value {
                "strict" => {
                    self.simplify_options.inv_trig = InverseTrigPolicy::Strict;
                    self.state.options.inv_trig = InverseTrigPolicy::Strict;
                }
                "principal" => {
                    self.simplify_options.inv_trig = InverseTrigPolicy::PrincipalValue;
                    self.state.options.inv_trig = InverseTrigPolicy::PrincipalValue;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'inv_trig'", value);
                    println!("Allowed: strict, principal");
                    return false;
                }
            },
            "const_fold" => {
                use cas_engine::const_fold::ConstFoldMode;
                match value {
                    "off" => {
                        self.state.options.const_fold = ConstFoldMode::Off;
                    }
                    "safe" => {
                        self.state.options.const_fold = ConstFoldMode::Safe;
                    }
                    _ => {
                        println!("ERROR: Invalid value '{}' for axis 'const_fold'", value);
                        println!("Allowed: off, safe");
                        return false;
                    }
                }
            }
            "assumptions" => match value {
                "off" => {
                    self.state.options.assumption_reporting = cas_engine::AssumptionReporting::Off;
                    self.simplify_options.assumption_reporting =
                        cas_engine::AssumptionReporting::Off;
                }
                "summary" => {
                    self.state.options.assumption_reporting =
                        cas_engine::AssumptionReporting::Summary;
                    self.simplify_options.assumption_reporting =
                        cas_engine::AssumptionReporting::Summary;
                }
                "trace" => {
                    self.state.options.assumption_reporting =
                        cas_engine::AssumptionReporting::Trace;
                    self.simplify_options.assumption_reporting =
                        cas_engine::AssumptionReporting::Trace;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'assumptions'", value);
                    println!("Allowed: off, summary, trace");
                    return false;
                }
            },
            "assume_scope" => match value {
                "real" => {
                    self.simplify_options.assume_scope = cas_engine::AssumeScope::Real;
                    self.state.options.assume_scope = cas_engine::AssumeScope::Real;
                }
                "wildcard" => {
                    self.simplify_options.assume_scope = cas_engine::AssumeScope::Wildcard;
                    self.state.options.assume_scope = cas_engine::AssumeScope::Wildcard;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'assume_scope'", value);
                    println!("Allowed: real, wildcard");
                    return false;
                }
            },
            "hints" => match value {
                "on" => {
                    self.state.options.hints_enabled = true;
                }
                "off" => {
                    self.state.options.hints_enabled = false;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'hints'", value);
                    println!("Allowed: on, off");
                    return false;
                }
            },
            "solve" => match value {
                "check" => {
                    // "solve check" is special: toggle without secondary value
                    println!("ERROR: Use 'semantics set solve check on' or 'semantics set solve check off'");
                    return false;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'solve'", value);
                    println!("Allowed: 'check on', 'check off'");
                    return false;
                }
            },
            "requires" => match value {
                "essential" => {
                    self.state.options.requires_display =
                        cas_engine::implicit_domain::RequiresDisplayLevel::Essential;
                    println!("Requires display: essential (hide if witness survives)");
                }
                "all" => {
                    self.state.options.requires_display =
                        cas_engine::implicit_domain::RequiresDisplayLevel::All;
                    println!("Requires display: all (show everything)");
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'requires'", value);
                    println!("Allowed: essential, all");
                    return false;
                }
            },
            _ => {
                println!("ERROR: Unknown axis '{}'", axis);
                println!("Valid axes: domain, value, branch, inv_trig, const_fold, assumptions, assume_scope, hints, solve, requires");
                return false;
            }
        }
        true
    }

    /// Handle "context" command - show or switch context mode
    fn handle_context_command(&mut self, line: &str) {
        use cas_engine::options::ContextMode;

        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "context" - show current context
                let ctx_str = match self.state.options.context_mode {
                    ContextMode::Auto => "auto",
                    ContextMode::Standard => "standard",
                    ContextMode::Solve => "solve",
                    ContextMode::IntegratePrep => "integrate",
                };
                println!("Current context: {}", ctx_str);
                println!("  (use 'context auto|standard|solve|integrate' to change)");
            }
            Some(&"auto") => {
                self.state.options.context_mode = ContextMode::Auto;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                println!("Context: auto (infers from expression)");
            }
            Some(&"standard") => {
                self.state.options.context_mode = ContextMode::Standard;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                println!("Context: standard (safe simplification only)");
            }
            Some(&"solve") => {
                self.state.options.context_mode = ContextMode::Solve;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                println!("Context: solve (preserves solver-friendly forms)");
            }
            Some(&"integrate") => {
                self.state.options.context_mode = ContextMode::IntegratePrep;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                println!("Context: integrate-prep");
                println!("  ⚠️ Enables transforms for integration (telescoping, product→sum)");
            }
            Some(other) => {
                println!("Unknown context: '{}'", other);
                println!("Usage: context [auto | standard | solve | integrate]");
            }
        }
    }

    /// Handle "steps" command - show or switch steps collection mode AND display verbosity
    /// Collection: on, off, compact (controls StepsMode in engine)
    /// Display: verbose, succinct, normal, none (controls Verbosity in CLI)
    fn handle_steps_command(&mut self, line: &str) {
        use cas_engine::options::StepsMode;

        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "steps" - show current mode
                let mode_str = match self.state.options.steps_mode {
                    StepsMode::On => "on",
                    StepsMode::Off => "off",
                    StepsMode::Compact => "compact",
                };
                let verbosity_str = match self.verbosity {
                    Verbosity::None => "none",
                    Verbosity::Succinct => "succinct",
                    Verbosity::Normal => "normal",
                    Verbosity::Verbose => "verbose",
                };
                println!("Steps collection: {}", mode_str);
                println!("Steps display: {}", verbosity_str);
                println!("  (use 'steps on|off|compact' for collection)");
                println!("  (use 'steps verbose|succinct|normal|none' for display)");
            }
            // Collection modes (StepsMode)
            Some(&"on") => {
                self.state.options.steps_mode = StepsMode::On;
                self.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Normal;
                println!("Steps: on (full collection, normal display)");
            }
            Some(&"off") => {
                self.state.options.steps_mode = StepsMode::Off;
                self.engine.simplifier.set_steps_mode(StepsMode::Off);
                self.verbosity = Verbosity::None;
                println!("Steps: off");
                println!("  ⚡ Steps disabled (faster). Warnings still enabled.");
            }
            Some(&"compact") => {
                self.state.options.steps_mode = StepsMode::Compact;
                self.engine.simplifier.set_steps_mode(StepsMode::Compact);
                println!("Steps: compact (no before/after snapshots)");
            }
            // Display modes (Verbosity)
            Some(&"verbose") => {
                self.state.options.steps_mode = StepsMode::On;
                self.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Verbose;
                println!("Steps: verbose (all rules, full detail)");
            }
            Some(&"succinct") => {
                self.state.options.steps_mode = StepsMode::On;
                self.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Succinct;
                println!("Steps: succinct (compact 1-line per step)");
            }
            Some(&"normal") => {
                self.state.options.steps_mode = StepsMode::On;
                self.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Normal;
                println!("Steps: normal (default display)");
            }
            Some(&"none") => {
                self.verbosity = Verbosity::None;
                println!("Steps display: none (collection still active)");
                println!("  Use 'steps off' to also disable collection.");
            }
            Some(other) => {
                println!("Unknown steps mode: '{}'", other);
                println!("Usage: steps [on | off | compact | verbose | succinct | normal | none]");
                println!("  Collection modes:");
                println!("    on      - Full steps with snapshots (default)");
                println!("    off     - No steps (fastest, warnings preserved)");
                println!("    compact - Minimal steps (no snapshots)");
                println!("  Display modes:");
                println!("    verbose - Show all rules, full detail");
                println!("    succinct- Compact 1-line per step");
                println!("    normal  - Standard display (default)");
                println!("    none    - Hide steps output (collection still active)");
            }
        }
    }

    /// Handle "autoexpand" command - show or switch auto-expand policy
    fn handle_autoexpand_command(&mut self, line: &str) {
        use cas_engine::phase::ExpandPolicy;

        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "autoexpand" - show current mode
                let policy_str = match self.state.options.expand_policy {
                    ExpandPolicy::Off => "off",
                    ExpandPolicy::Auto => "on",
                };
                println!("Auto-expand: {}", policy_str);
                let budget = &self.state.options.expand_budget;
                println!(
                    "  Budget: pow<={}, base_terms<={}, gen_terms<={}, vars<={}",
                    budget.max_pow_exp,
                    budget.max_base_terms,
                    budget.max_generated_terms,
                    budget.max_vars
                );
                println!("  (use 'autoexpand on|off' to change)");
            }
            Some(&"on") => {
                self.state.options.expand_policy = ExpandPolicy::Auto;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                let budget = &self.state.options.expand_budget;
                println!("Auto-expand: on");
                println!(
                    "  Budget: pow<={}, base_terms<={}, gen_terms<={}, vars<={}",
                    budget.max_pow_exp,
                    budget.max_base_terms,
                    budget.max_generated_terms,
                    budget.max_vars
                );
                println!("  ⚠️ Expands small (sum)^n patterns automatically.");
            }
            Some(&"off") => {
                self.state.options.expand_policy = ExpandPolicy::Off;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                println!("Auto-expand: off");
                println!("  Polynomial expansions require explicit expand().");
            }
            Some(other) => {
                println!("Unknown autoexpand mode: '{}'", other);
                println!("Usage: autoexpand [on | off]");
                println!("  on  - Auto-expand cheap polynomial powers");
                println!("  off - Only expand when explicitly requested (default)");
            }
        }
    }
}
