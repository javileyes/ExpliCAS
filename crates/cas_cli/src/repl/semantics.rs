use super::*;

impl Repl {
    pub(crate) fn parse_semantics_set(&mut self, args: &[&str]) {
        let reply = self.parse_semantics_set_core(args);
        self.print_reply(reply);
    }

    fn parse_semantics_set_core(&mut self, args: &[&str]) -> ReplReply {
        if args.is_empty() {
            return reply_output(
                "Usage: semantics set <axis> <value>\n\
                   or:  semantics set <axis>=<value> ...",
            );
        }

        let mut i = 0;
        while i < args.len() {
            let arg = args[i];

            // Check for key=value format
            if let Some((key, value)) = arg.split_once('=') {
                if let Some(err) = self.set_semantic_axis(key, value) {
                    return reply_output(err);
                }
                i += 1;
            } else {
                // key value format
                if i + 1 >= args.len() {
                    return reply_output(format!("ERROR: Missing value for axis '{}'", arg));
                }

                // Special case: "solve check on|off" is a 3-part axis
                if arg == "solve" && args.get(i + 1) == Some(&"check") && i + 2 < args.len() {
                    let on_off = args[i + 2];
                    match on_off {
                        "on" => {
                            self.core.state.options.check_solutions = true;
                        }
                        "off" => {
                            self.core.state.options.check_solutions = false;
                        }
                        _ => {
                            return reply_output(format!(
                                "ERROR: Invalid value '{}' for 'solve check'\nAllowed: on, off",
                                on_off
                            ));
                        }
                    }
                    i += 3;
                    continue;
                }

                let value = args[i + 1];
                if let Some(err) = self.set_semantic_axis(arg, value) {
                    return reply_output(err);
                }
                i += 2;
            }
        }

        self.sync_config_to_simplifier();
        self.print_semantics();
        ReplReply::new() // print_semantics already printed
    }

    /// Returns Some(error_message) on failure, None on success
    pub(crate) fn set_semantic_axis(&mut self, axis: &str, value: &str) -> Option<String> {
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        match axis {
            "domain" => match value {
                "strict" => {
                    self.core.simplify_options.semantics.domain_mode = DomainMode::Strict;
                    self.core.state.options.semantics.domain_mode = DomainMode::Strict;
                }
                "generic" => {
                    self.core.simplify_options.semantics.domain_mode = DomainMode::Generic;
                    self.core.state.options.semantics.domain_mode = DomainMode::Generic;
                }
                "assume" => {
                    self.core.simplify_options.semantics.domain_mode = DomainMode::Assume;
                    self.core.state.options.semantics.domain_mode = DomainMode::Assume;
                }
                _ => {
                    return Some(format!(
                        "ERROR: Invalid value '{}' for axis 'domain'\nAllowed: strict, generic, assume",
                        value
                    ));
                }
            },
            "value" => match value {
                "real" => {
                    self.core.simplify_options.semantics.value_domain = ValueDomain::RealOnly;
                    self.core.state.options.semantics.value_domain = ValueDomain::RealOnly;
                }
                "complex" => {
                    self.core.simplify_options.semantics.value_domain = ValueDomain::ComplexEnabled;
                    self.core.state.options.semantics.value_domain = ValueDomain::ComplexEnabled;
                }
                _ => {
                    return Some(format!(
                        "ERROR: Invalid value '{}' for axis 'value'\nAllowed: real, complex",
                        value
                    ));
                }
            },
            "branch" => match value {
                "principal" => {
                    self.core.simplify_options.semantics.branch = BranchPolicy::Principal;
                    self.core.state.options.semantics.branch = BranchPolicy::Principal;
                }
                _ => {
                    return Some(format!(
                        "ERROR: Invalid value '{}' for axis 'branch'\nAllowed: principal",
                        value
                    ));
                }
            },
            "inv_trig" => match value {
                "strict" => {
                    self.core.simplify_options.semantics.inv_trig = InverseTrigPolicy::Strict;
                    self.core.state.options.semantics.inv_trig = InverseTrigPolicy::Strict;
                }
                "principal" => {
                    self.core.simplify_options.semantics.inv_trig =
                        InverseTrigPolicy::PrincipalValue;
                    self.core.state.options.semantics.inv_trig = InverseTrigPolicy::PrincipalValue;
                }
                _ => {
                    return Some(format!(
                        "ERROR: Invalid value '{}' for axis 'inv_trig'\nAllowed: strict, principal",
                        value
                    ));
                }
            },
            "const_fold" => {
                use cas_engine::const_fold::ConstFoldMode;
                match value {
                    "off" => {
                        self.core.state.options.const_fold = ConstFoldMode::Off;
                    }
                    "safe" => {
                        self.core.state.options.const_fold = ConstFoldMode::Safe;
                    }
                    _ => {
                        return Some(format!(
                            "ERROR: Invalid value '{}' for axis 'const_fold'\nAllowed: off, safe",
                            value
                        ));
                    }
                }
            }
            "assumptions" => match value {
                "off" => {
                    self.core.state.options.assumption_reporting =
                        cas_engine::AssumptionReporting::Off;
                    self.core.simplify_options.assumption_reporting =
                        cas_engine::AssumptionReporting::Off;
                }
                "summary" => {
                    self.core.state.options.assumption_reporting =
                        cas_engine::AssumptionReporting::Summary;
                    self.core.simplify_options.assumption_reporting =
                        cas_engine::AssumptionReporting::Summary;
                }
                "trace" => {
                    self.core.state.options.assumption_reporting =
                        cas_engine::AssumptionReporting::Trace;
                    self.core.simplify_options.assumption_reporting =
                        cas_engine::AssumptionReporting::Trace;
                }
                _ => {
                    return Some(format!(
                        "ERROR: Invalid value '{}' for axis 'assumptions'\nAllowed: off, summary, trace",
                        value
                    ));
                }
            },
            "assume_scope" => match value {
                "real" => {
                    self.core.simplify_options.semantics.assume_scope =
                        cas_engine::AssumeScope::Real;
                    self.core.state.options.semantics.assume_scope = cas_engine::AssumeScope::Real;
                }
                "wildcard" => {
                    self.core.simplify_options.semantics.assume_scope =
                        cas_engine::AssumeScope::Wildcard;
                    self.core.state.options.semantics.assume_scope =
                        cas_engine::AssumeScope::Wildcard;
                }
                _ => {
                    return Some(format!(
                        "ERROR: Invalid value '{}' for axis 'assume_scope'\nAllowed: real, wildcard",
                        value
                    ));
                }
            },
            "hints" => match value {
                "on" => {
                    self.core.state.options.hints_enabled = true;
                }
                "off" => {
                    self.core.state.options.hints_enabled = false;
                }
                _ => {
                    return Some(format!(
                        "ERROR: Invalid value '{}' for axis 'hints'\nAllowed: on, off",
                        value
                    ));
                }
            },
            "solve" => match value {
                "check" => {
                    return Some(
                        "ERROR: Use 'semantics set solve check on' or 'semantics set solve check off'"
                            .to_string(),
                    );
                }
                _ => {
                    return Some(format!(
                        "ERROR: Invalid value '{}' for axis 'solve'\nAllowed: 'check on', 'check off'",
                        value
                    ));
                }
            },
            "requires" => match value {
                "essential" => {
                    self.core.state.options.requires_display =
                        cas_engine::implicit_domain::RequiresDisplayLevel::Essential;
                }
                "all" => {
                    self.core.state.options.requires_display =
                        cas_engine::implicit_domain::RequiresDisplayLevel::All;
                }
                _ => {
                    return Some(format!(
                        "ERROR: Invalid value '{}' for axis 'requires'\nAllowed: essential, all",
                        value
                    ));
                }
            },
            _ => {
                return Some(format!(
                    "ERROR: Unknown axis '{}'\n\
                     Valid axes: domain, value, branch, inv_trig, const_fold, assumptions, assume_scope, hints, solve, requires",
                    axis
                ));
            }
        }
        None
    }

    /// Handle "context" command - show or switch context mode
    pub(crate) fn handle_context_command(&mut self, line: &str) {
        let reply = self.handle_context_command_core(line);
        self.print_reply(reply);
    }

    fn handle_context_command_core(&mut self, line: &str) -> ReplReply {
        use cas_engine::options::ContextMode;

        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "context" - show current context
                let ctx_str = match self.core.state.options.context_mode {
                    ContextMode::Auto => "auto",
                    ContextMode::Standard => "standard",
                    ContextMode::Solve => "solve",
                    ContextMode::IntegratePrep => "integrate",
                };
                reply_output(format!(
                    "Current context: {}\n  (use 'context auto|standard|solve|integrate' to change)",
                    ctx_str
                ))
            }
            Some(&"auto") => {
                self.core.state.options.context_mode = ContextMode::Auto;
                self.core.engine.simplifier =
                    cas_engine::Simplifier::with_profile(&self.core.state.options);
                self.sync_config_to_simplifier();
                reply_output("Context: auto (infers from expression)")
            }
            Some(&"standard") => {
                self.core.state.options.context_mode = ContextMode::Standard;
                self.core.engine.simplifier =
                    cas_engine::Simplifier::with_profile(&self.core.state.options);
                self.sync_config_to_simplifier();
                reply_output("Context: standard (safe simplification only)")
            }
            Some(&"solve") => {
                self.core.state.options.context_mode = ContextMode::Solve;
                self.core.engine.simplifier =
                    cas_engine::Simplifier::with_profile(&self.core.state.options);
                self.sync_config_to_simplifier();
                reply_output("Context: solve (preserves solver-friendly forms)")
            }
            Some(&"integrate") => {
                self.core.state.options.context_mode = ContextMode::IntegratePrep;
                self.core.engine.simplifier =
                    cas_engine::Simplifier::with_profile(&self.core.state.options);
                self.sync_config_to_simplifier();
                reply_output(
                    "Context: integrate-prep\n  ⚠️ Enables transforms for integration (telescoping, product→sum)"
                )
            }
            Some(other) => reply_output(format!(
                "Unknown context: '{}'\nUsage: context [auto | standard | solve | integrate]",
                other
            )),
        }
    }

    /// Handle "steps" command - show or switch steps collection mode AND display verbosity
    /// Collection: on, off, compact (controls StepsMode in engine)
    /// Display: verbose, succinct, normal, none (controls Verbosity in CLI)
    pub(crate) fn handle_steps_command(&mut self, line: &str) {
        let reply = self.handle_steps_command_core(line);
        self.print_reply(reply);
    }

    fn handle_steps_command_core(&mut self, line: &str) -> ReplReply {
        use cas_engine::options::StepsMode;

        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "steps" - show current mode
                let mode_str = match self.core.state.options.steps_mode {
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
                reply_output(format!(
                    "Steps collection: {}\n\
                     Steps display: {}\n\
                       (use 'steps on|off|compact' for collection)\n\
                       (use 'steps verbose|succinct|normal|none' for display)",
                    mode_str, verbosity_str
                ))
            }
            // Collection modes (StepsMode)
            Some(&"on") => {
                self.core.state.options.steps_mode = StepsMode::On;
                self.core.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Normal;
                reply_output("Steps: on (full collection, normal display)")
            }
            Some(&"off") => {
                self.core.state.options.steps_mode = StepsMode::Off;
                self.core.engine.simplifier.set_steps_mode(StepsMode::Off);
                self.verbosity = Verbosity::None;
                reply_output("Steps: off\n  ⚡ Steps disabled (faster). Warnings still enabled.")
            }
            Some(&"compact") => {
                self.core.state.options.steps_mode = StepsMode::Compact;
                self.core
                    .engine
                    .simplifier
                    .set_steps_mode(StepsMode::Compact);
                reply_output("Steps: compact (no before/after snapshots)")
            }
            // Display modes (Verbosity)
            Some(&"verbose") => {
                self.core.state.options.steps_mode = StepsMode::On;
                self.core.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Verbose;
                reply_output("Steps: verbose (all rules, full detail)")
            }
            Some(&"succinct") => {
                self.core.state.options.steps_mode = StepsMode::On;
                self.core.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Succinct;
                reply_output("Steps: succinct (compact 1-line per step)")
            }
            Some(&"normal") => {
                self.core.state.options.steps_mode = StepsMode::On;
                self.core.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Normal;
                reply_output("Steps: normal (default display)")
            }
            Some(&"none") => {
                self.verbosity = Verbosity::None;
                reply_output(
                    "Steps display: none (collection still active)\n  Use 'steps off' to also disable collection."
                )
            }
            Some(other) => reply_output(format!(
                "Unknown steps mode: '{}'\n\
                     Usage: steps [on | off | compact | verbose | succinct | normal | none]\n\
                       Collection modes:\n\
                         on      - Full steps with snapshots (default)\n\
                         off     - No steps (fastest, warnings preserved)\n\
                         compact - Minimal steps (no snapshots)\n\
                       Display modes:\n\
                         verbose - Show all rules, full detail\n\
                         succinct- Compact 1-line per step\n\
                         normal  - Standard display (default)\n\
                         none    - Hide steps output (collection still active)",
                other
            )),
        }
    }

    /// Handle "autoexpand" command - show or switch auto-expand policy
    pub(crate) fn handle_autoexpand_command(&mut self, line: &str) {
        let reply = self.handle_autoexpand_command_core(line);
        self.print_reply(reply);
    }

    fn handle_autoexpand_command_core(&mut self, line: &str) -> ReplReply {
        use cas_engine::phase::ExpandPolicy;

        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "autoexpand" - show current mode
                let policy_str = match self.core.state.options.expand_policy {
                    ExpandPolicy::Off => "off",
                    ExpandPolicy::Auto => "on",
                };
                let budget = &self.core.state.options.expand_budget;
                reply_output(format!(
                    "Auto-expand: {}\n\
                       Budget: pow<={}, base_terms<={}, gen_terms<={}, vars<={}\n\
                       (use 'autoexpand on|off' to change)",
                    policy_str,
                    budget.max_pow_exp,
                    budget.max_base_terms,
                    budget.max_generated_terms,
                    budget.max_vars
                ))
            }
            Some(&"on") => {
                self.core.state.options.expand_policy = ExpandPolicy::Auto;
                self.core.engine.simplifier =
                    cas_engine::Simplifier::with_profile(&self.core.state.options);
                self.sync_config_to_simplifier();
                let budget = &self.core.state.options.expand_budget;
                reply_output(format!(
                    "Auto-expand: on\n\
                       Budget: pow<={}, base_terms<={}, gen_terms<={}, vars<={}\n\
                       ⚠️ Expands small (sum)^n patterns automatically.",
                    budget.max_pow_exp,
                    budget.max_base_terms,
                    budget.max_generated_terms,
                    budget.max_vars
                ))
            }
            Some(&"off") => {
                self.core.state.options.expand_policy = ExpandPolicy::Off;
                self.core.engine.simplifier =
                    cas_engine::Simplifier::with_profile(&self.core.state.options);
                self.sync_config_to_simplifier();
                reply_output("Auto-expand: off\n  Polynomial expansions require explicit expand().")
            }
            Some(other) => reply_output(format!(
                "Unknown autoexpand mode: '{}'\n\
                     Usage: autoexpand [on | off]\n\
                       on  - Auto-expand cheap polynomial powers\n\
                       off - Only expand when explicitly requested (default)",
                other
            )),
        }
    }
}
