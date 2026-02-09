use super::*;

impl Repl {
    /// Handle "semantics" command - unified control for semantic axes
    pub(crate) fn handle_semantics(&mut self, line: &str) {
        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "semantics" - show current settings
                self.print_semantics();
            }
            Some(&"help") => {
                self.print_semantics_help();
            }
            Some(&"set") => {
                // Parse remaining args as axis=value pairs or axis value pairs
                self.parse_semantics_set(&args[2..]);
            }
            Some(&"domain") => {
                self.print_axis_status("domain");
            }
            Some(&"value") => {
                self.print_axis_status("value");
            }
            Some(&"branch") => {
                self.print_axis_status("branch");
            }
            Some(&"inv_trig") => {
                self.print_axis_status("inv_trig");
            }
            Some(&"const_fold") => {
                self.print_axis_status("const_fold");
            }
            Some(&"assumptions") => {
                self.print_axis_status("assumptions");
            }
            Some(&"assume_scope") => {
                self.print_axis_status("assume_scope");
            }
            Some(&"requires") => {
                self.print_axis_status("requires");
            }
            Some(&"preset") => {
                self.handle_preset(&args[2..]);
            }
            Some(other) => {
                let error_text = format!(
                    "Unknown semantics subcommand: '{}'\n\
                     Usage: semantics [set|preset|help|<axis>]\n\
                       semantics            Show all settings\n\
                       semantics <axis>     Show one axis (domain|value|branch|inv_trig|const_fold|assumptions|assume_scope|requires)\n\
                       semantics help       Show help\n\
                       semantics set ...    Change settings\n\
                       semantics preset     List/apply presets",
                    other
                );
                self.print_reply(reply_output(error_text));
            }
        }
    }

    pub(crate) fn print_semantics(&self) {
        let reply = self.print_semantics_core();
        self.print_reply(reply);
    }

    fn print_semantics_core(&self) -> ReplReply {
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        let mut lines = Vec::new();

        let domain = match self.core.simplify_options.shared.semantics.domain_mode {
            DomainMode::Strict => "strict",
            DomainMode::Assume => "assume",
            DomainMode::Generic => "generic",
        };

        let value = match self.core.simplify_options.shared.semantics.value_domain {
            ValueDomain::RealOnly => "real",
            ValueDomain::ComplexEnabled => "complex",
        };

        let branch = match self.core.simplify_options.shared.semantics.branch {
            BranchPolicy::Principal => "principal",
        };

        let inv_trig = match self.core.simplify_options.shared.semantics.inv_trig {
            InverseTrigPolicy::Strict => "strict",
            InverseTrigPolicy::PrincipalValue => "principal",
        };

        lines.push("Semantics:".to_string());
        lines.push(format!("  domain_mode: {}", domain));
        lines.push(format!("  value_domain: {}", value));

        // Show branch with inactive note if value=real
        if self.core.simplify_options.shared.semantics.value_domain == ValueDomain::RealOnly {
            lines.push(format!(
                "  branch: {} (inactive: value_domain=real)",
                branch
            ));
        } else {
            lines.push(format!("  branch: {}", branch));
        }

        lines.push(format!("  inv_trig: {}", inv_trig));

        let const_fold = match self.core.state.options.const_fold {
            cas_engine::const_fold::ConstFoldMode::Off => "off",
            cas_engine::const_fold::ConstFoldMode::Safe => "safe",
        };
        lines.push(format!("  const_fold: {}", const_fold));

        let assumptions = match self.core.state.options.shared.assumption_reporting {
            cas_engine::AssumptionReporting::Off => "off",
            cas_engine::AssumptionReporting::Summary => "summary",
            cas_engine::AssumptionReporting::Trace => "trace",
        };
        lines.push(format!("  assumptions: {}", assumptions));

        // Show assume_scope with inactive note if domain_mode != Assume
        let assume_scope = match self.core.simplify_options.shared.semantics.assume_scope {
            cas_engine::AssumeScope::Real => "real",
            cas_engine::AssumeScope::Wildcard => "wildcard",
        };
        if self.core.simplify_options.shared.semantics.domain_mode != DomainMode::Assume {
            lines.push(format!(
                "  assume_scope: {} (inactive: domain_mode != assume)",
                assume_scope
            ));
        } else {
            lines.push(format!("  assume_scope: {}", assume_scope));
        }

        // Show hints_enabled
        let hints = if self.core.state.options.hints_enabled {
            "on"
        } else {
            "off"
        };
        lines.push(format!("  hints: {}", hints));

        // Show requires display level
        let requires = match self.core.state.options.requires_display {
            cas_engine::implicit_domain::RequiresDisplayLevel::Essential => "essential",
            cas_engine::implicit_domain::RequiresDisplayLevel::All => "all",
        };
        lines.push(format!("  requires: {}", requires));

        reply_output(lines.join("\n"))
    }

    /// Print status for a single semantic axis with current value and available options
    pub(crate) fn print_axis_status(&self, axis: &str) {
        let reply = self.print_axis_status_core(axis);
        self.print_reply(reply);
    }

    fn print_axis_status_core(&self, axis: &str) -> ReplReply {
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        let mut lines = Vec::new();

        match axis {
            "domain" => {
                let current = match self.core.simplify_options.shared.semantics.domain_mode {
                    DomainMode::Strict => "strict",
                    DomainMode::Assume => "assume",
                    DomainMode::Generic => "generic",
                };
                lines.push(format!("domain: {}", current));
                lines.push("  Values: strict | generic | assume".to_string());
                lines.push("  strict:  No domain assumptions (x/x stays x/x)".to_string());
                lines.push("  generic: Classic CAS 'almost everywhere' algebra".to_string());
                lines.push("  assume:  Use assumptions with warnings".to_string());
            }
            "value" => {
                let current = match self.core.simplify_options.shared.semantics.value_domain {
                    ValueDomain::RealOnly => "real",
                    ValueDomain::ComplexEnabled => "complex",
                };
                lines.push(format!("value: {}", current));
                lines.push("  Values: real | complex".to_string());
                lines.push("  real:    ℝ only (sqrt(-1) undefined)".to_string());
                lines.push("  complex: ℂ enabled (sqrt(-1) = i)".to_string());
            }
            "branch" => {
                let current = match self.core.simplify_options.shared.semantics.branch {
                    BranchPolicy::Principal => "principal",
                };
                let inactive = self.core.simplify_options.shared.semantics.value_domain
                    == ValueDomain::RealOnly;
                if inactive {
                    lines.push(format!("branch: {} (inactive: value=real)", current));
                } else {
                    lines.push(format!("branch: {}", current));
                }
                lines.push("  Values: principal".to_string());
                lines.push(
                    "  principal: Use principal branch for multi-valued functions".to_string(),
                );
                if inactive {
                    lines.push("  Note: Only active when value=complex".to_string());
                }
            }
            "inv_trig" => {
                let current = match self.core.simplify_options.shared.semantics.inv_trig {
                    InverseTrigPolicy::Strict => "strict",
                    InverseTrigPolicy::PrincipalValue => "principal",
                };
                lines.push(format!("inv_trig: {}", current));
                lines.push("  Values: strict | principal".to_string());
                lines.push("  strict:    arctan(tan(x)) unchanged".to_string());
                lines.push("  principal: arctan(tan(x)) → x with warning".to_string());
            }
            "const_fold" => {
                let current = match self.core.state.options.const_fold {
                    cas_engine::const_fold::ConstFoldMode::Off => "off",
                    cas_engine::const_fold::ConstFoldMode::Safe => "safe",
                };
                lines.push(format!("const_fold: {}", current));
                lines.push("  Values: off | safe".to_string());
                lines.push("  off:  No constant folding (defer semantic decisions)".to_string());
                lines.push("  safe: Fold literals (2^3 → 8, sqrt(-1) → i if complex)".to_string());
            }
            "assumptions" => {
                let current = match self.core.state.options.shared.assumption_reporting {
                    cas_engine::AssumptionReporting::Off => "off",
                    cas_engine::AssumptionReporting::Summary => "summary",
                    cas_engine::AssumptionReporting::Trace => "trace",
                };
                lines.push(format!("assumptions: {}", current));
                lines.push("  Values: off | summary | trace".to_string());
                lines.push("  off:     No assumption reporting".to_string());
                lines.push("  summary: Deduped summary line at end".to_string());
                lines.push("  trace:   Detailed trace (future)".to_string());
            }
            "assume_scope" => {
                let current = match self.core.simplify_options.shared.semantics.assume_scope {
                    cas_engine::AssumeScope::Real => "real",
                    cas_engine::AssumeScope::Wildcard => "wildcard",
                };
                let inactive =
                    self.core.simplify_options.shared.semantics.domain_mode != DomainMode::Assume;
                if inactive {
                    lines.push(format!(
                        "assume_scope: {} (inactive: domain_mode != assume)",
                        current
                    ));
                } else {
                    lines.push(format!("assume_scope: {}", current));
                }
                lines.push("  Values: real | wildcard".to_string());
                lines.push("  real:     Assume for ℝ, error if ℂ needed".to_string());
                lines.push("  wildcard: Assume for ℝ, residual+warning if ℂ needed".to_string());
                if inactive {
                    lines.push("  Note: Only active when domain_mode=assume".to_string());
                }
            }
            "requires" => {
                let current = match self.core.state.options.requires_display {
                    cas_engine::implicit_domain::RequiresDisplayLevel::Essential => "essential",
                    cas_engine::implicit_domain::RequiresDisplayLevel::All => "all",
                };
                lines.push(format!("requires: {}", current));
                lines.push("  Values: essential | all".to_string());
                lines
                    .push("  essential: Only show requires whose witness was consumed".to_string());
                lines.push("  all:       Show all requires including implicit ones".to_string());
            }
            _ => {
                lines.push(format!("Unknown axis: {}", axis));
            }
        }

        reply_output(lines.join("\n"))
    }

    pub(crate) fn print_semantics_help(&self) {
        let reply = self.print_semantics_help_core();
        self.print_reply(reply);
    }

    fn print_semantics_help_core(&self) -> ReplReply {
        let text = r#"Semantics: Control evaluation semantics

Usage:
  semantics                    Show current settings
  semantics set <axis> <val>   Set one axis
  semantics set k=v k=v ...    Set multiple axes

Axes:
  domain      strict | generic | assume
              strict:  No domain assumptions (x/x stays x/x)
              generic: Classic CAS 'almost everywhere' algebra
              assume:  Use assumptions with warnings

  value       real | complex
              real:    ℝ only (sqrt(-1) undefined)
              complex: ℂ enabled (sqrt(-1) = i)

  branch      principal
              (only active when value=complex)

  inv_trig    strict | principal
              strict:    arctan(tan(x)) unchanged
              principal: arctan(tan(x)) → x with warning

  const_fold  off | safe
              off:  No constant folding
              safe: Fold literals (2^3 → 8)

  assume_scope real | wildcard
              real:     Assume for ℝ, error if ℂ needed
              wildcard: Assume for ℝ, residual+warning if ℂ needed
              (only active when domain_mode=assume)

  requires    essential | all
              essential: Show only requires whose witness was consumed
              all:       Show all requires including implicit ones

Examples:
  semantics set domain strict
  semantics set value complex inv_trig principal
  semantics set domain=strict value=complex
  semantics set assume_scope wildcard

Presets:
  semantics preset              List available presets
  semantics preset <name>       Apply a preset
  semantics preset help <name>  Show preset details"#;
        reply_output(text)
    }

    /// Handle "semantics preset" subcommand
    pub(crate) fn handle_preset(&mut self, args: &[&str]) {
        let reply = self.handle_preset_core(args);
        self.print_reply(reply);
    }

    fn handle_preset_core(&mut self, args: &[&str]) -> ReplReply {
        use cas_engine::const_fold::ConstFoldMode;
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        // Preset definitions: (name, description, domain, value, branch, inv_trig, const_fold)
        struct Preset {
            name: &'static str,
            description: &'static str,
            domain: DomainMode,
            value: ValueDomain,
            branch: BranchPolicy,
            inv_trig: InverseTrigPolicy,
            const_fold: ConstFoldMode,
        }

        let presets = [
            Preset {
                name: "default",
                description: "Reset to engine defaults",
                domain: DomainMode::Generic,
                value: ValueDomain::RealOnly,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::Strict,
                const_fold: ConstFoldMode::Off,
            },
            Preset {
                name: "strict",
                description: "Conservative real + strict domain",
                domain: DomainMode::Strict,
                value: ValueDomain::RealOnly,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::Strict,
                const_fold: ConstFoldMode::Off,
            },
            Preset {
                name: "complex",
                description: "Enable ℂ + safe const_fold (sqrt(-1) → i)",
                domain: DomainMode::Generic,
                value: ValueDomain::ComplexEnabled,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::Strict,
                const_fold: ConstFoldMode::Safe,
            },
            Preset {
                name: "school",
                description: "Real + principal inverse trig (arctan(tan(x)) → x)",
                domain: DomainMode::Generic,
                value: ValueDomain::RealOnly,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::PrincipalValue,
                const_fold: ConstFoldMode::Off,
            },
        ];

        let mut lines = Vec::new();

        match args.first() {
            None => {
                // List presets
                lines.push("Available presets:".to_string());
                for p in &presets {
                    lines.push(format!("  {:10} {}", p.name, p.description));
                }
                lines.push(String::new());
                lines.push("Usage:".to_string());
                lines.push("  semantics preset <name>       Apply preset".to_string());
                lines.push("  semantics preset help <name>  Show preset axes".to_string());
            }
            Some(&"help") => {
                // Show preset details
                let name = match args.get(1) {
                    Some(name) => *name,
                    None => {
                        lines.push("Usage: semantics preset help <name>".to_string());
                        lines.push("Presets: default, strict, complex, school".to_string());
                        return reply_output(lines.join("\n"));
                    }
                };
                if let Some(p) = presets.iter().find(|p| p.name == name) {
                    let domain_str = match p.domain {
                        DomainMode::Strict => "strict",
                        DomainMode::Generic => "generic",
                        DomainMode::Assume => "assume",
                    };
                    let value_str = match p.value {
                        ValueDomain::RealOnly => "real",
                        ValueDomain::ComplexEnabled => "complex",
                    };
                    let inv_trig_str = match p.inv_trig {
                        InverseTrigPolicy::Strict => "strict",
                        InverseTrigPolicy::PrincipalValue => "principal",
                    };
                    let const_fold_str = match p.const_fold {
                        ConstFoldMode::Off => "off",
                        ConstFoldMode::Safe => "safe",
                    };
                    lines.push(format!("{}:", p.name));
                    lines.push(format!("  domain_mode  = {}", domain_str));
                    lines.push(format!("  value_domain = {}", value_str));
                    lines.push("  branch       = principal".to_string());
                    lines.push(format!("  inv_trig     = {}", inv_trig_str));
                    lines.push(format!("  const_fold   = {}", const_fold_str));
                    lines.push(String::new());
                    lines.push(format!("Purpose: {}", p.description));
                } else {
                    lines.push(format!("Unknown preset: '{}'", name));
                    lines.push("Available: default, strict, complex, school".to_string());
                }
            }
            Some(name) => {
                // Apply preset
                if let Some(p) = presets.iter().find(|preset| preset.name == *name) {
                    // Capture old values for diff
                    let old_domain = self.core.simplify_options.shared.semantics.domain_mode;
                    let old_value = self.core.simplify_options.shared.semantics.value_domain;
                    let old_branch = self.core.simplify_options.shared.semantics.branch;
                    let old_inv_trig = self.core.simplify_options.shared.semantics.inv_trig;
                    let old_const_fold = self.core.state.options.const_fold;

                    // Apply preset
                    self.core.simplify_options.shared.semantics.domain_mode = p.domain;
                    self.core.simplify_options.shared.semantics.value_domain = p.value;
                    self.core.simplify_options.shared.semantics.branch = p.branch;
                    self.core.simplify_options.shared.semantics.inv_trig = p.inv_trig;
                    self.core.state.options.const_fold = p.const_fold;
                    // Sync to state.options (used by evaluation pipeline)
                    self.core.state.options.shared.semantics.domain_mode = p.domain;
                    self.core.state.options.shared.semantics.value_domain = p.value;
                    self.core.state.options.shared.semantics.branch = p.branch;
                    self.core.state.options.shared.semantics.inv_trig = p.inv_trig;

                    self.sync_config_to_simplifier();

                    lines.push(format!("Applied preset: {}", p.name));
                    lines.push("Changes:".to_string());

                    // Print changes
                    let mut changes = 0;
                    if old_domain != p.domain {
                        let old_str = match old_domain {
                            DomainMode::Strict => "strict",
                            DomainMode::Generic => "generic",
                            DomainMode::Assume => "assume",
                        };
                        let new_str = match p.domain {
                            DomainMode::Strict => "strict",
                            DomainMode::Generic => "generic",
                            DomainMode::Assume => "assume",
                        };
                        lines.push(format!("  domain_mode:  {} → {}", old_str, new_str));
                        changes += 1;
                    }
                    if old_value != p.value {
                        let old_str = match old_value {
                            ValueDomain::RealOnly => "real",
                            ValueDomain::ComplexEnabled => "complex",
                        };
                        let new_str = match p.value {
                            ValueDomain::RealOnly => "real",
                            ValueDomain::ComplexEnabled => "complex",
                        };
                        lines.push(format!("  value_domain: {} → {}", old_str, new_str));
                        changes += 1;
                    }
                    if old_branch != p.branch {
                        lines.push("  branch:       principal → principal".to_string());
                        changes += 1;
                    }
                    if old_inv_trig != p.inv_trig {
                        let old_str = match old_inv_trig {
                            InverseTrigPolicy::Strict => "strict",
                            InverseTrigPolicy::PrincipalValue => "principal",
                        };
                        let new_str = match p.inv_trig {
                            InverseTrigPolicy::Strict => "strict",
                            InverseTrigPolicy::PrincipalValue => "principal",
                        };
                        lines.push(format!("  inv_trig:     {} → {}", old_str, new_str));
                        changes += 1;
                    }
                    if old_const_fold != p.const_fold {
                        let old_str = match old_const_fold {
                            ConstFoldMode::Off => "off",
                            ConstFoldMode::Safe => "safe",
                        };
                        let new_str = match p.const_fold {
                            ConstFoldMode::Off => "off",
                            ConstFoldMode::Safe => "safe",
                        };
                        lines.push(format!("  const_fold:   {} → {}", old_str, new_str));
                        changes += 1;
                    }
                    if changes == 0 {
                        lines.push("  (no changes - already at this preset)".to_string());
                    }
                } else {
                    lines.push(format!("Unknown preset: '{}'", name));
                    lines.push("Available: default, strict, complex, school".to_string());
                }
            }
        }

        reply_output(lines.join("\n"))
    }
}
