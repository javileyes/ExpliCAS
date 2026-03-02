/// Snapshot of semantic settings used for user-facing formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsViewState {
    pub domain_mode: crate::DomainMode,
    pub value_domain: crate::ValueDomain,
    pub branch: crate::BranchPolicy,
    pub inv_trig: crate::InverseTrigPolicy,
    pub const_fold: crate::ConstFoldMode,
    pub assumption_reporting: crate::AssumptionReporting,
    pub assume_scope: crate::AssumeScope,
    pub hints_enabled: bool,
    pub requires_display: crate::RequiresDisplayLevel,
}

/// Build a semantics view snapshot from simplifier + eval options.
pub fn semantics_view_state_from_options(
    simplify_options: &crate::SimplifyOptions,
    eval_options: &crate::EvalOptions,
) -> SemanticsViewState {
    SemanticsViewState {
        domain_mode: simplify_options.shared.semantics.domain_mode,
        value_domain: simplify_options.shared.semantics.value_domain,
        branch: simplify_options.shared.semantics.branch,
        inv_trig: simplify_options.shared.semantics.inv_trig,
        const_fold: eval_options.const_fold,
        assumption_reporting: eval_options.shared.assumption_reporting,
        assume_scope: simplify_options.shared.semantics.assume_scope,
        hints_enabled: eval_options.hints_enabled,
        requires_display: eval_options.requires_display,
    }
}

/// Format all semantic axis settings.
pub fn format_semantics_overview_lines(state: &SemanticsViewState) -> Vec<String> {
    let mut lines = Vec::new();

    let domain = match state.domain_mode {
        crate::DomainMode::Strict => "strict",
        crate::DomainMode::Assume => "assume",
        crate::DomainMode::Generic => "generic",
    };

    let value = match state.value_domain {
        crate::ValueDomain::RealOnly => "real",
        crate::ValueDomain::ComplexEnabled => "complex",
    };

    let branch = match state.branch {
        crate::BranchPolicy::Principal => "principal",
    };

    let inv_trig = match state.inv_trig {
        crate::InverseTrigPolicy::Strict => "strict",
        crate::InverseTrigPolicy::PrincipalValue => "principal",
    };

    lines.push("Semantics:".to_string());
    lines.push(format!("  domain_mode: {}", domain));
    lines.push(format!("  value_domain: {}", value));

    if state.value_domain == crate::ValueDomain::RealOnly {
        lines.push(format!(
            "  branch: {} (inactive: value_domain=real)",
            branch
        ));
    } else {
        lines.push(format!("  branch: {}", branch));
    }

    lines.push(format!("  inv_trig: {}", inv_trig));

    let const_fold = match state.const_fold {
        crate::ConstFoldMode::Off => "off",
        crate::ConstFoldMode::Safe => "safe",
    };
    lines.push(format!("  const_fold: {}", const_fold));

    let assumptions = match state.assumption_reporting {
        crate::AssumptionReporting::Off => "off",
        crate::AssumptionReporting::Summary => "summary",
        crate::AssumptionReporting::Trace => "trace",
    };
    lines.push(format!("  assumptions: {}", assumptions));

    let assume_scope = match state.assume_scope {
        crate::AssumeScope::Real => "real",
        crate::AssumeScope::Wildcard => "wildcard",
    };
    if state.domain_mode != crate::DomainMode::Assume {
        lines.push(format!(
            "  assume_scope: {} (inactive: domain_mode != assume)",
            assume_scope
        ));
    } else {
        lines.push(format!("  assume_scope: {}", assume_scope));
    }

    lines.push(format!(
        "  hints: {}",
        if state.hints_enabled { "on" } else { "off" }
    ));

    let requires = match state.requires_display {
        crate::RequiresDisplayLevel::Essential => "essential",
        crate::RequiresDisplayLevel::All => "all",
    };
    lines.push(format!("  requires: {}", requires));

    lines
}

/// Format one semantic axis description and values.
pub fn format_semantics_axis_lines(state: &SemanticsViewState, axis: &str) -> Vec<String> {
    let mut lines = Vec::new();

    match axis {
        "domain" => {
            let current = match state.domain_mode {
                crate::DomainMode::Strict => "strict",
                crate::DomainMode::Assume => "assume",
                crate::DomainMode::Generic => "generic",
            };
            lines.push(format!("domain: {}", current));
            lines.push("  Values: strict | generic | assume".to_string());
            lines.push("  strict:  No domain assumptions (x/x stays x/x)".to_string());
            lines.push("  generic: Classic CAS 'almost everywhere' algebra".to_string());
            lines.push("  assume:  Use assumptions with warnings".to_string());
        }
        "value" => {
            let current = match state.value_domain {
                crate::ValueDomain::RealOnly => "real",
                crate::ValueDomain::ComplexEnabled => "complex",
            };
            lines.push(format!("value: {}", current));
            lines.push("  Values: real | complex".to_string());
            lines.push("  real:    ℝ only (sqrt(-1) undefined)".to_string());
            lines.push("  complex: ℂ enabled (sqrt(-1) = i)".to_string());
        }
        "branch" => {
            let current = match state.branch {
                crate::BranchPolicy::Principal => "principal",
            };
            let inactive = state.value_domain == crate::ValueDomain::RealOnly;
            if inactive {
                lines.push(format!("branch: {} (inactive: value=real)", current));
            } else {
                lines.push(format!("branch: {}", current));
            }
            lines.push("  Values: principal".to_string());
            lines.push("  principal: Use principal branch for multi-valued functions".to_string());
            if inactive {
                lines.push("  Note: Only active when value=complex".to_string());
            }
        }
        "inv_trig" => {
            let current = match state.inv_trig {
                crate::InverseTrigPolicy::Strict => "strict",
                crate::InverseTrigPolicy::PrincipalValue => "principal",
            };
            lines.push(format!("inv_trig: {}", current));
            lines.push("  Values: strict | principal".to_string());
            lines.push("  strict:    arctan(tan(x)) unchanged".to_string());
            lines.push("  principal: arctan(tan(x)) → x with warning".to_string());
        }
        "const_fold" => {
            let current = match state.const_fold {
                crate::ConstFoldMode::Off => "off",
                crate::ConstFoldMode::Safe => "safe",
            };
            lines.push(format!("const_fold: {}", current));
            lines.push("  Values: off | safe".to_string());
            lines.push("  off:  No constant folding (defer semantic decisions)".to_string());
            lines.push("  safe: Fold literals (2^3 → 8, sqrt(-1) → i if complex)".to_string());
        }
        "assumptions" => {
            let current = match state.assumption_reporting {
                crate::AssumptionReporting::Off => "off",
                crate::AssumptionReporting::Summary => "summary",
                crate::AssumptionReporting::Trace => "trace",
            };
            lines.push(format!("assumptions: {}", current));
            lines.push("  Values: off | summary | trace".to_string());
            lines.push("  off:     No assumption reporting".to_string());
            lines.push("  summary: Deduped summary line at end".to_string());
            lines.push("  trace:   Detailed trace (future)".to_string());
        }
        "assume_scope" => {
            let current = match state.assume_scope {
                crate::AssumeScope::Real => "real",
                crate::AssumeScope::Wildcard => "wildcard",
            };
            let inactive = state.domain_mode != crate::DomainMode::Assume;
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
            let current = match state.requires_display {
                crate::RequiresDisplayLevel::Essential => "essential",
                crate::RequiresDisplayLevel::All => "all",
            };
            lines.push(format!("requires: {}", current));
            lines.push("  Values: essential | all".to_string());
            lines.push("  essential: Only show requires whose witness was consumed".to_string());
            lines.push("  all:       Show all requires including implicit ones".to_string());
        }
        _ => {
            lines.push(format!("Unknown axis: {}", axis));
        }
    }

    lines
}

pub fn format_semantics_unknown_subcommand_message(subcommand: &str) -> String {
    format!(
        "Unknown semantics subcommand: '{}'\n\
         Usage: semantics [set|preset|help|<axis>]\n\
           semantics            Show all settings\n\
           semantics <axis>     Show one axis (domain|value|branch|inv_trig|const_fold|assumptions|assume_scope|requires)\n\
           semantics help       Show help\n\
           semantics set ...    Change settings\n\
           semantics preset     List/apply presets",
        subcommand
    )
}

pub fn semantics_help_message() -> &'static str {
    r#"Semantics: Control evaluation semantics

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
  semantics preset help <name>  Show preset details"#
}

#[cfg(test)]
mod tests {
    use super::{
        format_semantics_axis_lines, format_semantics_overview_lines,
        format_semantics_unknown_subcommand_message, semantics_help_message,
        semantics_view_state_from_options, SemanticsViewState,
    };

    fn state() -> SemanticsViewState {
        SemanticsViewState {
            domain_mode: crate::DomainMode::Generic,
            value_domain: crate::ValueDomain::RealOnly,
            branch: crate::BranchPolicy::Principal,
            inv_trig: crate::InverseTrigPolicy::Strict,
            const_fold: crate::ConstFoldMode::Off,
            assumption_reporting: crate::AssumptionReporting::Summary,
            assume_scope: crate::AssumeScope::Real,
            hints_enabled: true,
            requires_display: crate::RequiresDisplayLevel::Essential,
        }
    }

    #[test]
    fn format_semantics_overview_lines_marks_inactive_branch_for_real_mode() {
        let lines = format_semantics_overview_lines(&state());
        assert!(lines
            .iter()
            .any(|line| line.contains("branch: principal (inactive: value_domain=real)")));
    }

    #[test]
    fn format_semantics_axis_lines_reports_unknown_axis() {
        let lines = format_semantics_axis_lines(&state(), "missing");
        assert_eq!(lines, vec!["Unknown axis: missing".to_string()]);
    }

    #[test]
    fn semantics_help_message_mentions_presets() {
        assert!(semantics_help_message().contains("semantics preset"));
    }

    #[test]
    fn format_semantics_unknown_subcommand_message_mentions_usage() {
        let text = format_semantics_unknown_subcommand_message("nope");
        assert!(text.contains("Unknown semantics subcommand: 'nope'"));
        assert!(text.contains("Usage: semantics"));
    }

    #[test]
    fn semantics_view_state_from_options_reads_requires_display() {
        let simplify_options = crate::SimplifyOptions::default();
        let eval_options = crate::EvalOptions {
            requires_display: crate::RequiresDisplayLevel::All,
            ..crate::EvalOptions::default()
        };
        let state = semantics_view_state_from_options(&simplify_options, &eval_options);
        assert_eq!(state.requires_display, crate::RequiresDisplayLevel::All);
    }
}
