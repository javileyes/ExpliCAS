//! Domain-airbag policy routing shared across runtime crates.
//!
//! Converts dropped implicit-domain predicates into one of:
//! - block rewrite (`Strict`)
//! - attach required conditions (`Generic`)
//! - attach assumptions (`Assume`)

use crate::domain_mode::DomainMode;

/// Result of applying domain-airbag policy for one rewrite candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainAirbagOutcome<Condition, Assumption> {
    /// True when the rewrite must be blocked.
    pub blocked: bool,
    /// Conditions to attach as `required`.
    pub required: Vec<Condition>,
    /// Assumptions to emit in Assume mode.
    pub assumptions: Vec<Assumption>,
}

impl<Condition, Assumption> Default for DomainAirbagOutcome<Condition, Assumption> {
    fn default() -> Self {
        Self {
            blocked: false,
            required: Vec::new(),
            assumptions: Vec::new(),
        }
    }
}

/// Apply domain policy to dropped predicates.
///
/// The caller decides how to convert dropped conditions into assumption events
/// via `to_assumption`. Returning `None` means the condition is skipped in
/// Assume mode.
pub fn apply_domain_airbag_policy<Condition, Assumption, I, F>(
    mode: DomainMode,
    dropped: I,
    mut to_assumption: F,
) -> DomainAirbagOutcome<Condition, Assumption>
where
    I: IntoIterator<Item = Condition>,
    F: FnMut(&Condition) -> Option<Assumption>,
{
    let dropped_vec: Vec<Condition> = dropped.into_iter().collect();

    match mode {
        DomainMode::Strict => DomainAirbagOutcome {
            blocked: !dropped_vec.is_empty(),
            ..Default::default()
        },
        DomainMode::Generic => DomainAirbagOutcome {
            required: dropped_vec,
            ..Default::default()
        },
        DomainMode::Assume => {
            let mut assumptions = Vec::new();
            for cond in &dropped_vec {
                if let Some(event) = to_assumption(cond) {
                    assumptions.push(event);
                }
            }
            DomainAirbagOutcome {
                assumptions,
                ..Default::default()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_domain_airbag_policy, DomainAirbagOutcome};
    use crate::domain_mode::DomainMode;

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Cond {
        A,
        B,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Assumption {
        FromA,
    }

    #[test]
    fn strict_blocks_when_dropped_is_non_empty() {
        let out = apply_domain_airbag_policy(DomainMode::Strict, vec![Cond::A, Cond::B], |_cond| {
            None::<Assumption>
        });
        assert_eq!(
            out,
            DomainAirbagOutcome {
                blocked: true,
                required: vec![],
                assumptions: vec![],
            }
        );
    }

    #[test]
    fn strict_noop_when_dropped_is_empty() {
        let out = apply_domain_airbag_policy(DomainMode::Strict, Vec::<Cond>::new(), |_cond| {
            None::<Assumption>
        });
        assert!(!out.blocked);
        assert!(out.required.is_empty());
        assert!(out.assumptions.is_empty());
    }

    #[test]
    fn generic_attaches_required_conditions() {
        let dropped = vec![Cond::A, Cond::B];
        let out = apply_domain_airbag_policy(DomainMode::Generic, dropped.clone(), |_cond| {
            None::<Assumption>
        });
        assert!(!out.blocked);
        assert_eq!(out.required, dropped);
        assert!(out.assumptions.is_empty());
    }

    #[test]
    fn assume_maps_conditions_to_assumptions() {
        let out =
            apply_domain_airbag_policy(
                DomainMode::Assume,
                vec![Cond::A, Cond::B],
                |cond| match cond {
                    Cond::A => Some(Assumption::FromA),
                    Cond::B => None,
                },
            );
        assert!(!out.blocked);
        assert!(out.required.is_empty());
        assert_eq!(out.assumptions, vec![Assumption::FromA]);
    }
}
