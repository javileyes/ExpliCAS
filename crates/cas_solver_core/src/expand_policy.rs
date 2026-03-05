//! Shared auto-expand policy types.
//!
//! These types are runtime-agnostic and can be reused by both `cas_engine`
//! and `cas_solver` without forcing a dependency on engine internals.

/// Policy controlling automatic expansion during simplification.
///
/// Default is `Off` (structure-preserving).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExpandPolicy {
    /// Standard mode: never auto-expand. Preserves `(x+1)^n` forms.
    #[default]
    Off,
    /// Automatically expand cheap polynomial powers within budget limits.
    Auto,
}

/// Budget limits for auto-expansion to prevent combinatorial explosion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpandBudget {
    /// Maximum exponent to auto-expand.
    pub max_pow_exp: u32,
    /// Maximum terms in base.
    pub max_base_terms: u32,
    /// Maximum terms in expanded result.
    pub max_generated_terms: u32,
    /// Maximum number of variables in base expression.
    pub max_vars: u32,
}

impl Default for ExpandBudget {
    fn default() -> Self {
        Self {
            max_pow_exp: 6,
            max_base_terms: 4,
            max_generated_terms: 300,
            max_vars: 4,
        }
    }
}

impl ExpandBudget {
    /// Check if a log expansion is allowed by budget.
    ///
    /// Returns true if `base_terms`, `gen_terms`, and optional `pow_exp`
    /// are all within limits.
    pub fn allows_log_expansion(
        &self,
        base_terms: u32,
        gen_terms: u32,
        pow_exp: Option<u32>,
    ) -> bool {
        if base_terms > self.max_base_terms {
            return false;
        }
        if gen_terms > self.max_generated_terms {
            return false;
        }
        if let Some(n) = pow_exp {
            if n > self.max_pow_exp {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpandBudget, ExpandPolicy};

    #[test]
    fn policy_default_is_off() {
        assert_eq!(ExpandPolicy::default(), ExpandPolicy::Off);
    }

    #[test]
    fn budget_defaults_match_contract() {
        let budget = ExpandBudget::default();
        assert_eq!(budget.max_pow_exp, 6);
        assert_eq!(budget.max_base_terms, 4);
        assert_eq!(budget.max_generated_terms, 300);
        assert_eq!(budget.max_vars, 4);
    }

    #[test]
    fn budget_gate_rejects_over_limit_inputs() {
        let budget = ExpandBudget::default();
        assert!(!budget.allows_log_expansion(5, 20, Some(2)));
        assert!(!budget.allows_log_expansion(3, 301, Some(2)));
        assert!(!budget.allows_log_expansion(3, 20, Some(7)));
        assert!(budget.allows_log_expansion(3, 20, Some(6)));
    }
}
