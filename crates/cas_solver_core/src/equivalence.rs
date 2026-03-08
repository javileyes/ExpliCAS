use std::collections::HashMap;

/// Result of equivalence checking between two expressions.
#[derive(Debug, Clone)]
pub enum EquivalenceResult {
    /// A ≡ B unconditionally (no domain assumptions needed)
    True,
    /// A ≡ B under specified conditions (domain restrictions)
    ConditionalTrue {
        /// Requires conditions introduced during simplification
        requires: Vec<String>,
    },
    /// A ≢ B (found counterexample or proved non-equivalent)
    False,
    /// Cannot determine (no proof either way)
    Unknown,
}

impl EquivalenceResult {
    /// Returns true if the result indicates equivalence (True or ConditionalTrue)
    pub fn is_equivalent(&self) -> bool {
        matches!(
            self,
            EquivalenceResult::True | EquivalenceResult::ConditionalTrue { .. }
        )
    }
}

/// Default numeric probe value used in equivalence fallback checks.
pub const DEFAULT_EQUIV_NUMERIC_PROBE: f64 = 1.23456789;
/// Default epsilon for "near zero" numeric equivalence checks.
pub const DEFAULT_EQUIV_NUMERIC_EPS: f64 = 1e-9;

/// Build the default variable->probe-value map for numeric equivalence checks.
pub fn default_equiv_probe_map<I>(vars: I) -> HashMap<String, f64>
where
    I: IntoIterator<Item = String>,
{
    vars.into_iter()
        .map(|var| (var, DEFAULT_EQUIV_NUMERIC_PROBE))
        .collect()
}

/// Returns true when numeric residual is close enough to zero.
#[inline]
pub fn is_numeric_equiv_zero(value: f64) -> bool {
    value.abs() < DEFAULT_EQUIV_NUMERIC_EPS
}

/// Normalize, deduplicate, and sort requires strings for stable output.
///
/// Normalization rules:
/// - `expr ≠ 0` where `expr` starts with `-` is canonicalized to positive form.
/// - exact duplicates are removed.
/// - output is sorted lexicographically for deterministic rendering.
pub fn normalize_requires(requires: &mut Vec<String>) {
    use std::collections::HashSet;

    for req in requires.iter_mut() {
        if let Some(expr_part) = req.strip_suffix(" ≠ 0") {
            let trimmed = expr_part.trim();
            if let Some(inner) = trimmed.strip_prefix("-(") {
                if let Some(inner) = inner.strip_suffix(")") {
                    *req = format!("{} ≠ 0", inner.trim());
                }
            } else if let Some(inner) = trimmed.strip_prefix("-") {
                if !inner.starts_with('(') && !inner.contains(' ') {
                    *req = format!("{} ≠ 0", inner.trim());
                }
            }
        }
    }

    let mut seen = HashSet::new();
    requires.retain(|r| seen.insert(r.clone()));
    requires.sort();
}

#[cfg(test)]
mod tests {
    use super::{
        default_equiv_probe_map, is_numeric_equiv_zero, normalize_requires, EquivalenceResult,
        DEFAULT_EQUIV_NUMERIC_PROBE,
    };

    #[test]
    fn is_equivalent_true_and_conditional() {
        assert!(EquivalenceResult::True.is_equivalent());
        assert!(EquivalenceResult::ConditionalTrue {
            requires: vec!["x != 0".to_string()]
        }
        .is_equivalent());
    }

    #[test]
    fn is_equivalent_false_and_unknown() {
        assert!(!EquivalenceResult::False.is_equivalent());
        assert!(!EquivalenceResult::Unknown.is_equivalent());
    }

    #[test]
    fn normalize_requires_strips_leading_negation_and_dedupes() {
        let mut requires = vec![
            "-x ≠ 0".to_string(),
            "-(y + 1) ≠ 0".to_string(),
            "x ≠ 0".to_string(),
            "z > 0".to_string(),
            "x ≠ 0".to_string(),
        ];
        normalize_requires(&mut requires);
        assert_eq!(
            requires,
            vec![
                "x ≠ 0".to_string(),
                "y + 1 ≠ 0".to_string(),
                "z > 0".to_string(),
            ]
        );
    }

    #[test]
    fn default_probe_map_assigns_probe_value_to_all_vars() {
        let map = default_equiv_probe_map(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(map.get("x"), Some(&DEFAULT_EQUIV_NUMERIC_PROBE));
        assert_eq!(map.get("y"), Some(&DEFAULT_EQUIV_NUMERIC_PROBE));
    }

    #[test]
    fn numeric_zero_uses_default_epsilon() {
        assert!(is_numeric_equiv_zero(1e-10));
        assert!(!is_numeric_equiv_zero(1e-6));
    }
}
