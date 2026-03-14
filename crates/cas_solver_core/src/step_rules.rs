//! Shared rule-name heuristics for runtime step post-processing.

/// Returns true when rule name corresponds to expansion.
pub fn is_expansion_rule_name(name: &str) -> bool {
    name == "Binomial Expansion" || name == "Expand"
}

/// Returns true when a step rule should always be kept for didactic reasons.
pub fn is_always_keep_step_rule_name(name: &str) -> bool {
    name == "Sum Exponents"
        || name == "Evaluate Numeric Power"
        || name == "Expand to Cancel Fraction"
}

/// Returns true when rule name corresponds to canonicalization/reordering.
pub fn is_canonicalization_rule_name(name: &str) -> bool {
    name.starts_with("Canonicalize") || name == "Collect" || name.starts_with("Sort")
}

/// Returns true when rule name corresponds to mostly mechanical grind.
pub fn is_mechanical_rule_name(name: &str) -> bool {
    if matches!(
        name,
        "Distributive Property"
            | "Distribute"
            | "Combine Like Terms"
            | "Expand"
            | "Binomial Expansion"
            | "Add Inverse"
            | "Identity Property of Addition"
    ) {
        return true;
    }

    is_canonicalization_rule_name(name)
}

/// Find `expand -> ... -> factor` no-op cycle closing index.
///
/// Returns index of the closing factor step when:
/// - `steps[start]` is treated as expansion origin,
/// - within `lookahead` steps there is a factor step whose `after` equals
///   the origin `before`.
pub fn find_expand_factor_cycle_index_by<T, E, FRuleName, FBefore, FAfter>(
    steps: &[T],
    start: usize,
    lookahead: usize,
    mut rule_name_of: FRuleName,
    mut before_of: FBefore,
    mut after_of: FAfter,
) -> Option<usize>
where
    E: Copy + Eq,
    FRuleName: FnMut(&T) -> &str,
    FBefore: FnMut(&T) -> E,
    FAfter: FnMut(&T) -> E,
{
    let start_before = before_of(&steps[start]);
    for (j, step) in steps.iter().enumerate().skip(start + 1).take(lookahead) {
        if rule_name_of(step) == "Factor" && after_of(step) == start_before {
            return Some(j);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        find_expand_factor_cycle_index_by, is_always_keep_step_rule_name,
        is_canonicalization_rule_name, is_expansion_rule_name, is_mechanical_rule_name,
    };

    #[test]
    fn name_classifiers_cover_expected_cases() {
        assert!(is_expansion_rule_name("Expand"));
        assert!(is_always_keep_step_rule_name("Sum Exponents"));
        assert!(is_always_keep_step_rule_name("Expand to Cancel Fraction"));
        assert!(is_canonicalization_rule_name("Canonicalize Terms"));
        assert!(is_mechanical_rule_name("Combine Like Terms"));
        assert!(!is_mechanical_rule_name("Quadratic Formula"));
    }

    #[test]
    fn cycle_detector_finds_matching_factor_step() {
        #[derive(Clone, Copy)]
        struct S {
            name: &'static str,
            before: i32,
            after: i32,
        }
        let steps = vec![
            S {
                name: "Expand",
                before: 10,
                after: 11,
            },
            S {
                name: "Combine Like Terms",
                before: 11,
                after: 12,
            },
            S {
                name: "Factor",
                before: 12,
                after: 10,
            },
        ];
        let idx =
            find_expand_factor_cycle_index_by(&steps, 0, 5, |s| s.name, |s| s.before, |s| s.after);
        assert_eq!(idx, Some(2));
    }
}
