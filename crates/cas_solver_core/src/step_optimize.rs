//! Generic step-optimization helpers shared by runtime crates.
//!
//! The core algorithm is runtime-agnostic: detect local expand/factor no-op
//! cycles, coalesce low-importance canonicalization runs, and drop trivial
//! numeric-power evaluations.

use crate::step_rules::find_expand_factor_cycle_index_by;

/// Optimize a list of step-like items with caller-provided adapters.
///
/// The algorithm performs:
/// - no-op expand/factor cycle removal,
/// - low-importance canonicalization coalescing on same path,
/// - filtering of trivial numeric-power evaluations.
#[allow(clippy::too_many_arguments)]
pub fn optimize_steps_with_rules<
    StepT,
    Id,
    FRuleName,
    FIsExpansionRule,
    FIsCanonicalizationRule,
    FIsMediumOrHigher,
    FSamePath,
    FIsTrivialNumericPower,
    FBuildCoalesced,
    FBefore,
    FAfter,
    FGlobalBefore,
    FGlobalAfter,
>(
    steps: Vec<StepT>,
    lookahead_window: usize,
    mut rule_name: FRuleName,
    mut is_expansion_rule: FIsExpansionRule,
    mut is_canonicalization_rule: FIsCanonicalizationRule,
    mut is_medium_or_higher: FIsMediumOrHigher,
    mut same_path: FSamePath,
    mut is_trivial_numeric_power: FIsTrivialNumericPower,
    mut build_coalesced: FBuildCoalesced,
    mut before: FBefore,
    mut after: FAfter,
    mut global_before: FGlobalBefore,
    mut global_after: FGlobalAfter,
) -> Vec<StepT>
where
    StepT: Clone,
    Id: Eq + Copy,
    FRuleName: FnMut(&StepT) -> &str,
    FIsExpansionRule: FnMut(&str) -> bool,
    FIsCanonicalizationRule: FnMut(&str) -> bool,
    FIsMediumOrHigher: FnMut(&StepT) -> bool,
    FSamePath: FnMut(&StepT, &StepT) -> bool,
    FIsTrivialNumericPower: FnMut(&StepT) -> bool,
    FBuildCoalesced: FnMut(&StepT, &StepT) -> StepT,
    FBefore: FnMut(&StepT) -> Id,
    FAfter: FnMut(&StepT) -> Id,
    FGlobalBefore: FnMut(&StepT) -> Option<Id>,
    FGlobalAfter: FnMut(&StepT) -> Option<Id>,
{
    let mut optimized = Vec::new();
    let mut i = 0;

    while i < steps.len() {
        let current = &steps[i];

        if is_expansion_rule(rule_name(current)) {
            if let Some(end_idx) = find_expand_factor_cycle_index_by(
                &steps,
                i,
                lookahead_window,
                |s| rule_name(s),
                |s| before(s),
                |s| after(s),
            ) {
                let last = &steps[end_idx];
                let is_noop = match (global_before(current), global_after(last)) {
                    (Some(before_id), Some(after_id)) => before_id == after_id,
                    _ => false,
                };

                if is_noop {
                    i = end_idx + 1;
                    continue;
                }
            }
        }

        if is_canonicalization_rule(rule_name(current)) && !is_medium_or_higher(current) {
            let mut j = i + 1;
            let mut last_same_path_idx = i;

            while j < steps.len() {
                let next = &steps[j];
                if is_medium_or_higher(next) {
                    break;
                }
                if is_canonicalization_rule(rule_name(next)) && same_path(next, current) {
                    last_same_path_idx = j;
                    j += 1;
                } else {
                    break;
                }
            }

            if last_same_path_idx > i {
                let last = &steps[last_same_path_idx];
                optimized.push(build_coalesced(current, last));
                i = last_same_path_idx + 1;
                continue;
            }
        }

        if is_trivial_numeric_power(current) {
            i += 1;
            continue;
        }

        optimized.push(current.clone());
        i += 1;
    }

    optimized
}

#[cfg(test)]
mod tests {
    use super::optimize_steps_with_rules;
    use crate::step_rules::{is_canonicalization_rule_name, is_expansion_rule_name};

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct StepLite {
        rule_name: String,
        description: String,
        before: u32,
        after: u32,
        global_before: Option<u32>,
        global_after: Option<u32>,
        importance: u8, // 0=low, 1=medium
        path: Vec<u8>,
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        rule_name: &str,
        description: &str,
        before: u32,
        after: u32,
        global_before: Option<u32>,
        global_after: Option<u32>,
        importance: u8,
        path: Vec<u8>,
    ) -> StepLite {
        StepLite {
            rule_name: rule_name.to_string(),
            description: description.to_string(),
            before,
            after,
            global_before,
            global_after,
            importance,
            path,
        }
    }

    fn run(steps: Vec<StepLite>) -> Vec<StepLite> {
        optimize_steps_with_rules(
            steps,
            5,
            |s| s.rule_name.as_str(),
            is_expansion_rule_name,
            is_canonicalization_rule_name,
            |s| s.importance >= 1,
            |a, b| a.path == b.path,
            |s| {
                s.rule_name == "Evaluate Numeric Power"
                    && s.description.contains("1^")
                    && s.description.contains("-> 1")
            },
            |first, last| StepLite {
                rule_name: "Canonicalize".to_string(),
                description: "Canonicalization".to_string(),
                before: first.before,
                after: last.after,
                global_before: first.global_before,
                global_after: last.global_after,
                importance: 0,
                path: first.path.clone(),
            },
            |s| s.before,
            |s| s.after,
            |s| s.global_before,
            |s| s.global_after,
        )
    }

    #[test]
    fn removes_expand_factor_noop_cycle() {
        let steps = vec![
            step(
                "Binomial Expansion",
                "expand",
                1,
                2,
                Some(100),
                Some(200),
                0,
                vec![],
            ),
            step("Some Rule", "mid", 2, 3, Some(200), Some(300), 0, vec![]),
            step("Factor", "factor", 3, 1, Some(300), Some(100), 0, vec![]),
            step("Other", "keep", 9, 10, Some(9), Some(10), 0, vec![]),
        ];

        let out = run(steps);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].rule_name, "Other");
    }

    #[test]
    fn coalesces_low_importance_canonicalization_run() {
        let steps = vec![
            step("CanonicalizeSigns", "a", 1, 2, Some(1), Some(2), 0, vec![1]),
            step(
                "CanonicalizeFractions",
                "b",
                2,
                3,
                Some(2),
                Some(3),
                0,
                vec![1],
            ),
            step("Other", "keep", 3, 4, Some(3), Some(4), 0, vec![]),
        ];

        let out = run(steps);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].rule_name, "Canonicalize");
        assert_eq!(out[0].before, 1);
        assert_eq!(out[0].after, 3);
    }

    #[test]
    fn does_not_cross_medium_importance_barrier_when_coalescing() {
        let steps = vec![
            step("CanonicalizeSigns", "a", 1, 2, Some(1), Some(2), 0, vec![1]),
            step(
                "CanonicalizeFractions",
                "b",
                2,
                3,
                Some(2),
                Some(3),
                1,
                vec![1],
            ),
            step("CanonicalizeTerms", "c", 3, 4, Some(3), Some(4), 0, vec![1]),
        ];

        let out = run(steps);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].rule_name, "CanonicalizeSigns");
        assert_eq!(out[1].rule_name, "CanonicalizeFractions");
        assert_eq!(out[2].rule_name, "CanonicalizeTerms");
    }

    #[test]
    fn drops_trivial_numeric_power_steps() {
        let steps = vec![
            step(
                "Evaluate Numeric Power",
                "1^5 -> 1",
                1,
                1,
                Some(1),
                Some(1),
                0,
                vec![],
            ),
            step("Other", "keep", 2, 3, Some(2), Some(3), 0, vec![]),
        ];

        let out = run(steps);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].rule_name, "Other");
    }
}
