use std::collections::HashSet;

use crate::Step;

pub(super) fn collect_assumed_conditions_from_steps(steps: &[Step]) -> Vec<(String, String)> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for step in steps {
        for event in step.assumption_events() {
            let fp = crate::assumption_key_dedupe_fingerprint(&event.key);
            if seen.insert(fp) {
                result.push((
                    crate::assumption_condition_text(&event.key, &event.expr_display),
                    step.rule_name.to_string(),
                ));
            }
        }
    }

    result
}
