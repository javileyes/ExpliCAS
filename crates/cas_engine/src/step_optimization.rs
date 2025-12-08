use crate::step::Step;

pub fn optimize_steps(steps: Vec<Step>) -> Vec<Step> {
    let mut optimized = Vec::new();
    let mut i = 0;

    while i < steps.len() {
        let current = &steps[i];

        // Check for consecutive "Canonicalize" steps of ANY type (Canonicalize, Sort, Collect)
        // provided they operate on the SAME path.
        if is_canonicalization_rule(&current.rule_name) {
            let mut j = i + 1;
            let mut last_same_path_idx = i;

            while j < steps.len() {
                let next = &steps[j];
                // Merge if it's also a canonicalization rule AND path matches
                if is_canonicalization_rule(&next.rule_name) && next.path == current.path {
                    last_same_path_idx = j;
                    j += 1;
                } else {
                    break;
                }
            }

            if last_same_path_idx > i {
                // Coalesce:
                // Description: "Canonicalization"
                // RuleName: "Canonicalize"
                // Before: First step's before
                // After: Last step's after
                let last = &steps[last_same_path_idx];
                let coalesced = Step {
                    description: "Canonicalization".to_string(),
                    rule_name: "Canonicalize".to_string(),
                    before: current.before,
                    after: last.after,
                    path: current.path.clone(),
                    after_str: last.after_str.clone(),
                    global_before: current.global_before, // First step's global_before
                    global_after: last.global_after,      // Last step's global_after
                };
                optimized.push(coalesced);
                i = last_same_path_idx + 1;
                continue;
            }
        }

        // Special case: If current is "Collect" and previous was "Canonicalize...", merge them?
        // Only if "Collect" is just reordering or grouping that Canonicalize started.
        // AND paths must match.
        // (This logic is now covered by the general loop above if Collect is considered canonicalization)
        // But let's keep a check if we want to merge Collect into a previous *optimized* step
        // that might have been created in a previous iteration?
        // Actually, the loop above handles consecutive steps.
        // If we have [Canonicalize, Collect], the loop sees Canonicalize, looks ahead to Collect, matches path, merges.
        // So we don't need the special case anymore.

        optimized.push(current.clone());
        i += 1;
    }

    optimized
}

fn is_canonicalization_rule(name: &str) -> bool {
    name.starts_with("Canonicalize") || name == "Collect" || name.starts_with("Sort")
}
