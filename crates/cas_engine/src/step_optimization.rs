use crate::step::Step;


pub fn optimize_steps(steps: Vec<Step>) -> Vec<Step> {
    let mut optimized = Vec::new();
    let mut i = 0;

    while i < steps.len() {
        let current = &steps[i];
        
        // Check for consecutive "Canonicalize" steps of the same type
        if is_canonicalization_rule(&current.rule_name) {
            let mut j = i + 1;
            let mut last_same_rule_idx = i;
            
            while j < steps.len() {
                let next = &steps[j];
                if next.rule_name == current.rule_name && next.path == current.path {
                    last_same_rule_idx = j;
                    j += 1;
                } else {
                    break;
                }
            }
            
            if last_same_rule_idx > i {
                // Coalesce: take the description from the LAST step (or a generic one)
                // and the 'before' from the FIRST step, and 'after' from the LAST step.
                let last = &steps[last_same_rule_idx];
                let coalesced = Step::new(
                    &last.description, // Or maybe "Canonicalize (Multiple Steps)"
                    &current.rule_name,
                    current.before,
                    last.after,
                    current.path.clone(),
                );
                optimized.push(coalesced);
                i = last_same_rule_idx + 1;
                continue;
            }
        }

        // Special case: If current is "Collect" and previous was "Canonicalize...", merge them?
        // Only if "Collect" is just reordering or grouping that Canonicalize started.
        // AND paths must match.
        if is_collect_rule(&current.rule_name) {
            if let Some(last) = optimized.last_mut() {
                if is_canonicalization_rule(&last.rule_name) && last.path == current.path {
                    // Merge Collect into previous Canonicalize
                    // We keep the previous description but update 'after'
                    last.after = current.after;
                    i += 1;
                    continue;
                }
            }
        }
        
        optimized.push(current.clone());
        i += 1;
    }

    optimized
}

fn is_canonicalization_rule(name: &str) -> bool {
    name.starts_with("Canonicalize") || name == "Collect" || name.starts_with("Sort")
}

fn is_collect_rule(name: &str) -> bool {
    name == "Collect"
}
