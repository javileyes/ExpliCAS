use crate::cas_solver::Step;
use std::collections::HashSet;

pub(super) fn collect_filtered_indices(filtered_steps: &[&Step]) -> HashSet<*const Step> {
    filtered_steps
        .iter()
        .map(|step| *step as *const Step)
        .collect()
}
