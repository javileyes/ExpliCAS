//! Generic helpers for step-absorption patterns.
//!
//! These utilities are runtime-agnostic and operate on caller-provided
//! predicates (marker step, absorbable step, barrier step).

use std::collections::HashSet;

/// Find indices to absorb when a marker step appears.
///
/// For each marker at index `j`, this scans backwards up to `window` steps:
/// - stops at first barrier,
/// - absorbs consecutive absorbable steps,
/// - stops at first non-absorbable/non-barrier step.
pub fn find_absorption_indices_before_markers_with<T, FIsMarker, FIsAbsorbable, FIsBarrier>(
    steps: &[T],
    window: usize,
    mut is_marker: FIsMarker,
    mut is_absorbable: FIsAbsorbable,
    mut is_barrier: FIsBarrier,
) -> Vec<usize>
where
    FIsMarker: FnMut(&T) -> bool,
    FIsAbsorbable: FnMut(&T) -> bool,
    FIsBarrier: FnMut(&T) -> bool,
{
    let mut to_absorb = Vec::new();

    for (j, step) in steps.iter().enumerate() {
        if !is_marker(step) {
            continue;
        }

        let window_start = j.saturating_sub(window);
        for i in (window_start..j).rev() {
            let s = &steps[i];
            if is_barrier(s) {
                break;
            }
            if !is_absorbable(s) {
                break;
            }
            to_absorb.push(i);
        }
    }

    to_absorb
}

/// Apply index-based absorption to a step vector.
pub fn absorb_indices<T>(steps: Vec<T>, indices: &[usize]) -> Vec<T> {
    let set: HashSet<usize> = indices.iter().copied().collect();
    steps
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !set.contains(i))
        .map(|(_, s)| s)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{absorb_indices, find_absorption_indices_before_markers_with};

    #[derive(Clone, Copy)]
    enum Kind {
        Mechanical,
        Marker,
        Barrier,
        Other,
    }

    #[test]
    fn finds_absorption_indices_with_barrier_stop() {
        let steps = vec![
            Kind::Mechanical, // 0
            Kind::Mechanical, // 1
            Kind::Barrier,    // 2
            Kind::Mechanical, // 3
            Kind::Marker,     // 4 -> should absorb only index 3
        ];

        let indices = find_absorption_indices_before_markers_with(
            &steps,
            8,
            |s| matches!(s, Kind::Marker),
            |s| matches!(s, Kind::Mechanical),
            |s| matches!(s, Kind::Barrier),
        );

        assert_eq!(indices, vec![3]);
    }

    #[test]
    fn absorb_indices_removes_marked_positions() {
        let steps = vec![10, 20, 30, 40];
        let out = absorb_indices(steps, &[1, 3]);
        assert_eq!(out, vec![10, 30]);
    }

    #[test]
    fn stops_on_non_absorbable_non_barrier_step() {
        let steps = vec![
            Kind::Mechanical, // 0
            Kind::Other,      // 1
            Kind::Mechanical, // 2
            Kind::Marker,     // 3 -> should absorb only index 2
        ];

        let indices = find_absorption_indices_before_markers_with(
            &steps,
            8,
            |s| matches!(s, Kind::Marker),
            |s| matches!(s, Kind::Mechanical),
            |s| matches!(s, Kind::Barrier),
        );

        assert_eq!(indices, vec![2]);
    }
}
