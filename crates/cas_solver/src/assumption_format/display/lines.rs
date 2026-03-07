use crate::{AssumptionEvent, Step};

/// Format displayable assumption events into compact single-line strings.
///
/// Output format: `"<icon> <label>: <message>"`.
pub fn format_displayable_assumption_lines(events: &[AssumptionEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| {
            let kind = event.kind;
            if kind.should_display() {
                Some(format!(
                    "{} {}: {}",
                    kind.icon(),
                    kind.label(),
                    event.message
                ))
            } else {
                None
            }
        })
        .collect()
}

/// Format displayable assumptions emitted by one step.
pub fn format_displayable_assumption_lines_for_step(step: &Step) -> Vec<String> {
    format_displayable_assumption_lines(step.assumption_events())
}
