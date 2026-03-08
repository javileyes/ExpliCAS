use crate::{AssumptionEvent, AssumptionKind, Step};

/// Group displayable assumption events by kind for compact timeline/HTML presentation.
///
/// Output format: one line per kind in stable order:
/// `Requires`, `Branch`, `Domain`, `Assumes`.
pub fn format_displayable_assumption_lines_grouped(events: &[AssumptionEvent]) -> Vec<String> {
    let mut requires = Vec::new();
    let mut branches = Vec::new();
    let mut domain_ext = Vec::new();
    let mut assumes = Vec::new();

    for event in events.iter().filter(|event| event.kind.should_display()) {
        match event.kind {
            AssumptionKind::RequiresIntroduced => requires.push(event.message.clone()),
            AssumptionKind::BranchChoice => branches.push(event.message.clone()),
            AssumptionKind::DomainExtension => domain_ext.push(event.message.clone()),
            AssumptionKind::HeuristicAssumption => assumes.push(event.message.clone()),
            AssumptionKind::DerivedFromRequires => {}
        }
    }

    let mut lines = Vec::new();
    if !requires.is_empty() {
        lines.push(format!(
            "{} {}: {}",
            AssumptionKind::RequiresIntroduced.icon(),
            AssumptionKind::RequiresIntroduced.label(),
            requires.join(", ")
        ));
    }
    if !branches.is_empty() {
        lines.push(format!(
            "{} {}: {}",
            AssumptionKind::BranchChoice.icon(),
            AssumptionKind::BranchChoice.label(),
            branches.join(", ")
        ));
    }
    if !domain_ext.is_empty() {
        lines.push(format!(
            "{} {}: {}",
            AssumptionKind::DomainExtension.icon(),
            AssumptionKind::DomainExtension.label(),
            domain_ext.join(", ")
        ));
    }
    if !assumes.is_empty() {
        lines.push(format!(
            "{} {}: {}",
            AssumptionKind::HeuristicAssumption.icon(),
            AssumptionKind::HeuristicAssumption.label(),
            assumes.join(", ")
        ));
    }
    lines
}

/// Group displayable assumptions emitted by one step.
pub fn format_displayable_assumption_lines_grouped_for_step(step: &Step) -> Vec<String> {
    format_displayable_assumption_lines_grouped(step.assumption_events())
}
