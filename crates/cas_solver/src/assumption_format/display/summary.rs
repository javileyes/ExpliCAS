use crate::AssumptionRecord;

fn assumption_record_summary_item(record: &AssumptionRecord) -> String {
    if record.count > 1 {
        format!("{}({}) (×{})", record.kind, record.expr, record.count)
    } else {
        format!("{}({})", record.kind, record.expr)
    }
}

/// Format assumptions summary payload for REPL/UI.
///
/// Returns only the right side content (without the `⚠ Assumptions:` prefix).
pub fn format_assumption_records_summary(records: &[AssumptionRecord]) -> Option<String> {
    if records.is_empty() {
        return None;
    }
    let items: Vec<String> = records.iter().map(assumption_record_summary_item).collect();
    Some(items.join(", "))
}
