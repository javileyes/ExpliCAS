/// Aggregated assumption record produced by solver flows.
///
/// This mirrors engine payload shape but is owned by `cas_solver` so consumers
/// don't need to depend on `cas_engine` domain types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssumptionRecord {
    pub kind: String,
    pub expr: String,
    pub message: String,
    pub count: u32,
}

pub(crate) fn assumption_record_from_engine(
    value: cas_engine::AssumptionRecord,
) -> AssumptionRecord {
    AssumptionRecord {
        kind: value.kind,
        expr: value.expr,
        message: value.message,
        count: value.count,
    }
}
