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

impl AssumptionRecord {
    /// Converts to the engine representation.
    #[inline]
    pub fn into_engine(self) -> cas_engine::AssumptionRecord {
        self.into()
    }
}

impl From<cas_engine::AssumptionRecord> for AssumptionRecord {
    fn from(value: cas_engine::AssumptionRecord) -> Self {
        Self {
            kind: value.kind,
            expr: value.expr,
            message: value.message,
            count: value.count,
        }
    }
}

impl From<AssumptionRecord> for cas_engine::AssumptionRecord {
    fn from(value: AssumptionRecord) -> Self {
        Self {
            kind: value.kind,
            expr: value.expr,
            message: value.message,
            count: value.count,
        }
    }
}
