use cas_api_models::ExprStatsWire;

pub(crate) struct EvalOutputResultPayload {
    pub(crate) result: String,
    pub(crate) result_truncated: bool,
    pub(crate) result_chars: usize,
    pub(crate) result_latex: Option<String>,
    pub(crate) stats: ExprStatsWire,
    pub(crate) hash: Option<String>,
}
