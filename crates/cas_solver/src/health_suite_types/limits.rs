/// Health limits for a test case
#[derive(Debug, Clone)]
pub struct HealthLimits {
    /// Maximum total rewrites across all phases
    pub max_total_rewrites: usize,
    /// Maximum positive node growth
    pub max_growth: i64,
    /// Maximum rewrites in Transform phase
    pub max_transform_rewrites: usize,
    /// Whether cycles should cause failure (default: true)
    pub forbid_cycles: bool,
}

impl Default for HealthLimits {
    fn default() -> Self {
        Self {
            max_total_rewrites: 100,
            max_growth: 200,
            max_transform_rewrites: 50,
            forbid_cycles: true,
        }
    }
}
