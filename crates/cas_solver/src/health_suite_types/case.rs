use super::{Category, HealthLimits};

/// A single health test case
#[derive(Debug, Clone)]
pub struct HealthCase {
    /// Human-readable name
    pub name: &'static str,
    /// Category
    pub category: Category,
    /// Input expression
    pub expr: &'static str,
    /// Health limits
    pub limits: HealthLimits,
}
