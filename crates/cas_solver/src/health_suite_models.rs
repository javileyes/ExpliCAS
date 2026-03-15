use crate::health_category::Category;

/// A single health test case.
#[derive(Debug, Clone)]
pub struct HealthCase {
    /// Human-readable name.
    pub name: &'static str,
    /// Category.
    pub category: Category,
    /// Input expression.
    pub expr: &'static str,
    /// Health limits.
    pub limits: HealthLimits,
}

/// Health limits for a test case.
#[derive(Debug, Clone)]
pub struct HealthLimits {
    /// Maximum total rewrites across all phases.
    pub max_total_rewrites: usize,
    /// Maximum positive node growth.
    pub max_growth: i64,
    /// Maximum rewrites in Transform phase.
    pub max_transform_rewrites: usize,
    /// Whether cycles should cause failure (default: true).
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

/// Result of running a health case.
#[derive(Debug)]
pub struct HealthCaseResult {
    pub case: HealthCase,
    pub passed: bool,
    pub total_rewrites: usize,
    /// Per-phase rewrites.
    pub core_rewrites: usize,
    pub transform_rewrites: usize,
    pub rationalize_rewrites: usize,
    pub post_rewrites: usize,
    /// Growth metrics.
    pub growth: i64,
    pub shrink: i64,
    pub cycle_detected: Option<(cas_solver_core::simplify_phase::SimplifyPhase, usize)>,
    pub top_rules: Vec<(String, usize)>,
    pub failure_reason: Option<String>,
    /// Warning: cycle detected but not failing (forbid_cycles=false) or near limit.
    pub warning: Option<String>,
}
