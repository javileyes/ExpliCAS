use crate::SimplifyPhase;

use super::HealthCase;

/// Result of running a health case
#[derive(Debug)]
pub struct HealthCaseResult {
    pub case: HealthCase,
    pub passed: bool,
    pub total_rewrites: usize,
    /// Per-phase rewrites
    pub core_rewrites: usize,
    pub transform_rewrites: usize,
    pub rationalize_rewrites: usize,
    pub post_rewrites: usize,
    /// Growth metrics
    pub growth: i64,
    pub shrink: i64,
    pub cycle_detected: Option<(SimplifyPhase, usize)>,
    pub top_rules: Vec<(String, usize)>,
    pub failure_reason: Option<String>,
    /// Warning: cycle detected but not failing (forbid_cycles=false) or near limit
    pub warning: Option<String>,
}
