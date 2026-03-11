use super::SubStep;
use crate::runtime::Step;

/// An enriched step with optional sub-steps for didactic explanation
#[derive(Debug, Clone)]
pub struct EnrichedStep {
    /// The original step from the engine
    pub base_step: Step,
    /// Synthetic sub-steps that explain hidden operations
    pub sub_steps: Vec<SubStep>,
}
