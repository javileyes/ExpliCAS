/// Rendering hints for CLI sub-step blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CliSubstepsRenderPlan {
    /// Optional category header to display before sub-steps.
    pub header: Option<&'static str>,
    /// If true, this block should be shown only once (deduplicated across steps).
    pub dedupe_once: bool,
}

/// Rendering hints for timeline sub-step blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineSubstepsRenderPlan {
    /// Category header to display before sub-steps.
    pub header: &'static str,
    /// If true, this block should be shown only once globally.
    pub dedupe_once: bool,
}
