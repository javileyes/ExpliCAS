/// Mutable rendering state for CLI enriched sub-step blocks.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CliSubstepsRenderState {
    pub(crate) dedupe_shown: bool,
}
