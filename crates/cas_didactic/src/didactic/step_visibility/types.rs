/// Shared visibility policy for step-oriented didactic frontends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepVisibility {
    All,
    MediumOrHigher,
    HighOrHigher,
}
