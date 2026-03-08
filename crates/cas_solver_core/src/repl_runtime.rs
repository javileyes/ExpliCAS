#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplSemanticsApplyOutput {
    pub message: String,
    pub rebuilt_simplifier: bool,
}
