/// Display mode used by full-simplify rendering helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullSimplifyDisplayMode {
    None,
    Succinct,
    Normal,
    Verbose,
}
