mod render;
mod steps;
mod visibility;

pub use self::render::format_full_simplify_eval_lines;
/// Display mode used by full-simplify rendering helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullSimplifyDisplayMode {
    None,
    Succinct,
    Normal,
    Verbose,
}
