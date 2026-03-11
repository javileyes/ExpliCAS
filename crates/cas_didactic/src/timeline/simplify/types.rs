use super::VerbosityLevel;
use crate::cas_solver::{ImplicitCondition, Step};
use cas_ast::{Context, ExprId};

/// Timeline HTML generator - exports simplification steps to interactive HTML
pub struct TimelineHtml<'a> {
    pub(super) context: &'a mut Context,
    pub(super) steps: &'a [Step],
    pub(super) original_expr: ExprId,
    pub(super) simplified_result: Option<ExprId>,
    pub(super) title: String,
    pub(super) verbosity_level: VerbosityLevel,
    /// V2.12.13: Global requires inferred from input expression.
    /// Shown at the end of the timeline, after final result.
    pub(super) global_requires: Vec<ImplicitCondition>,
    /// V2.14.40: Style preferences derived from input string for consistent root rendering
    pub(super) style_prefs: cas_formatter::root_style::StylePreferences,
}
