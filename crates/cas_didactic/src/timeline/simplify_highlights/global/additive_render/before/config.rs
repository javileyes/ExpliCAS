use cas_ast::ExprPath;
use cas_formatter::{HighlightColor, PathHighlightConfig};

pub(super) fn build_before_additive_focus_config(found_paths: &[ExprPath]) -> PathHighlightConfig {
    let mut before_config = PathHighlightConfig::new();
    for path in found_paths.iter().cloned() {
        before_config.add(path, HighlightColor::Red);
    }
    before_config
}
