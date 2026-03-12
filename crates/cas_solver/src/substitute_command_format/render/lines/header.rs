use crate::substitute::SubstituteSimplifyEvalOutput;
use crate::substitute_command_parse::split_by_comma_ignoring_parens;
use cas_solver_core::substitute_command_types::SubstituteRenderMode;

pub(super) fn build_header_lines(
    input: &str,
    output: &SubstituteSimplifyEvalOutput,
    mode: SubstituteRenderMode,
) -> Vec<String> {
    let display_parts = split_by_comma_ignoring_parens(input);
    let expr_str = display_parts.first().map(|s| s.trim()).unwrap_or_default();
    let target_str = display_parts.get(1).map(|s| s.trim()).unwrap_or_default();
    let replacement_str = display_parts.get(2).map(|s| s.trim()).unwrap_or_default();

    let mut lines = Vec::new();
    if mode != SubstituteRenderMode::None {
        let label = match output.strategy {
            crate::SubstituteStrategy::Variable => "Variable substitution",
            crate::SubstituteStrategy::PowerAware => "Expression substitution",
        };
        lines.push(format!(
            "{label}: {} → {} in {}",
            target_str, replacement_str, expr_str
        ));
    }
    lines
}
