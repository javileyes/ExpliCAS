mod assumptions;
mod hints;
mod requires;
mod warnings;

use crate::command_api::eval::{EvalCommandEvalView, EvalMetadataLines};

pub(crate) fn format_eval_metadata_lines(
    context: &mut cas_ast::Context,
    output: &EvalCommandEvalView,
    requires_display: crate::RequiresDisplayLevel,
    debug_mode: bool,
    hints_enabled: bool,
    domain_mode: crate::DomainMode,
    assumption_reporting: crate::AssumptionReporting,
) -> EvalMetadataLines {
    EvalMetadataLines {
        warning_lines: warnings::format_warning_lines(output),
        requires_lines: requires::format_requires_lines(
            context,
            output,
            requires_display,
            debug_mode,
        ),
        hint_lines: hints::format_hint_lines(context, output, hints_enabled, domain_mode),
        assumption_lines: assumptions::format_assumption_lines(output, assumption_reporting),
    }
}
