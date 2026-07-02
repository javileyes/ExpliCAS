mod eval;
mod runtime;

use cas_api_models::AutoexpandCommandApplyOutput;

/// Evaluate and apply an `autoexpand` command directly to runtime options.
pub(crate) fn evaluate_and_apply_autoexpand_command(
    line: &str,
    eval_options: &mut crate::EvalOptions,
) -> AutoexpandCommandApplyOutput {
    runtime::evaluate_and_apply_autoexpand_command(line, eval_options)
}
