use crate::{PipelineStats, Simplifier};

/// Runtime context needed by health command adapters.
pub trait ReplHealthRuntimeContext {
    fn simplifier(&self) -> &Simplifier;
    fn simplifier_mut(&mut self) -> &mut Simplifier;
    fn health_enabled(&self) -> bool;
    fn set_health_enabled(&mut self, value: bool);
    fn last_stats(&self) -> Option<&PipelineStats>;
    fn last_health_report(&self) -> Option<&str>;
    fn clear_last_health_report(&mut self);
    fn set_last_health_report(&mut self, value: Option<String>);
}

/// Refresh last health report using current runtime simplifier and health flag.
pub fn update_health_report_on_runtime<C: ReplHealthRuntimeContext>(context: &mut C) {
    context.set_last_health_report(crate::capture_health_report_if_enabled(
        context.simplifier(),
        context.health_enabled(),
    ));
}

/// Evaluate `health ...` command and apply returned side-effects on runtime state.
pub fn evaluate_health_command_message_on_runtime<C: ReplHealthRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    let last_stats = context.last_stats().cloned();
    let last_health_report = context.last_health_report().map(str::to_string);
    let out = crate::evaluate_health_command(
        context.simplifier_mut(),
        line,
        last_stats.as_ref(),
        last_health_report.as_deref(),
    )?;

    if let Some(enabled) = out.set_enabled {
        context.set_health_enabled(enabled);
    }
    if out.clear_last_report {
        context.clear_last_health_report();
    }

    Ok(out.lines.join("\n"))
}
