//! Step visualization - pure output generation via ReplReply
//!
//! This module generates step-by-step output as strings, no direct I/O.

use super::output::{reply_output, ReplReply};
use super::*;

impl Repl {
    fn didactic_display_mode_from_repl(mode: SetDisplayMode) -> cas_didactic::StepDisplayMode {
        match mode {
            SetDisplayMode::None => cas_didactic::StepDisplayMode::None,
            SetDisplayMode::Succinct => cas_didactic::StepDisplayMode::Succinct,
            SetDisplayMode::Normal => cas_didactic::StepDisplayMode::Normal,
            SetDisplayMode::Verbose => cas_didactic::StepDisplayMode::Verbose,
        }
    }

    /// Show simplification steps - returns ReplReply (no I/O)
    pub(crate) fn show_simplification_steps_core(
        &mut self,
        expr: cas_ast::ExprId,
        steps: &[cas_solver::runtime::Step],
        style_signals: cas_formatter::root_style::ParseStyleSignals,
    ) -> ReplReply {
        let lines = self.core.with_simplifier_mut(|simplifier| {
            cas_didactic::format_cli_simplification_steps_with_simplifier(
                simplifier,
                expr,
                steps,
                style_signals,
                Self::didactic_display_mode_from_repl(Self::set_display_mode_from_verbosity(
                    self.verbosity,
                )),
            )
        });
        if lines.is_empty() {
            vec![]
        } else {
            reply_output(lines.join("\n"))
        }
    }
}
