use super::output::{CoreResult, UiDelta};
use super::*;

impl Repl {
    /// Core: handle set command, returns CoreResult with UI delta for verbosity changes
    pub(crate) fn handle_set_command_core(&mut self, line: &str) -> CoreResult {
        let mut ui_delta = UiDelta::default();
        match cas_solver::evaluate_set_command_input(line, self.set_command_state()) {
            cas_solver::SetCommandResult::ShowHelp { message } => {
                CoreResult::reply_only(vec![ReplMsg::output(message)])
            }
            cas_solver::SetCommandResult::ShowValue { message } => {
                CoreResult::reply_only(vec![ReplMsg::info(message)])
            }
            cas_solver::SetCommandResult::Invalid { message } => {
                CoreResult::reply_only(vec![ReplMsg::info(message)])
            }
            cas_solver::SetCommandResult::Apply { plan } => {
                self.apply_set_plan(&plan, &mut ui_delta);
                CoreResult::with_delta(vec![ReplMsg::info(plan.message)], ui_delta)
            }
        }
    }

    fn set_command_state(&self) -> cas_solver::SetCommandState {
        cas_solver::SetCommandState {
            transform: self.core.simplify_options.enable_transform,
            rationalize: self.core.simplify_options.rationalize.auto_level,
            heuristic_poly: self.core.simplify_options.shared.heuristic_poly,
            autoexpand_binomials: self.core.simplify_options.shared.autoexpand_binomials,
            steps_mode: self.core.state.options().steps_mode,
            display_mode: Self::set_display_mode_from_verbosity(self.verbosity),
            max_rewrites: self.core.simplify_options.budgets.max_total_rewrites,
            debug_mode: self.core.debug_mode,
        }
    }

    pub(crate) fn set_display_mode_from_verbosity(
        verbosity: Verbosity,
    ) -> cas_solver::SetDisplayMode {
        match verbosity {
            Verbosity::None => cas_solver::SetDisplayMode::None,
            Verbosity::Succinct => cas_solver::SetDisplayMode::Succinct,
            Verbosity::Normal => cas_solver::SetDisplayMode::Normal,
            Verbosity::Verbose => cas_solver::SetDisplayMode::Verbose,
        }
    }

    fn verbosity_from_set_display_mode(mode: cas_solver::SetDisplayMode) -> Verbosity {
        match mode {
            cas_solver::SetDisplayMode::None => Verbosity::None,
            cas_solver::SetDisplayMode::Succinct => Verbosity::Succinct,
            cas_solver::SetDisplayMode::Normal => Verbosity::Normal,
            cas_solver::SetDisplayMode::Verbose => Verbosity::Verbose,
        }
    }

    fn apply_set_plan(&mut self, plan: &cas_solver::SetCommandPlan, ui_delta: &mut UiDelta) {
        let effects = cas_solver::apply_set_command_plan(
            plan,
            &mut self.core.simplify_options,
            self.core.state.options_mut(),
            &mut self.core.debug_mode,
        );

        if let Some(mode) = effects.set_steps_mode {
            self.core.engine.simplifier.set_steps_mode(mode);
        }
        if let Some(display) = effects.set_display_mode {
            ui_delta.verbosity = Some(Self::verbosity_from_set_display_mode(display));
        }
    }

    /// Core: handle help command, returns ReplReply (no I/O)
    pub(crate) fn handle_help_core(&self, line: &str) -> ReplReply {
        let topic = match cas_solver::parse_help_command_input(line) {
            cas_solver::HelpCommandInput::General => return self.print_general_help_core(),
            cas_solver::HelpCommandInput::Topic(topic) => topic,
        };

        match cas_solver::help_topic_text(topic) {
            Some(text) => vec![ReplMsg::output(text)],
            None => {
                let mut reply = vec![ReplMsg::error(format!("Unknown command: {}", topic))];
                reply.extend(self.print_general_help_core());
                reply
            }
        }
    }

    /// Core version: returns help text as ReplReply (no I/O)
    pub(crate) fn print_general_help_core(&self) -> ReplReply {
        vec![ReplMsg::output(cas_solver::general_help_text())]
    }
}
