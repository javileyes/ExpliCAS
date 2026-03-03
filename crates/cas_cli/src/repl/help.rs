use super::output::{CoreResult, UiDelta};
use super::*;

impl Repl {
    /// Core: handle set command, returns CoreResult with UI delta for verbosity changes
    pub(crate) fn handle_set_command_core(&mut self, line: &str) -> CoreResult {
        let mut ui_delta = UiDelta::default();
        let out = cas_session::evaluate_set_command_on_repl_core(
            line,
            &mut self.core,
            Self::set_display_mode_from_verbosity(self.verbosity),
        );
        if let Some(display) = out.set_display_mode {
            ui_delta.verbosity = Some(Self::verbosity_from_set_display_mode(display));
        }
        let reply = match out.message_kind {
            cas_session::ReplSetMessageKind::Output => vec![ReplMsg::output(out.message)],
            cas_session::ReplSetMessageKind::Info => vec![ReplMsg::info(out.message)],
        };
        if ui_delta.verbosity.is_some() {
            CoreResult::with_delta(reply, ui_delta)
        } else {
            CoreResult::reply_only(reply)
        }
    }

    pub(crate) fn set_display_mode_from_verbosity(verbosity: Verbosity) -> SetDisplayMode {
        match verbosity {
            Verbosity::None => SetDisplayMode::None,
            Verbosity::Succinct => SetDisplayMode::Succinct,
            Verbosity::Normal => SetDisplayMode::Normal,
            Verbosity::Verbose => SetDisplayMode::Verbose,
        }
    }

    fn verbosity_from_set_display_mode(mode: SetDisplayMode) -> Verbosity {
        match mode {
            SetDisplayMode::None => Verbosity::None,
            SetDisplayMode::Succinct => Verbosity::Succinct,
            SetDisplayMode::Normal => Verbosity::Normal,
            SetDisplayMode::Verbose => Verbosity::Verbose,
        }
    }

    /// Core: handle help command, returns ReplReply (no I/O)
    pub(crate) fn handle_help_core(&self, line: &str) -> ReplReply {
        let topic = match super::help_command::parse_help_command_input(line) {
            super::help_command::HelpCommandInput::General => {
                return self.print_general_help_core();
            }
            super::help_command::HelpCommandInput::Topic(topic) => topic,
        };

        match super::help_topics::help_topic_text(topic) {
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
        vec![ReplMsg::output(super::general_help::general_help_text())]
    }
}
