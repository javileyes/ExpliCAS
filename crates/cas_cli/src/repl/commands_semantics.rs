use super::*;

impl Repl {
    /// Handle "semantics" command - unified control for semantic axes
    pub(crate) fn handle_semantics_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::parse_semantics_command_input(line) {
            cas_solver::SemanticsCommandInput::Show => {
                // Just "semantics" - show current settings
                self.print_semantics_core()
            }
            cas_solver::SemanticsCommandInput::Help => self.print_semantics_help_core(),
            cas_solver::SemanticsCommandInput::Set { args } => {
                // Parse remaining args as axis=value pairs or axis value pairs
                let refs: Vec<&str> = args.iter().map(String::as_str).collect();
                self.parse_semantics_set_core(&refs)
            }
            cas_solver::SemanticsCommandInput::Axis { axis } => self.print_axis_status_core(&axis),
            cas_solver::SemanticsCommandInput::Preset { args } => {
                let refs: Vec<&str> = args.iter().map(String::as_str).collect();
                self.handle_preset_core(&refs)
            }
            cas_solver::SemanticsCommandInput::Unknown { subcommand } => reply_output(
                cas_solver::format_semantics_unknown_subcommand_message(&subcommand),
            ),
        }
    }

    pub(crate) fn print_semantics_core(&self) -> ReplReply {
        let state = cas_solver::semantics_view_state_from_options(
            &self.core.simplify_options,
            self.core.state.options(),
        );
        let lines = cas_solver::format_semantics_overview_lines(&state);
        reply_output(lines.join("\n"))
    }

    /// Print status for a single semantic axis with current value and available options
    pub(crate) fn print_axis_status_core(&self, axis: &str) -> ReplReply {
        let state = cas_solver::semantics_view_state_from_options(
            &self.core.simplify_options,
            self.core.state.options(),
        );
        let lines = cas_solver::format_semantics_axis_lines(&state, axis);
        reply_output(lines.join("\n"))
    }

    pub(crate) fn print_semantics_help_core(&self) -> ReplReply {
        reply_output(cas_solver::semantics_help_message())
    }

    /// Handle "semantics preset" subcommand
    pub(crate) fn handle_preset_core(&mut self, args: &[&str]) -> ReplReply {
        let lines = match args.first() {
            None => cas_solver::format_semantics_preset_list_lines(),
            Some(&"help") => cas_solver::format_semantics_preset_help_lines(args.get(1).copied()),
            Some(name) => {
                let current = cas_solver::semantics_preset_state_from_options(
                    &self.core.simplify_options,
                    self.core.state.options(),
                );
                match cas_solver::apply_semantics_preset_by_name(name) {
                    Ok(application) => {
                        let next = application.next;
                        cas_solver::apply_semantics_preset_state_to_options(
                            next,
                            &mut self.core.simplify_options,
                            self.core.state.options_mut(),
                        );

                        self.sync_config_to_simplifier();

                        cas_solver::format_semantics_preset_application_lines(current, &application)
                    }
                    Err(cas_solver::SemanticsPresetApplyError::UnknownPreset { .. }) => {
                        cas_solver::format_semantics_preset_help_lines(Some(name))
                    }
                }
            }
        };

        reply_output(lines.join("\n"))
    }
}
