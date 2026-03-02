use super::*;

impl Repl {
    pub(crate) fn render_timeline_solve_eval_output(
        &mut self,
        out: cas_solver::TimelineSolveEvalOutput,
    ) -> ReplReply {
        let render = cas_didactic::render_solve_timeline_cli_output(
            &mut self.core.engine.simplifier.context,
            &out,
        );
        self.timeline_cli_render_to_reply(render)
    }

    /// Handle the 'weierstrass' command for applying Weierstrass substitution
    /// Transforms sin(x), cos(x), tan(x) into rational expressions in t = tan(x/2)
    pub(crate) fn handle_weierstrass_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_weierstrass_command_lines(&mut self.core.engine.simplifier, line)
        {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_solve_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let options = self.core.state.options().clone();
        let default_check_enabled = options.check_solutions;
        let step_verbosity = cas_solver::solve_step_verbosity_from_display_mode(
            Self::set_display_mode_from_verbosity(verbosity),
        );
        let lines = cas_solver::evaluate_solve_command_lines(
            &mut self.core.engine,
            &mut self.core.state,
            line,
            default_check_enabled,
            cas_solver::SolveCommandRenderConfig {
                show_steps: step_verbosity.show_steps,
                show_verbose_substeps: step_verbosity.show_verbose_substeps,
                requires_display: options.requires_display,
                debug_mode: self.core.debug_mode,
                hints_enabled: options.hints_enabled,
                domain_mode: options.shared.semantics.domain_mode,
                check_solutions: default_check_enabled,
            },
        );

        reply_output(lines.join("\n"))
    }
}
