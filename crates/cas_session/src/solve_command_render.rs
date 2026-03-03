pub use crate::solve_command_render_lines::format_solve_command_eval_lines;
pub use crate::solve_command_render_types::{
    solve_render_config_from_eval_options, SolveCommandRenderConfig,
};

#[cfg(test)]
mod tests {
    #[test]
    fn solve_render_config_from_eval_options_maps_modes() {
        let options = cas_solver::EvalOptions::default();
        let cfg = super::solve_render_config_from_eval_options(
            &options,
            crate::SetDisplayMode::Verbose,
            true,
        );
        assert!(cfg.show_steps);
        assert!(cfg.show_verbose_substeps);
        assert!(cfg.debug_mode);
    }
}
