#[cfg(test)]
mod tests {
    use crate::{solve_render_config_from_eval_options, SetDisplayMode};

    #[test]
    fn solve_render_config_from_eval_options_maps_modes() {
        let options = cas_solver::EvalOptions::default();
        let cfg = solve_render_config_from_eval_options(&options, SetDisplayMode::Verbose, true);
        assert!(cfg.show_steps);
        assert!(cfg.show_verbose_substeps);
        assert!(cfg.debug_mode);
    }
}
