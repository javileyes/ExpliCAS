#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveCommandRenderConfig {
    pub show_steps: bool,
    pub show_verbose_substeps: bool,
    pub requires_display: cas_solver::RequiresDisplayLevel,
    pub debug_mode: bool,
    pub hints_enabled: bool,
    pub domain_mode: cas_solver::DomainMode,
    pub check_solutions: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SolveStepVerbosity {
    show_steps: bool,
    show_verbose_substeps: bool,
}

fn solve_step_verbosity_from_display_mode(mode: crate::SetDisplayMode) -> SolveStepVerbosity {
    match mode {
        crate::SetDisplayMode::None => SolveStepVerbosity {
            show_steps: false,
            show_verbose_substeps: false,
        },
        crate::SetDisplayMode::Succinct | crate::SetDisplayMode::Normal => SolveStepVerbosity {
            show_steps: true,
            show_verbose_substeps: false,
        },
        crate::SetDisplayMode::Verbose => SolveStepVerbosity {
            show_steps: true,
            show_verbose_substeps: true,
        },
    }
}

pub fn solve_render_config_from_eval_options(
    options: &cas_solver::EvalOptions,
    display_mode: crate::SetDisplayMode,
    debug_mode: bool,
) -> SolveCommandRenderConfig {
    let step_verbosity = solve_step_verbosity_from_display_mode(display_mode);
    SolveCommandRenderConfig {
        show_steps: step_verbosity.show_steps,
        show_verbose_substeps: step_verbosity.show_verbose_substeps,
        requires_display: options.requires_display,
        debug_mode,
        hints_enabled: options.hints_enabled,
        domain_mode: options.shared.semantics.domain_mode,
        check_solutions: options.check_solutions,
    }
}
