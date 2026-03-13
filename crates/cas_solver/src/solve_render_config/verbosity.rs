use super::{SolveCommandRenderConfig, SolveDisplayMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SolveStepVerbosity {
    show_steps: bool,
    show_verbose_substeps: bool,
}

fn solve_step_verbosity_from_display_mode(mode: SolveDisplayMode) -> SolveStepVerbosity {
    match mode {
        SolveDisplayMode::None => SolveStepVerbosity {
            show_steps: false,
            show_verbose_substeps: false,
        },
        SolveDisplayMode::Succinct | SolveDisplayMode::Normal => SolveStepVerbosity {
            show_steps: true,
            show_verbose_substeps: false,
        },
        SolveDisplayMode::Verbose => SolveStepVerbosity {
            show_steps: true,
            show_verbose_substeps: true,
        },
    }
}

pub fn solve_render_config_from_eval_options(
    options: &crate::EvalOptions,
    display_mode: SolveDisplayMode,
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
