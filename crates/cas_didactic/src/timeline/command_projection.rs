use super::types::{
    TimelineCommandOutput, TimelineSimplifyCommandOutput, TimelineSolveCommandOutput,
};

/// Convert solver timeline evaluation output into didactic render payload.
pub fn timeline_command_output_from_solver(
    output: cas_session::TimelineCommandEvalOutput,
) -> TimelineCommandOutput {
    match output {
        cas_session::TimelineCommandEvalOutput::Solve(out) => {
            TimelineCommandOutput::Solve(TimelineSolveCommandOutput {
                equation: out.equation,
                var: out.var,
                solution_set: out.solution_set,
                display_steps: out.display_steps,
            })
        }
        cas_session::TimelineCommandEvalOutput::Simplify {
            expr_input,
            aggressive,
            output,
        } => TimelineCommandOutput::Simplify(TimelineSimplifyCommandOutput {
            expr_input,
            use_aggressive: aggressive,
            parsed_expr: output.parsed_expr,
            simplified_expr: output.simplified_expr,
            steps: output.steps,
        }),
    }
}
