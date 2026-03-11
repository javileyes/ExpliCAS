use super::types::{
    TimelineCommandOutput, TimelineSimplifyCommandOutput, TimelineSolveCommandOutput,
};
use cas_solver::session_api::types::TimelineCommandEvalOutput;

/// Convert solver timeline evaluation output into didactic render payload.
pub fn timeline_command_output_from_solver(
    output: TimelineCommandEvalOutput,
) -> TimelineCommandOutput {
    match output {
        TimelineCommandEvalOutput::Solve(out) => {
            TimelineCommandOutput::Solve(TimelineSolveCommandOutput {
                equation: out.equation,
                var: out.var,
                solution_set: out.solution_set,
                display_steps: out.display_steps,
            })
        }
        TimelineCommandEvalOutput::Simplify {
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
