use cas_math::limit_types::Approach;

use crate::LimitCommandEvalOutput;

pub(super) fn format_limit_command_eval_lines(output: &LimitCommandEvalOutput) -> Vec<String> {
    let dir_disp = match output.approach {
        Approach::PosInfinity => "+∞",
        Approach::NegInfinity => "-∞",
    };
    let mut lines = vec![format!(
        "lim_{{{}→{}}} = {}",
        output.var, dir_disp, output.result
    )];
    if let Some(warning) = &output.warning {
        lines.push(format!("Warning: {}", warning));
    }
    lines
}
