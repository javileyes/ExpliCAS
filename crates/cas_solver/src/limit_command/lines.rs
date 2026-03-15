use crate::LimitCommandEvalOutput;
use cas_api_models::LimitCommandApproach;

pub(super) fn format_limit_command_eval_lines(output: &LimitCommandEvalOutput) -> Vec<String> {
    let dir_disp = match output.approach {
        LimitCommandApproach::Infinity => "+∞",
        LimitCommandApproach::NegInfinity => "-∞",
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
