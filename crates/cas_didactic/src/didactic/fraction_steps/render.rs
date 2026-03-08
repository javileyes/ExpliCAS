use super::common_denominator::CommonDenominatorData;
use super::{FractionSumInfo, SubStep};
use crate::didactic::format_fraction;

pub(crate) fn render_fraction_sum_substeps(
    info: &FractionSumInfo,
    common: CommonDenominatorData,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if common.needs_conversion {
        sub_steps.push(SubStep {
            description: format!("Find common denominator: {}", common.lcm),
            before_expr: common.original_sum.join(" + "),
            after_expr: common.converted.join(" + "),
            before_latex: None,
            after_latex: None,
        });
    }

    sub_steps.push(SubStep {
        description: "Sum the fractions".to_string(),
        before_expr: if common.needs_conversion {
            common.converted.join(" + ")
        } else {
            common.original_sum.join(" + ")
        },
        after_expr: format_fraction(&info.result),
        before_latex: None,
        after_latex: None,
    });

    sub_steps
}
