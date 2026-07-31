use super::common_denominator::{format_fraction_plain, CommonDenominatorData};
use super::{FractionSumInfo, SubStep};
use crate::didactic::format_fraction;

pub(crate) fn render_fraction_sum_substeps(
    info: &FractionSumInfo,
    common: CommonDenominatorData,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if common.needs_conversion {
        sub_steps.push(
            SubStep::keyed(
                "fraction.find_common_denominator",
                vec![common.lcm.to_string()],
                common.original_sum.join(" + "),
                common.converted.join(" + "),
            )
            .with_before_latex(common.original_sum_latex.join(" + "))
            .with_after_latex(common.converted_latex.join(" + ")),
        );
    }

    let (summed_plain, summed_latex) = if common.needs_conversion {
        (
            common.converted.join(" + "),
            common.converted_latex.join(" + "),
        )
    } else {
        (
            common.original_sum.join(" + "),
            common.original_sum_latex.join(" + "),
        )
    };
    sub_steps.push(
        SubStep::keyed(
            "fraction.sum_fractions",
            Vec::new(),
            summed_plain,
            format_fraction_plain(&info.result),
        )
        .with_before_latex(summed_latex)
        .with_after_latex(format_fraction(&info.result)),
    );

    sub_steps
}
