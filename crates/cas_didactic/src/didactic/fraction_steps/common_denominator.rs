use super::FractionSumInfo;
use num_bigint::BigInt;
use num_rational::BigRational;

pub(crate) struct CommonDenominatorData {
    /// Plain-text renderings of the original fractions (`1/2`), for CLI display.
    pub(crate) original_sum: Vec<String>,
    /// LaTeX renderings of the original fractions (`\frac{1}{2}`), for web/MathJax.
    pub(crate) original_sum_latex: Vec<String>,
    pub(crate) lcm: BigInt,
    /// Plain-text renderings of the converted fractions (`3/6`).
    pub(crate) converted: Vec<String>,
    /// LaTeX renderings of the converted fractions (`\frac{3}{6}`).
    pub(crate) converted_latex: Vec<String>,
    pub(crate) needs_conversion: bool,
}

pub(crate) fn build_common_denominator_data(
    info: &FractionSumInfo,
    format_fraction: fn(&BigRational) -> String,
    lcm_bigint: fn(&BigInt, &BigInt) -> BigInt,
) -> CommonDenominatorData {
    let original_sum: Vec<String> = info.fractions.iter().map(format_fraction_plain).collect();
    let original_sum_latex: Vec<String> = info.fractions.iter().map(format_fraction).collect();
    let lcm = info
        .fractions
        .iter()
        .fold(BigInt::from(1), |acc, f| lcm_bigint(&acc, f.denom()));
    let converted: Vec<String> = info
        .fractions
        .iter()
        .map(|fraction| {
            let multiplier = &lcm / fraction.denom();
            let new_numer = fraction.numer() * &multiplier;
            format!("{}/{}", new_numer, lcm)
        })
        .collect();
    let converted_latex: Vec<String> = info
        .fractions
        .iter()
        .map(|fraction| {
            let multiplier = &lcm / fraction.denom();
            let new_numer = fraction.numer() * &multiplier;
            format!("\\frac{{{}}}{{{}}}", new_numer, lcm)
        })
        .collect();
    let needs_conversion = info
        .fractions
        .iter()
        .any(|fraction| fraction.denom() != &lcm);

    CommonDenominatorData {
        original_sum,
        original_sum_latex,
        lcm,
        converted,
        converted_latex,
        needs_conversion,
    }
}

/// Plain-text twin of `format_fraction` (`1/2`, integers bare).
pub(crate) fn format_fraction_plain(r: &BigRational) -> String {
    r.to_string()
}
