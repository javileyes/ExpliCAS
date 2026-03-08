use super::FractionSumInfo;
use num_bigint::BigInt;
use num_rational::BigRational;

pub(crate) struct CommonDenominatorData {
    pub(crate) original_sum: Vec<String>,
    pub(crate) lcm: BigInt,
    pub(crate) converted: Vec<String>,
    pub(crate) needs_conversion: bool,
}

pub(crate) fn build_common_denominator_data(
    info: &FractionSumInfo,
    format_fraction: fn(&BigRational) -> String,
    lcm_bigint: fn(&BigInt, &BigInt) -> BigInt,
) -> CommonDenominatorData {
    let original_sum: Vec<String> = info.fractions.iter().map(format_fraction).collect();
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
            format!("\\frac{{{}}}{{{}}}", new_numer, lcm)
        })
        .collect();
    let needs_conversion = info
        .fractions
        .iter()
        .any(|fraction| fraction.denom() != &lcm);

    CommonDenominatorData {
        original_sum,
        lcm,
        converted,
        needs_conversion,
    }
}
