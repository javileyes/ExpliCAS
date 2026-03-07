use super::FractionSumInfo;

pub(super) fn max_fraction_count(all_fraction_sums: &[FractionSumInfo]) -> usize {
    all_fraction_sums
        .iter()
        .map(|sum| sum.fractions.len())
        .max()
        .unwrap_or(0)
}
