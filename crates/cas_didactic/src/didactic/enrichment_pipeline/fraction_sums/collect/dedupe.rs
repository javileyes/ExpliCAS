use super::FractionSumInfo;

pub(super) fn collect_unique_primary_fraction_sums(
    all_fraction_sums: Vec<FractionSumInfo>,
    max_fractions: usize,
) -> Vec<FractionSumInfo> {
    let mut seen = std::collections::HashSet::new();

    all_fraction_sums
        .into_iter()
        .filter(|info| info.fractions.len() == max_fractions)
        .filter(|info| seen.insert(info.result.to_string()))
        .collect()
}
