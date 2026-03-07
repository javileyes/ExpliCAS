use super::super::FractionSumInfo;

pub(super) fn push_fraction_sum_if_present(
    info: Option<FractionSumInfo>,
    results: &mut Vec<FractionSumInfo>,
) {
    if let Some(info) = info {
        results.push(info);
    }
}
