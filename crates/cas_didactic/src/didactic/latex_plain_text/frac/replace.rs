pub(super) fn replace_last_fraction(
    value: &str,
    find_balanced_braces: fn(&str) -> Option<(String, usize)>,
) -> Option<String> {
    let start = value.rfind("\\frac{")?;
    let rest = &value[start + 5..];
    let (numerator, numerator_end) = find_balanced_braces(rest)?;
    let after_numerator = &rest[numerator_end + 1..];
    if !after_numerator.starts_with('{') {
        return None;
    }
    let (denominator, denominator_end) = find_balanced_braces(after_numerator)?;
    let total_end = start + 5 + numerator_end + 1 + denominator_end + 1;
    let replacement = format!("({}/{})", numerator, denominator);
    Some(format!(
        "{}{}{}",
        &value[..start],
        replacement,
        &value[total_end..]
    ))
}
