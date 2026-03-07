mod balanced;
mod replace;

pub(super) fn replace_last_fraction(value: &str) -> Option<String> {
    replace::replace_last_fraction(value, balanced::find_balanced_braces)
}
