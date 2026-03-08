pub(super) fn format_infinite_result() -> String {
    "System has infinitely many solutions.\n\
                 The equations are dependent."
        .to_string()
}

pub(super) fn format_inconsistent_result() -> String {
    "System has no solution.\n\
                 The equations are inconsistent."
        .to_string()
}
