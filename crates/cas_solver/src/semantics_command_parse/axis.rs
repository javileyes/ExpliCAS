pub(super) fn is_axis_subcommand(subcommand: &str) -> bool {
    matches!(
        subcommand,
        "domain"
            | "value"
            | "branch"
            | "inv_trig"
            | "const_fold"
            | "assumptions"
            | "assume_scope"
            | "requires"
    )
}
