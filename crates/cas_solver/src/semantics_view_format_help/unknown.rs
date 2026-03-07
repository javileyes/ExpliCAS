pub fn format_semantics_unknown_subcommand_message(subcommand: &str) -> String {
    format!(
        "Unknown semantics subcommand: '{}'\n\
         Usage: semantics [set|preset|help|<axis>]\n\
           semantics            Show all settings\n\
           semantics <axis>     Show one axis (domain|value|branch|inv_trig|const_fold|assumptions|assume_scope|requires)\n\
           semantics help       Show help\n\
           semantics set ...    Change settings\n\
           semantics preset     List/apply presets",
        subcommand
    )
}
