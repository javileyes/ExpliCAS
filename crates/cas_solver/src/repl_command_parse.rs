use cas_solver_core::repl_command_types::ReplCommandInput;

pub fn parse_repl_command_input(line: &str) -> ReplCommandInput<'_> {
    if let Some(parsed) = crate::repl_command_parse_early::parse_repl_command_early(line) {
        return parsed;
    }
    crate::repl_command_parse_routing::parse_repl_command_routing(line)
}
