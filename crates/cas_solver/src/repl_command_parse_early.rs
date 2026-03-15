use crate::repl_command_types::ReplCommandInput;

pub(crate) fn parse_repl_command_early(line: &str) -> Option<ReplCommandInput<'_>> {
    if line.starts_with("help") {
        return Some(ReplCommandInput::Help(line));
    }

    if let Some(rest) = line.strip_prefix("let ") {
        return Some(ReplCommandInput::Let(rest));
    }

    if let Some(idx) = line.find(":=") {
        let name = line[..idx].trim();
        let expr = line[idx + 2..].trim();
        if !name.is_empty() && !expr.is_empty() {
            return Some(ReplCommandInput::Assignment {
                name,
                expr,
                lazy: true,
            });
        }
    }

    None
}
