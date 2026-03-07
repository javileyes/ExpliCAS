use crate::repl_command_types::ReplCommandInput;

pub(crate) fn try_parse_algebra_command(line: &str) -> Option<ReplCommandInput<'_>> {
    if line == "det" || line.starts_with("det ") {
        return Some(ReplCommandInput::Det(line));
    }

    if line == "transpose" || line.starts_with("transpose ") {
        return Some(ReplCommandInput::Transpose(line));
    }

    if line == "trace" || line.starts_with("trace ") {
        return Some(ReplCommandInput::Trace(line));
    }

    if line == "telescope" || line.starts_with("telescope ") {
        return Some(ReplCommandInput::Telescope(line));
    }

    if line == "weierstrass" || line.starts_with("weierstrass ") {
        return Some(ReplCommandInput::Weierstrass(line));
    }

    if line.starts_with("expand_log ") || line == "expand_log" {
        return Some(ReplCommandInput::ExpandLog(line));
    }

    if line == "expand" || line.starts_with("expand ") {
        return Some(ReplCommandInput::Expand(line));
    }

    if line == "rationalize" || line.starts_with("rationalize ") {
        return Some(ReplCommandInput::Rationalize(line));
    }

    if line == "limit" || line.starts_with("limit ") {
        return Some(ReplCommandInput::Limit(line));
    }

    if line.starts_with("profile") {
        return Some(ReplCommandInput::Profile(line));
    }

    if line.starts_with("health") {
        return Some(ReplCommandInput::Health(line));
    }

    None
}
