use cas_solver_core::repl_command_types::ReplCommandInput;

pub(crate) fn try_parse_session_command(line: &str) -> Option<ReplCommandInput<'_>> {
    if line == "vars" {
        return Some(ReplCommandInput::Vars);
    }

    if line == "clear" || line.starts_with("clear ") {
        return Some(ReplCommandInput::Clear(line));
    }

    if line == "reset" {
        return Some(ReplCommandInput::Reset);
    }

    if line == "reset full" {
        return Some(ReplCommandInput::ResetFull);
    }

    if line == "cache clear" || line == "cache" {
        return Some(ReplCommandInput::Cache(line));
    }

    if line == "semantics" || line.starts_with("semantics ") {
        return Some(ReplCommandInput::Semantics(line));
    }

    if line == "context" || line.starts_with("context ") {
        return Some(ReplCommandInput::Context(line));
    }

    if line == "steps" || line.starts_with("steps ") {
        return Some(ReplCommandInput::Steps(line));
    }

    if line == "autoexpand" || line.starts_with("autoexpand ") {
        return Some(ReplCommandInput::Autoexpand(line));
    }

    if line == "budget" || line.starts_with("budget ") {
        return Some(ReplCommandInput::Budget(line));
    }

    if line == "history" || line == "list" {
        return Some(ReplCommandInput::History);
    }

    if let Some(rest) = line.strip_prefix("show ") {
        return Some(ReplCommandInput::Show(rest));
    }

    if let Some(rest) = line.strip_prefix("del ") {
        return Some(ReplCommandInput::Del(rest));
    }

    if line.starts_with("set ") {
        return Some(ReplCommandInput::Set(line));
    }

    None
}
