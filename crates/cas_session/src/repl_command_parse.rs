use crate::repl_command_types::ReplCommandInput;

pub fn parse_repl_command_input(line: &str) -> ReplCommandInput<'_> {
    if line.starts_with("help") {
        return ReplCommandInput::Help(line);
    }

    if let Some(rest) = line.strip_prefix("let ") {
        return ReplCommandInput::Let(rest);
    }

    if let Some(idx) = line.find(":=") {
        let name = line[..idx].trim();
        let expr = line[idx + 2..].trim();
        if !name.is_empty() && !expr.is_empty() {
            return ReplCommandInput::Assignment {
                name,
                expr,
                lazy: true,
            };
        }
    }

    if line == "vars" {
        return ReplCommandInput::Vars;
    }

    if line == "clear" || line.starts_with("clear ") {
        return ReplCommandInput::Clear(line);
    }

    if line == "reset" {
        return ReplCommandInput::Reset;
    }

    if line == "reset full" {
        return ReplCommandInput::ResetFull;
    }

    if line == "cache clear" || line == "cache" {
        return ReplCommandInput::Cache(line);
    }

    if line == "semantics" || line.starts_with("semantics ") {
        return ReplCommandInput::Semantics(line);
    }

    if line == "context" || line.starts_with("context ") {
        return ReplCommandInput::Context(line);
    }

    if line == "steps" || line.starts_with("steps ") {
        return ReplCommandInput::Steps(line);
    }

    if line == "autoexpand" || line.starts_with("autoexpand ") {
        return ReplCommandInput::Autoexpand(line);
    }

    if line == "budget" || line.starts_with("budget ") {
        return ReplCommandInput::Budget(line);
    }

    if line == "history" || line == "list" {
        return ReplCommandInput::History;
    }

    if let Some(rest) = line.strip_prefix("show ") {
        return ReplCommandInput::Show(rest);
    }

    if let Some(rest) = line.strip_prefix("del ") {
        return ReplCommandInput::Del(rest);
    }

    if line.starts_with("set ") {
        return ReplCommandInput::Set(line);
    }

    if line == "equiv" || line.starts_with("equiv ") {
        return ReplCommandInput::Equiv(line);
    }

    if line == "subst" || line.starts_with("subst ") {
        return ReplCommandInput::Subst(line);
    }

    if line.starts_with("solve_system") {
        return ReplCommandInput::SolveSystem(line);
    }

    if line == "solve" || line.starts_with("solve ") {
        return ReplCommandInput::Solve(line);
    }

    if line == "simplify" || line.starts_with("simplify ") {
        return ReplCommandInput::Simplify(line);
    }

    if line.starts_with("config ") {
        return ReplCommandInput::Config(line);
    }

    if line == "timeline" || line.starts_with("timeline ") {
        return ReplCommandInput::Timeline(line);
    }

    if line == "visualize" || line.starts_with("visualize ") || line.starts_with("viz ") {
        return ReplCommandInput::Visualize(line);
    }

    if line == "explain" || line.starts_with("explain ") {
        return ReplCommandInput::Explain(line);
    }

    if line == "det" || line.starts_with("det ") {
        return ReplCommandInput::Det(line);
    }

    if line == "transpose" || line.starts_with("transpose ") {
        return ReplCommandInput::Transpose(line);
    }

    if line == "trace" || line.starts_with("trace ") {
        return ReplCommandInput::Trace(line);
    }

    if line == "telescope" || line.starts_with("telescope ") {
        return ReplCommandInput::Telescope(line);
    }

    if line == "weierstrass" || line.starts_with("weierstrass ") {
        return ReplCommandInput::Weierstrass(line);
    }

    if line.starts_with("expand_log ") || line == "expand_log" {
        return ReplCommandInput::ExpandLog(line);
    }

    if line == "expand" || line.starts_with("expand ") {
        return ReplCommandInput::Expand(line);
    }

    if line == "rationalize" || line.starts_with("rationalize ") {
        return ReplCommandInput::Rationalize(line);
    }

    if line == "limit" || line.starts_with("limit ") {
        return ReplCommandInput::Limit(line);
    }

    if line.starts_with("profile") {
        return ReplCommandInput::Profile(line);
    }

    if line.starts_with("health") {
        return ReplCommandInput::Health(line);
    }

    ReplCommandInput::Eval(line)
}
