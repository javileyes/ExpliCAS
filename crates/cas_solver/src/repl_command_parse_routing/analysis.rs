use cas_solver_core::repl_command_types::ReplCommandInput;

pub(crate) fn try_parse_analysis_command(line: &str) -> Option<ReplCommandInput<'_>> {
    if line == "equiv" || line.starts_with("equiv ") {
        return Some(ReplCommandInput::Equiv(line));
    }

    if line == "subst" || line.starts_with("subst ") {
        return Some(ReplCommandInput::Subst(line));
    }

    if line.starts_with("solve_system") {
        return Some(ReplCommandInput::SolveSystem(line));
    }

    if line == "solve" || line.starts_with("solve ") {
        return Some(ReplCommandInput::Solve(line));
    }

    if line == "simplify" || line.starts_with("simplify ") {
        return Some(ReplCommandInput::Simplify(line));
    }

    if line.starts_with("config ") {
        return Some(ReplCommandInput::Config(line));
    }

    if line == "timeline" || line.starts_with("timeline ") {
        return Some(ReplCommandInput::Timeline(line));
    }

    if line == "visualize" || line.starts_with("visualize ") || line.starts_with("viz ") {
        return Some(ReplCommandInput::Visualize(line));
    }

    if line == "explain" || line.starts_with("explain ") {
        return Some(ReplCommandInput::Explain(line));
    }

    None
}
