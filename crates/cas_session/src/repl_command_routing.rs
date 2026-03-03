/// Parsed top-level REPL command classification.
///
/// This keeps command routing logic outside the CLI transport layer so the
/// REPL remains a thin adapter.
pub fn preprocess_repl_function_syntax(line: &str) -> String {
    let line = line.trim();

    if line.starts_with("simplify(") && line.ends_with(')') {
        let content = &line["simplify(".len()..line.len() - 1];
        return format!("simplify {}", content);
    }

    if line.starts_with("solve(") && line.ends_with(')') {
        let content = &line["solve(".len()..line.len() - 1];
        return format!("solve {}", content);
    }

    line.to_string()
}

/// Split a raw REPL line into executable statements.
///
/// Keeps `solve_system ...` as a single statement because semicolons are part
/// of that command syntax.
pub fn split_repl_statements(line: &str) -> Vec<&str> {
    if line.starts_with("solve_system") {
        return vec![line];
    }

    line.split(';')
        .map(str::trim)
        .filter(|stmt| !stmt.is_empty())
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplCommandInput<'a> {
    Help(&'a str),
    Let(&'a str),
    Assignment {
        name: &'a str,
        expr: &'a str,
        lazy: bool,
    },
    Vars,
    Clear(&'a str),
    Reset,
    ResetFull,
    Cache(&'a str),
    Semantics(&'a str),
    Context(&'a str),
    Steps(&'a str),
    Autoexpand(&'a str),
    Budget(&'a str),
    History,
    Show(&'a str),
    Del(&'a str),
    Set(&'a str),
    Equiv(&'a str),
    Subst(&'a str),
    SolveSystem(&'a str),
    Solve(&'a str),
    Simplify(&'a str),
    Config(&'a str),
    Timeline(&'a str),
    Visualize(&'a str),
    Explain(&'a str),
    Det(&'a str),
    Transpose(&'a str),
    Trace(&'a str),
    Telescope(&'a str),
    Weierstrass(&'a str),
    ExpandLog(&'a str),
    Expand(&'a str),
    Rationalize(&'a str),
    Limit(&'a str),
    Profile(&'a str),
    Health(&'a str),
    Eval(&'a str),
}

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

#[cfg(test)]
mod tests {
    use super::{
        parse_repl_command_input, preprocess_repl_function_syntax, split_repl_statements,
        ReplCommandInput,
    };

    #[test]
    fn preprocess_repl_function_syntax_maps_simplify_form() {
        assert_eq!(
            preprocess_repl_function_syntax("simplify(x^2 + 1)"),
            "simplify x^2 + 1"
        );
    }

    #[test]
    fn preprocess_repl_function_syntax_maps_solve_form() {
        assert_eq!(
            preprocess_repl_function_syntax("solve(x+1=2,x)"),
            "solve x+1=2,x"
        );
    }

    #[test]
    fn preprocess_repl_function_syntax_trims_and_preserves_other_inputs() {
        assert_eq!(
            preprocess_repl_function_syntax("  x + x  "),
            "x + x".to_string()
        );
    }

    #[test]
    fn parses_help_prefix() {
        assert_eq!(
            parse_repl_command_input("help solve"),
            ReplCommandInput::Help("help solve")
        );
    }

    #[test]
    fn parses_lazy_assignment_before_eval() {
        assert_eq!(
            parse_repl_command_input("x := y + 1"),
            ReplCommandInput::Assignment {
                name: "x",
                expr: "y + 1",
                lazy: true,
            }
        );
    }

    #[test]
    fn parses_solve_system_before_solve() {
        assert_eq!(
            parse_repl_command_input("solve_system(x+y=1; x; y)"),
            ReplCommandInput::SolveSystem("solve_system(x+y=1; x; y)")
        );
    }

    #[test]
    fn parses_expand_log_before_expand() {
        assert_eq!(
            parse_repl_command_input("expand_log ln(x*y)"),
            ReplCommandInput::ExpandLog("expand_log ln(x*y)")
        );
    }

    #[test]
    fn parses_bare_commands_instead_of_falling_back_to_eval() {
        assert_eq!(
            parse_repl_command_input("equiv"),
            ReplCommandInput::Equiv("equiv")
        );
        assert_eq!(
            parse_repl_command_input("subst"),
            ReplCommandInput::Subst("subst")
        );
        assert_eq!(
            parse_repl_command_input("solve"),
            ReplCommandInput::Solve("solve")
        );
        assert_eq!(
            parse_repl_command_input("simplify"),
            ReplCommandInput::Simplify("simplify")
        );
        assert_eq!(
            parse_repl_command_input("telescope"),
            ReplCommandInput::Telescope("telescope")
        );
        assert_eq!(
            parse_repl_command_input("weierstrass"),
            ReplCommandInput::Weierstrass("weierstrass")
        );
        assert_eq!(
            parse_repl_command_input("rationalize"),
            ReplCommandInput::Rationalize("rationalize")
        );
        assert_eq!(
            parse_repl_command_input("limit"),
            ReplCommandInput::Limit("limit")
        );
    }

    #[test]
    fn parses_visualize_alias_viz() {
        assert_eq!(
            parse_repl_command_input("viz x + 1"),
            ReplCommandInput::Visualize("viz x + 1")
        );
    }

    #[test]
    fn falls_back_to_eval() {
        assert_eq!(
            parse_repl_command_input("x + x"),
            ReplCommandInput::Eval("x + x")
        );
    }

    #[test]
    fn split_repl_statements_preserves_solve_system_line() {
        let parts = split_repl_statements("solve_system x+y=3; x-y=1; x; y");
        assert_eq!(parts, vec!["solve_system x+y=3; x-y=1; x; y"]);
    }

    #[test]
    fn split_repl_statements_splits_regular_semicolons() {
        let parts = split_repl_statements("let a = 1; let b = 2; a + b");
        assert_eq!(parts, vec!["let a = 1", "let b = 2", "a + b"]);
    }
}
