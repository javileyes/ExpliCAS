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
