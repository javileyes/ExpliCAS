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

    // `solve([eq1, eq2, ...], [x, y, ...])` is the natural list form of a linear system;
    // rewrite it to `solve_system <eq; ...; var; ...>` so it routes to the system solver
    // (the same normalisation the eval/wire path applies). Must be tried BEFORE the
    // generic `solve(` handling, which would otherwise send it to the single-equation
    // solver and fail to parse the `[...]`.
    if let Some(spec) = cas_api_models::parse_solve_system_list_command(line) {
        return format!("solve_system {}", spec);
    }

    if line.starts_with("solve(") && line.ends_with(')') {
        let content = &line["solve(".len()..line.len() - 1];
        return format!("solve {}", content);
    }

    if line.starts_with("derive(") && line.ends_with(')') {
        let content = &line["derive(".len()..line.len() - 1];
        return format!("derive {}", content);
    }

    if line.starts_with("limit(") && line.ends_with(')') {
        let content = &line["limit(".len()..line.len() - 1];
        return format!("limit {}", content);
    }

    if line.starts_with("lim(") && line.ends_with(')') {
        let content = &line["lim(".len()..line.len() - 1];
        return format!("limit {}", content);
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
