use cas_ast::{Context, ExprId};

/// Parse statement input, allowing `#N` session-ref shorthand as expression.
pub fn parse_statement_or_session_ref(
    ctx: &mut Context,
    input: &str,
) -> Result<cas_parser::Statement, String> {
    if input.starts_with('#') && input[1..].chars().all(char::is_numeric) {
        Ok(cas_parser::Statement::Expression(ctx.var(input)))
    } else {
        cas_parser::parse_statement(input, ctx).map_err(|e| e.to_string())
    }
}

/// Split a REPL line into executable statements.
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

/// Parse an input as expression; equations are converted to `Equal(lhs, rhs)`.
pub fn parse_expr_or_equation_as_expr(ctx: &mut Context, input: &str) -> Result<ExprId, String> {
    let stmt = parse_statement_or_session_ref(ctx, input)?;
    Ok(match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(expr) => expr,
    })
}

/// Split string by delimiter, ignoring delimiters inside parentheses.
pub fn rsplit_ignoring_parens(s: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut balance = 0;
    let mut split_idx = None;

    for (i, c) in s.char_indices().rev() {
        if c == ')' {
            balance += 1;
        } else if c == '(' {
            balance -= 1;
        } else if c == delimiter && balance == 0 {
            split_idx = Some(i);
            break;
        }
    }

    split_idx.map(|idx| (&s[..idx], &s[idx + 1..]))
}

/// Split string by commas, respecting parenthesis/bracket nesting.
pub fn split_by_comma_ignoring_parens(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut balance = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => balance += 1,
            ')' | ']' | '}' => balance -= 1,
            ',' if balance == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    if start < s.len() {
        parts.push(&s[start..]);
    }

    parts
}

/// Parsed shape of a `timeline ...` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineCommandInput {
    Solve(String),
    Simplify { expr: String, aggressive: bool },
}

/// Parsed shape of a `cache ...` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheCommandInput {
    Status,
    Clear,
    Unknown(String),
}

/// Parse the tail of a `timeline` command.
pub fn parse_timeline_command_input(rest: &str) -> TimelineCommandInput {
    if let Some(solve_rest) = rest.strip_prefix("solve ") {
        return TimelineCommandInput::Solve(solve_rest.trim().to_string());
    }

    if let Some(inner) = rest
        .strip_prefix("simplify(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return TimelineCommandInput::Simplify {
            expr: inner.trim().to_string(),
            aggressive: true,
        };
    }

    if let Some(simplify_rest) = rest.strip_prefix("simplify ") {
        return TimelineCommandInput::Simplify {
            expr: simplify_rest.trim().to_string(),
            aggressive: true,
        };
    }

    TimelineCommandInput::Simplify {
        expr: rest.trim().to_string(),
        aggressive: false,
    }
}

/// Parse a `cache` command line.
pub fn parse_cache_command_input(line: &str) -> CacheCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1).copied() {
        None | Some("status") => CacheCommandInput::Status,
        Some("clear") => CacheCommandInput::Clear,
        Some(other) => CacheCommandInput::Unknown(other.to_string()),
    }
}

/// Parsed shape of a `limit` command tail:
/// `<expr> [, <var> [, <direction> [, safe]]]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitCommandInput<'a> {
    pub expr: &'a str,
    pub var: &'a str,
    pub approach: crate::Approach,
    pub presimplify: crate::PreSimplifyMode,
}

/// Parse the tail of a `limit` command.
/// Defaults:
/// - variable: `x`
/// - direction: `+infinity`
/// - mode: `off`
pub fn parse_limit_command_input(rest: &str) -> LimitCommandInput<'_> {
    let parts = split_by_comma_ignoring_parens(rest);
    let expr = parts.first().copied().unwrap_or("").trim();
    let var = parts.get(1).copied().unwrap_or("x").trim();
    let dir = parts.get(2).copied().unwrap_or("infinity").trim();
    let mode = parts.get(3).copied().unwrap_or("off").trim();

    let approach = if dir.contains("-infinity") || dir.contains("-inf") {
        crate::Approach::NegInfinity
    } else {
        crate::Approach::PosInfinity
    };
    let presimplify = if mode.eq_ignore_ascii_case("safe") {
        crate::PreSimplifyMode::Safe
    } else {
        crate::PreSimplifyMode::Off
    };

    LimitCommandInput {
        expr,
        var,
        approach,
        presimplify,
    }
}

/// Errors for parsing two expression arguments (e.g. `equiv a, b`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseExprPairError {
    MissingDelimiter,
    FirstArg(String),
    SecondArg(String),
}

/// Parse two top-level comma-separated expressions/equations.
pub fn parse_expr_pair(
    ctx: &mut Context,
    input: &str,
) -> Result<(ExprId, ExprId), ParseExprPairError> {
    let (left, right) =
        rsplit_ignoring_parens(input, ',').ok_or(ParseExprPairError::MissingDelimiter)?;
    let left = left.trim();
    let right = right.trim();

    let first = parse_expr_or_equation_as_expr(ctx, left).map_err(ParseExprPairError::FirstArg)?;
    let second =
        parse_expr_or_equation_as_expr(ctx, right).map_err(ParseExprPairError::SecondArg)?;
    Ok((first, second))
}

/// Errors for parsing substitution triple arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseSubstituteArgsError {
    InvalidArity,
    Expression(String),
    Target(String),
    Replacement(String),
}

/// Parse `subst` arguments: `<expr>, <target>, <replacement>`.
pub fn parse_substitute_args(
    ctx: &mut Context,
    input: &str,
) -> Result<(ExprId, ExprId, ExprId), ParseSubstituteArgsError> {
    let parts = split_by_comma_ignoring_parens(input);
    if parts.len() != 3 {
        return Err(ParseSubstituteArgsError::InvalidArity);
    }

    let expr_str = parts[0].trim();
    let target_str = parts[1].trim();
    let replacement_str = parts[2].trim();

    let expr = cas_parser::parse(expr_str, ctx)
        .map_err(|e| ParseSubstituteArgsError::Expression(e.to_string()))?;
    let target = cas_parser::parse(target_str, ctx)
        .map_err(|e| ParseSubstituteArgsError::Target(e.to_string()))?;
    let replacement = cas_parser::parse(replacement_str, ctx)
        .map_err(|e| ParseSubstituteArgsError::Replacement(e.to_string()))?;

    Ok((expr, target, replacement))
}

#[cfg(test)]
mod tests {
    use super::{
        parse_cache_command_input, parse_expr_or_equation_as_expr, parse_expr_pair,
        parse_limit_command_input, parse_statement_or_session_ref, parse_substitute_args,
        parse_timeline_command_input, rsplit_ignoring_parens, split_by_comma_ignoring_parens,
        split_repl_statements, CacheCommandInput, LimitCommandInput, ParseExprPairError,
        ParseSubstituteArgsError, TimelineCommandInput,
    };

    #[test]
    fn parse_statement_or_session_ref_accepts_session_id() {
        let mut ctx = cas_ast::Context::new();
        let stmt = parse_statement_or_session_ref(&mut ctx, "#3").expect("parse");
        match stmt {
            cas_parser::Statement::Expression(_) => {}
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn parse_expr_or_equation_as_expr_wraps_equation() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse_expr_or_equation_as_expr(&mut ctx, "x+1=2").expect("parse");
        match ctx.get(expr) {
            cas_ast::Expr::Function(_, args) => assert_eq!(args.len(), 2),
            _ => panic!("expected function-wrapped equation"),
        }
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

    #[test]
    fn rsplit_ignoring_parens_skips_inner_delimiters() {
        let split = rsplit_ignoring_parens("f(a,b),x", ',').expect("split");
        assert_eq!(split.0, "f(a,b)");
        assert_eq!(split.1, "x");
    }

    #[test]
    fn split_by_comma_ignoring_parens_handles_nested() {
        let parts = split_by_comma_ignoring_parens("a,b(c,d),e");
        assert_eq!(parts, vec!["a", "b(c,d)", "e"]);
    }

    #[test]
    fn parse_expr_pair_handles_equations_and_exprs() {
        let mut ctx = cas_ast::Context::new();
        let (left, right) = parse_expr_pair(&mut ctx, "x+1=2, y+3").expect("pair parse");
        match ctx.get(left) {
            cas_ast::Expr::Function(_, args) => assert_eq!(args.len(), 2),
            _ => panic!("expected function-wrapped equation"),
        }
        assert!(cas_ast::eq::unwrap_eq(&ctx, right).is_none());
    }

    #[test]
    fn parse_expr_pair_requires_delimiter() {
        let mut ctx = cas_ast::Context::new();
        let err = parse_expr_pair(&mut ctx, "x+1").expect_err("missing delimiter");
        assert_eq!(err, ParseExprPairError::MissingDelimiter);
    }

    #[test]
    fn parse_substitute_args_parses_three_parts() {
        let mut ctx = cas_ast::Context::new();
        let parsed = parse_substitute_args(&mut ctx, "x^2 + x, x, 3").expect("subst parse");
        assert_ne!(parsed.0, parsed.1);
        assert_ne!(parsed.1, parsed.2);
    }

    #[test]
    fn parse_substitute_args_rejects_bad_arity() {
        let mut ctx = cas_ast::Context::new();
        let err = parse_substitute_args(&mut ctx, "x^2 + x, x").expect_err("bad arity");
        assert_eq!(err, ParseSubstituteArgsError::InvalidArity);
    }

    #[test]
    fn parse_timeline_input_detects_solve() {
        let parsed = parse_timeline_command_input("solve x+2=5, x");
        assert_eq!(parsed, TimelineCommandInput::Solve("x+2=5, x".to_string()));
    }

    #[test]
    fn parse_timeline_input_detects_aggressive_simplify() {
        let parsed = parse_timeline_command_input("simplify(x+1)");
        assert_eq!(
            parsed,
            TimelineCommandInput::Simplify {
                expr: "x+1".to_string(),
                aggressive: true
            }
        );
    }

    #[test]
    fn parse_timeline_input_defaults_to_non_aggressive() {
        let parsed = parse_timeline_command_input("x+1");
        assert_eq!(
            parsed,
            TimelineCommandInput::Simplify {
                expr: "x+1".to_string(),
                aggressive: false
            }
        );
    }

    #[test]
    fn parse_limit_command_input_defaults() {
        let parsed = parse_limit_command_input("x^2");
        assert_eq!(
            parsed,
            LimitCommandInput {
                expr: "x^2",
                var: "x",
                approach: crate::Approach::PosInfinity,
                presimplify: crate::PreSimplifyMode::Off
            }
        );
    }

    #[test]
    fn parse_limit_command_input_handles_nested_commas() {
        let parsed = parse_limit_command_input("(x+1)/(f(a,b)), x, -infinity, safe");
        assert_eq!(parsed.expr, "(x+1)/(f(a,b))");
        assert_eq!(parsed.var, "x");
        assert_eq!(parsed.approach, crate::Approach::NegInfinity);
        assert_eq!(parsed.presimplify, crate::PreSimplifyMode::Safe);
    }

    #[test]
    fn parse_cache_command_input_defaults_to_status() {
        assert_eq!(
            parse_cache_command_input("cache"),
            CacheCommandInput::Status
        );
        assert_eq!(
            parse_cache_command_input("cache status"),
            CacheCommandInput::Status
        );
    }

    #[test]
    fn parse_cache_command_input_supports_clear() {
        assert_eq!(
            parse_cache_command_input("cache clear"),
            CacheCommandInput::Clear
        );
    }

    #[test]
    fn parse_cache_command_input_captures_unknown_subcommand() {
        assert_eq!(
            parse_cache_command_input("cache nope"),
            CacheCommandInput::Unknown("nope".to_string())
        );
    }
}
