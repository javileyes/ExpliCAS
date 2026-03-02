use crate::Approach;

/// Parsed special command forms accepted by eval-json input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalJsonSpecialCommand {
    Solve {
        equation: String,
        var: String,
    },
    Limit {
        expr: String,
        var: String,
        approach: Approach,
    },
}

/// Parse special eval-json command forms:
/// - `solve(equation, var)`
/// - `limit(expr, var, approach)` or `lim(expr, var, approach)`
pub fn parse_eval_json_special_command(input: &str) -> Option<EvalJsonSpecialCommand> {
    if let Some((equation, var)) = parse_solve_command(input) {
        return Some(EvalJsonSpecialCommand::Solve { equation, var });
    }
    if let Some((expr, var, approach)) = parse_limit_command(input) {
        return Some(EvalJsonSpecialCommand::Limit {
            expr,
            var,
            approach,
        });
    }
    None
}

fn parse_solve_command(input: &str) -> Option<(String, String)> {
    let trimmed = input.trim();
    if !trimmed.to_lowercase().starts_with("solve(") || !trimmed.ends_with(')') {
        return None;
    }

    let content = &trimmed[6..trimmed.len() - 1];
    let mut paren_depth = 0;
    let mut last_comma_pos = None;
    for (i, ch) in content.char_indices() {
        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            ',' if paren_depth == 0 => last_comma_pos = Some(i),
            _ => {}
        }
    }

    let comma_pos = last_comma_pos?;
    let equation_part = content[..comma_pos].trim();
    let variable_part = content[comma_pos + 1..].trim();

    if variable_part.is_empty() || !variable_part.chars().next()?.is_alphabetic() {
        return None;
    }
    if !variable_part
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_')
    {
        return None;
    }

    Some((equation_part.to_string(), variable_part.to_string()))
}

fn parse_limit_command(input: &str) -> Option<(String, String, Approach)> {
    let trimmed = input.trim();
    let lower = trimmed.to_lowercase();
    let prefix_len = if lower.starts_with("limit(") {
        6
    } else if lower.starts_with("lim(") {
        4
    } else {
        return None;
    };

    if !trimmed.ends_with(')') {
        return None;
    }

    let content = &trimmed[prefix_len..trimmed.len() - 1];
    let parts = split_by_comma_at_depth_0(content);
    if parts.len() < 2 || parts.len() > 3 {
        return None;
    }

    let expr_str = parts[0].trim();
    let var_str = parts[1].trim();
    if var_str.is_empty() || !var_str.chars().next()?.is_alphabetic() {
        return None;
    }
    if !var_str.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }

    let approach = if parts.len() == 3 {
        match parts[2].trim().to_lowercase().as_str() {
            "inf" | "infinity" | "+inf" | "+infinity" => Approach::PosInfinity,
            "-inf" | "-infinity" => Approach::NegInfinity,
            _ => return None,
        }
    } else {
        Approach::PosInfinity
    };

    Some((expr_str.to_string(), var_str.to_string(), approach))
}

fn split_by_comma_at_depth_0(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0;

    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                result.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    result.push(&s[start..]);
    result
}
