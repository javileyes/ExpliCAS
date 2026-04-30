use crate::limit_command_parse_types::LimitCommandInput;
use cas_math::limit_types::{Approach, PreSimplifyMode};

fn split_by_comma_ignoring_parens(s: &str) -> Vec<&str> {
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

fn parse_limit_approach(raw: &str) -> Result<Approach, String> {
    let dir = raw.trim();
    if dir.is_empty()
        || dir.eq_ignore_ascii_case("infinity")
        || dir.eq_ignore_ascii_case("+infinity")
        || dir.eq_ignore_ascii_case("inf")
        || dir.eq_ignore_ascii_case("+inf")
    {
        return Ok(Approach::PosInfinity);
    }
    if dir.eq_ignore_ascii_case("-infinity") || dir.eq_ignore_ascii_case("-inf") {
        return Ok(Approach::NegInfinity);
    }

    Err(format!(
        "Unsupported limit direction `{dir}`. Only infinity and -infinity are currently supported."
    ))
}

pub fn parse_limit_command_input(rest: &str) -> Result<LimitCommandInput<'_>, String> {
    let parts = split_by_comma_ignoring_parens(rest);
    let expr = parts.first().copied().unwrap_or("").trim();
    let var = parts.get(1).copied().unwrap_or("x").trim();
    let dir = parts.get(2).copied().unwrap_or("infinity").trim();
    let mode = parts.get(3).copied().unwrap_or("off").trim();

    let approach = parse_limit_approach(dir)?;
    let presimplify = if mode.eq_ignore_ascii_case("safe") {
        PreSimplifyMode::Safe
    } else {
        PreSimplifyMode::Off
    };

    Ok(LimitCommandInput {
        expr,
        var,
        approach,
        presimplify,
    })
}
