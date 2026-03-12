use cas_math::limit_types::{Approach, PreSimplifyMode};
use cas_solver_core::limit_command_types::LimitCommandInput;

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

pub fn parse_limit_command_input(rest: &str) -> LimitCommandInput<'_> {
    let parts = split_by_comma_ignoring_parens(rest);
    let expr = parts.first().copied().unwrap_or("").trim();
    let var = parts.get(1).copied().unwrap_or("x").trim();
    let dir = parts.get(2).copied().unwrap_or("infinity").trim();
    let mode = parts.get(3).copied().unwrap_or("off").trim();

    let approach = if dir.contains("-infinity") || dir.contains("-inf") {
        Approach::NegInfinity
    } else {
        Approach::PosInfinity
    };
    let presimplify = if mode.eq_ignore_ascii_case("safe") {
        PreSimplifyMode::Safe
    } else {
        PreSimplifyMode::Off
    };

    LimitCommandInput {
        expr,
        var,
        approach,
        presimplify,
    }
}
