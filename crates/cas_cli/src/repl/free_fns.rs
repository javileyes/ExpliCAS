use super::*;

// Helper to split string by delimiter, ignoring delimiters inside parentheses
pub(super) fn rsplit_ignoring_parens(s: &str, delimiter: char) -> Option<(&str, &str)> {
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

    if let Some(idx) = split_idx {
        Some((&s[..idx], &s[idx + 1..]))
    } else {
        None
    }
}

/// Split string by commas, respecting parentheses nesting.
/// Returns a Vec of the split parts.
pub(super) fn split_by_comma_ignoring_parens(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut balance = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' => balance += 1,
            ')' | ']' => balance -= 1,
            ',' if balance == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    // Add the last part
    if start < s.len() {
        parts.push(&s[start..]);
    }
    parts
}

pub(super) fn display_solution_set(ctx: &cas_ast::Context, set: &cas_ast::SolutionSet) -> String {
    match set {
        cas_ast::SolutionSet::Empty => "Empty Set".to_string(),
        cas_ast::SolutionSet::AllReals => "All Real Numbers".to_string(),
        cas_ast::SolutionSet::Discrete(exprs) => {
            let s: Vec<String> = exprs
                .iter()
                .map(|e| {
                    format!(
                        "{}",
                        DisplayExpr {
                            context: ctx,
                            id: *e
                        }
                    )
                })
                .collect();
            format!("{{ {} }}", s.join(", "))
        }
        cas_ast::SolutionSet::Continuous(interval) => display_interval(ctx, interval),
        cas_ast::SolutionSet::Union(intervals) => {
            let s: Vec<String> = intervals.iter().map(|i| display_interval(ctx, i)).collect();
            s.join(" U ")
        }
        cas_ast::SolutionSet::Residual(expr) => {
            // Display residual expression (unsolved)
            format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: *expr
                }
            )
        }
        cas_ast::SolutionSet::Conditional(cases) => {
            // V2.0 Phase 2C: Pretty-print conditional solutions
            // V2.1: Use "otherwise:" without "if" prefix for natural reading
            // V2.x: Skip "otherwise" cases that only contain Residual (not useful info)
            let case_strs: Vec<String> = cases
                .iter()
                .filter_map(|case| {
                    // Skip "otherwise" cases that only contain Residual
                    if case.when.is_otherwise()
                        && matches!(&case.then.solutions, cas_ast::SolutionSet::Residual(_))
                    {
                        return None;
                    }
                    let sol_str = display_solution_set(ctx, &case.then.solutions);
                    if case.when.is_otherwise() {
                        Some(format!("  otherwise: {}", sol_str))
                    } else {
                        let cond_str = case.when.display_with_context(ctx);
                        Some(format!("  if {}: {}", cond_str, sol_str))
                    }
                })
                .collect();
            // If only one case remains after filtering, display it more simply
            if case_strs.len() == 1 {
                // Extract and format the single case more compactly
                case_strs[0].trim().to_string()
            } else {
                format!("Conditional:\n{}", case_strs.join("\n"))
            }
        }
    }
}

pub(super) fn display_interval(ctx: &cas_ast::Context, interval: &cas_ast::Interval) -> String {
    let min_bracket = match interval.min_type {
        cas_ast::BoundType::Open => "(",
        cas_ast::BoundType::Closed => "[",
    };
    let max_bracket = match interval.max_type {
        cas_ast::BoundType::Open => ")",
        cas_ast::BoundType::Closed => "]",
    };

    // Simple display without trying to simplify
    // The intervals should already have simplified bounds
    format!(
        "{}{}, {}{}",
        min_bracket,
        DisplayExpr {
            context: ctx,
            id: interval.min
        },
        DisplayExpr {
            context: ctx,
            id: interval.max
        },
        max_bracket
    )
}
