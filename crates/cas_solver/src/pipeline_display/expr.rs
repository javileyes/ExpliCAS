use cas_ast::{Context, ExprId};
use cas_math::poly_store::try_render_poly_result;

/// Display an expression, preferring formatted poly output when available.
pub fn display_expr_or_poly(ctx: &Context, id: ExprId) -> String {
    if let Some(poly_str) = try_render_poly_result(ctx, id) {
        return poly_str;
    }
    let rendered = cas_formatter::clean_display_string(&format!(
        "{}",
        cas_formatter::DisplayExpr { context: ctx, id }
    ));
    compact_subtracted_difference_display(rendered)
}

pub(crate) fn compact_subtracted_difference_display(input: String) -> String {
    let Some(diff) = find_safe_subtracted_difference(&input) else {
        return compact_adjacent_sign_display(input);
    };

    compact_adjacent_sign_display(format!(
        "{}{}{}",
        &input[..diff.start],
        render_subtracted_additive_terms(&diff.terms),
        &input[diff.close + 1..]
    ))
}

fn compact_adjacent_sign_display(input: String) -> String {
    input.replace(" + -", " - ").replace(" - -", " + ")
}

struct SafeSubtractedDifference<'a> {
    start: usize,
    close: usize,
    terms: Vec<(AdditiveSign, &'a str)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AdditiveSign {
    Positive,
    Negative,
}

fn find_safe_subtracted_difference(input: &str) -> Option<SafeSubtractedDifference<'_>> {
    let start = find_top_level_subtracted_parenthesized(input)?;
    let open = start + " - ".len();
    let close = matching_close_paren(input, open)?;
    if !is_safe_subtracted_difference_suffix(&input[close + 1..]) {
        return None;
    }

    let inner = &input[open + 1..close];
    let terms = split_safe_top_level_additive_terms(inner)?;
    if terms.len() < 2
        || !terms
            .iter()
            .any(|(sign, _)| *sign == AdditiveSign::Negative)
    {
        return None;
    }

    Some(SafeSubtractedDifference {
        start,
        close,
        terms,
    })
}

fn find_top_level_subtracted_parenthesized(input: &str) -> Option<usize> {
    let mut depth = 0usize;

    for (idx, ch) in input.char_indices() {
        if depth == 0 && input[idx..].starts_with(" - (") {
            return Some(idx);
        }

        match ch {
            '(' => depth += 1,
            ')' => depth = depth.checked_sub(1)?,
            _ => {}
        }
    }

    None
}

fn is_safe_subtracted_difference_suffix(suffix: &str) -> bool {
    let suffix = suffix.trim_start();
    suffix.is_empty() || suffix.starts_with("+ ") || suffix.starts_with("- ")
}

fn matching_close_paren(input: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (idx, ch) in input[open..].char_indices() {
        let idx = open + idx;
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(idx);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_safe_top_level_additive_terms(input: &str) -> Option<Vec<(AdditiveSign, &str)>> {
    let mut depth = 0usize;
    let mut terms = Vec::new();
    let mut start = 0usize;
    let mut sign = AdditiveSign::Positive;

    for (idx, ch) in input.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.checked_sub(1)?,
            '+' | '-'
                if depth == 0
                    && idx > start
                    && input[..idx].ends_with(' ')
                    && input[idx + 1..].starts_with(' ') =>
            {
                let term = input[start..idx].trim();
                if term.is_empty() {
                    return None;
                }
                terms.push((sign, term));
                start = idx + 1;
                sign = if ch == '+' {
                    AdditiveSign::Positive
                } else {
                    AdditiveSign::Negative
                };
            }
            _ => {}
        }
    }

    let term = input[start..].trim();
    if term.is_empty() {
        return None;
    }
    terms.push((sign, term));
    Some(terms)
}

fn render_subtracted_additive_terms(terms: &[(AdditiveSign, &str)]) -> String {
    let mut rendered = String::new();
    for (index, (sign, term)) in terms.iter().enumerate() {
        let inverted = match sign {
            AdditiveSign::Positive => AdditiveSign::Negative,
            AdditiveSign::Negative => AdditiveSign::Positive,
        };
        if index == 0 {
            match inverted {
                AdditiveSign::Positive => {
                    rendered.push_str(" + ");
                    rendered.push_str(term);
                }
                AdditiveSign::Negative => {
                    rendered.push_str(" - ");
                    rendered.push_str(term);
                }
            }
            continue;
        }

        match inverted {
            AdditiveSign::Positive => rendered.push_str(" + "),
            AdditiveSign::Negative => rendered.push_str(" - "),
        }
        rendered.push_str(term);
    }
    rendered
}
