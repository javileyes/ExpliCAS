use crate::runtime::Step;
use cas_ast::{Context, ExprId};

fn strip_redundant_outer_parens(input: &str) -> &str {
    let mut current = input.trim();
    loop {
        if !current.starts_with('(') || !current.ends_with(')') {
            return current;
        }

        let mut depth = 0usize;
        let mut encloses_whole = true;
        for (index, ch) in current.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 && index + ch.len_utf8() < current.len() {
                        encloses_whole = false;
                        break;
                    }
                }
                _ => {}
            }
        }

        if !encloses_whole {
            return current;
        }

        current = &current[1..current.len() - 1];
    }
}

fn has_top_level_additive_operator(input: &str) -> bool {
    let mut depth = 0usize;
    for (index, ch) in input.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            '+' if depth == 0 => return true,
            '-' if depth == 0 && index > 0 => return true,
            _ => {}
        }
    }
    false
}

fn collapse_negated_redundant_parens(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;

    while i < chars.len() {
        if chars[i] == '-' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && chars[j] == '(' {
                let mut depth = 0usize;
                let mut k = j;
                while k < chars.len() {
                    match chars[k] {
                        '(' => depth += 1,
                        ')' => {
                            depth = depth.saturating_sub(1);
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }

                if k < chars.len() && depth == 0 {
                    let inner: String = chars[j + 1..k].iter().collect();
                    let trimmed = strip_redundant_outer_parens(inner.trim());
                    if !trimmed.is_empty() && !has_top_level_additive_operator(trimmed) {
                        out.push('-');
                        out.push_str(trimmed);
                        i = k + 1;
                        continue;
                    }
                }
            }
        }

        out.push(chars[i]);
        i += 1;
    }

    out
}

fn normalize_visible_equivalence(input: &str) -> String {
    collapse_negated_redundant_parens(strip_redundant_outer_parens(input))
}

pub(super) fn render_rule_with_scope_line(
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    local_rule_expr_ids: fn(&Step) -> (ExprId, ExprId),
    render_expr: fn(&mut Context, ExprId, &cas_formatter::root_style::StylePreferences) -> String,
) -> Option<String> {
    let (rule_before_id, rule_after_id) = local_rule_expr_ids(step);
    let before_disp =
        cas_formatter::clean_display_string(&render_expr(ctx, rule_before_id, style_prefs));
    let after_disp = cas_formatter::clean_display_string(&cas_formatter::render_with_rule_scope(
        ctx,
        rule_after_id,
        &step.rule_name,
        style_prefs,
    ));

    if normalize_visible_equivalence(&before_disp) == normalize_visible_equivalence(&after_disp) {
        return None;
    }

    Some(format!(
        "   Cambio local: {} -> {}",
        before_disp, after_disp
    ))
}
