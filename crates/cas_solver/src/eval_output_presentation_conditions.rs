use cas_api_models::{AssumptionDto, BlockedHintDto, RequiredConditionWire, WarningWire};
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_solver_core::domain_normalization::normalize_and_dedupe_conditions;
use std::collections::HashSet;

use crate::eval_output_condition_filter::AssumedConditionFilter;

pub(crate) fn collect_output_warnings(
    domain_warnings: &[crate::DomainWarning],
    assumptions_used: &[AssumptionDto],
) -> Vec<WarningWire> {
    let assumed_display: HashSet<&str> = assumptions_used
        .iter()
        .map(|assumption| assumption.display.as_str())
        .collect();
    domain_warnings
        .iter()
        .filter(|w| !assumed_display.contains(w.message.as_str()))
        .map(|w| WarningWire {
            rule: w.rule_name.clone(),
            assumption: w.message.clone(),
        })
        .collect()
}

pub(crate) fn collect_output_required_conditions(
    required_conditions: &[crate::ImplicitCondition],
    ctx: &mut Context,
    assumptions_used: &[AssumptionDto],
    raw_input: &str,
    result_display: Option<&str>,
) -> Vec<RequiredConditionWire> {
    let assumed_filter = AssumedConditionFilter::from_assumptions(assumptions_used);
    normalize_and_dedupe_conditions(ctx, required_conditions)
        .iter()
        .filter(|cond| !assumed_filter.covers_required_condition(ctx, cond))
        .map(|cond| {
            let (kind, expr_id) = match cond {
                crate::ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                crate::ImplicitCondition::LowerBound(e, _) => ("LowerBound", *e),
                crate::ImplicitCondition::Positive(e) => ("Positive", *e),
                crate::ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            let expr_str = expr_display(ctx, expr_id);
            let expr_display =
                apply_input_inverse_trig_alias_preferences(&expr_str, raw_input, result_display);
            RequiredConditionWire {
                kind: kind.to_string(),
                expr_display,
                expr_canonical: expr_str,
            }
        })
        .collect()
}

pub(crate) fn collect_output_required_display(
    required_conditions: &[crate::ImplicitCondition],
    ctx: &mut Context,
    assumptions_used: &[AssumptionDto],
    raw_input: &str,
    result_display: Option<&str>,
) -> Vec<String> {
    let assumed_filter = AssumedConditionFilter::from_assumptions(assumptions_used);
    normalize_and_dedupe_conditions(ctx, required_conditions)
        .iter()
        .filter(|cond| !assumed_filter.covers_required_condition(ctx, cond))
        .map(|cond| {
            apply_input_inverse_trig_alias_preferences(
                &cond.display(ctx),
                raw_input,
                result_display,
            )
        })
        .collect()
}

pub(crate) fn collect_output_blocked_hints(
    ctx: &Context,
    resolved: cas_ast::ExprId,
    required_conditions: &[crate::ImplicitCondition],
    blocked_hints: &[crate::BlockedHint],
) -> Vec<BlockedHintDto> {
    crate::filter_blocked_hints_for_eval(ctx, resolved, required_conditions, blocked_hints)
        .iter()
        .map(|hint| BlockedHintDto {
            rule: hint.rule.clone(),
            requires: vec![crate::format_blocked_hint_condition(ctx, hint)],
            tip: hint.suggestion.to_string(),
        })
        .collect()
}

pub(crate) fn collect_output_assumptions_used(steps: &[crate::Step]) -> Vec<AssumptionDto> {
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut assumptions = Vec::new();

    for step in steps {
        for event in step.assumption_events() {
            if !matches!(
                event.kind,
                cas_solver_core::assumption_model::AssumptionKind::HeuristicAssumption
            ) {
                continue;
            }

            let kind = event.key.kind().to_string();
            let rule = step.rule_name.to_string();
            let display = event.message.clone();
            let expr_canonical = event.expr_display.clone();
            if !seen.insert((kind.clone(), expr_canonical.clone(), display.clone())) {
                continue;
            }

            assumptions.push(AssumptionDto {
                kind,
                display,
                expr_canonical,
                rule,
            });
        }
    }

    assumptions
}

fn expr_display(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    DisplayExpr {
        context: ctx,
        id: expr_id,
    }
    .to_string()
}

pub(crate) fn apply_input_inverse_trig_alias_preferences(
    display: &str,
    raw_input: &str,
    result_display: Option<&str>,
) -> String {
    let mut adjusted = display.to_string();
    let result_lookup = result_display.map(normalize_alias_lookup_text);
    for (short, long) in [
        ("asin", "arcsin"),
        ("acos", "arccos"),
        ("atan", "arctan"),
        ("asec", "arcsec"),
        ("acsc", "arccsc"),
        ("acot", "arccot"),
    ] {
        adjusted = apply_single_input_inverse_trig_alias_preference(
            &adjusted,
            raw_input,
            result_lookup.as_deref(),
            short,
            long,
        );
    }
    adjusted
}

fn apply_single_input_inverse_trig_alias_preference(
    display: &str,
    raw_input: &str,
    result_lookup: Option<&str>,
    short: &str,
    long: &str,
) -> String {
    let long_call_prefix = format!("{long}(");
    let raw_lookup = normalize_alias_lookup_text(raw_input);
    let mut out = String::with_capacity(display.len());
    let mut cursor = 0;

    while let Some(relative_start) = display[cursor..].find(&long_call_prefix) {
        let start = cursor + relative_start;
        let Some(end) = matching_call_end(display, start + long.len()) else {
            break;
        };

        out.push_str(&display[cursor..start]);
        let long_call = &display[start..end];
        let short_call = format!("{short}{}", &long_call[long.len()..]);
        if result_lookup_contains_call(result_lookup, &short_call) {
            out.push_str(&short_call);
        } else if result_lookup_contains_call(result_lookup, long_call) {
            out.push_str(long_call);
        } else if raw_input_contains_short_alias_call(&raw_lookup, short, &short_call) {
            out.push_str(&short_call);
        } else {
            out.push_str(long_call);
        }
        cursor = end;
    }

    out.push_str(&display[cursor..]);
    out
}

fn result_lookup_contains_call(result_lookup: Option<&str>, call: &str) -> bool {
    result_lookup.is_some_and(|lookup| lookup.contains(&normalize_alias_lookup_text(call)))
}

fn raw_input_contains_short_alias_call(raw_lookup: &str, short: &str, short_call: &str) -> bool {
    let short_call_lookup = normalize_alias_lookup_text(short_call);
    if raw_lookup.contains(&short_call_lookup) {
        return true;
    }

    let Some(short_call_arg) = call_argument(&short_call_lookup, short.len()) else {
        return false;
    };
    let short_call_arg = strip_redundant_outer_parens(short_call_arg);
    let short_call_prefix = format!("{short}(");
    let mut cursor = 0;

    while let Some(relative_start) = raw_lookup[cursor..].find(&short_call_prefix) {
        let start = cursor + relative_start;
        let Some(end) = matching_call_end(raw_lookup, start + short.len()) else {
            break;
        };
        let raw_call = &raw_lookup[start..end];
        if call_argument(raw_call, short.len()).map(strip_redundant_outer_parens)
            == Some(short_call_arg)
        {
            return true;
        }
        cursor = end;
    }

    false
}

fn call_argument(call: &str, name_len: usize) -> Option<&str> {
    let open_paren = name_len;
    let end = matching_call_end(call, open_paren)?;
    if end != call.len() {
        return None;
    }
    Some(&call[open_paren + 1..end - 1])
}

fn strip_redundant_outer_parens(mut text: &str) -> &str {
    loop {
        if !text.starts_with('(') || !text.ends_with(')') {
            return text;
        }
        if matching_call_end(text, 0) != Some(text.len()) {
            return text;
        }
        text = &text[1..text.len() - 1];
    }
}

fn matching_call_end(text: &str, open_paren: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(open_paren) != Some(&b'(') {
        return None;
    }
    let mut depth = 0usize;
    for (offset, byte) in bytes[open_paren..].iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(open_paren + offset + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn normalize_alias_lookup_text(text: &str) -> String {
    text.chars()
        .filter_map(|ch| {
            if ch.is_whitespace() {
                None
            } else if ch == '·' {
                Some('*')
            } else {
                Some(ch)
            }
        })
        .collect()
}
