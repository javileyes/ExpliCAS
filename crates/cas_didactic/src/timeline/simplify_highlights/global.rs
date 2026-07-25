mod additive;
mod additive_render;
mod additive_search;
mod default_transition;
mod direct;
mod scope;

use self::scope::render_local_scope_transition;
use super::TimelineStepSnapshots;
use crate::runtime::Step;
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};

/// Truthfulness guard for a partial colour span.
///
/// A step's red/green is an ASSERTION: "replace this piece with that piece and
/// you get the next state". When the span lands on the wrong subtree the
/// assertion is false, and the audit found it published as an identity under a
/// rule name — `x → x`, `3·(−3) → (−3)²`, `∂/∂x(y·x²) → ∂/∂y(2·x·y)`. The
/// dominant cause is PATH DRIFT: the recorded path is valid on the RAW tree
/// (`ctx.add_raw` preserves operand order) while presentation renders the tree
/// after `normalize_expr_for_display`, which rebuilds through `ctx.add` and
/// canonicalises.
///
/// So the span is CHECKED before it is published: substituting every red span
/// of `before` by the corresponding green span of `after` must reproduce
/// `after`. When it does not, the pair is DECLINED and the caller falls back to
/// colouring the whole state — less precise, but true.
///
/// This is a string-level check on purpose: it sits at the chokepoint both
/// surfaces share, needs no path threading through the four sub-renderers, and
/// costs nothing at the expression layer. Recovering the spans that this
/// declines benignly (canonical reordering) is C2.1's job, which rewrites focus
/// resolution anyway.
fn colour_spans(latex: &str, colour: &str) -> Vec<(usize, usize, String)> {
    let needle = format!("{{\\color{{{colour}}}{{");
    let bytes = latex.as_bytes();
    let mut spans = Vec::new();
    let mut cursor = 0usize;
    while let Some(found) = latex[cursor..].find(&needle) {
        let start = cursor + found;
        let inner_start = start + needle.len();
        let mut depth = 1usize;
        let mut idx = inner_start;
        while idx < bytes.len() && depth > 0 {
            match bytes[idx] {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                _ => {}
            }
            idx += 1;
        }
        if depth != 0 || idx >= bytes.len() {
            return Vec::new();
        }
        let inner_end = idx - 1;
        // after the inner group closes, the wrapper's own `}` follows
        let whole_end = inner_end + 2;
        if whole_end > latex.len() {
            return Vec::new();
        }
        spans.push((start, whole_end, latex[inner_start..inner_end].to_string()));
        cursor = whole_end;
    }
    spans
}

fn strip_colours(latex: &str) -> String {
    let mut current = latex.to_string();
    loop {
        let mut spans = colour_spans(&current, "red");
        spans.extend(colour_spans(&current, "green"));
        if spans.is_empty() {
            return current;
        }
        spans.sort_by_key(|(start, _, _)| *start);
        let mut rebuilt = String::with_capacity(current.len());
        let mut last = 0usize;
        for (start, end, inner) in spans {
            rebuilt.push_str(&current[last..start]);
            rebuilt.push_str(&inner);
            last = end;
        }
        rebuilt.push_str(&current[last..]);
        current = rebuilt;
    }
}

fn normalized(latex: &str) -> String {
    latex
        .replace("\\left", "")
        .replace("\\right", "")
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

/// `true` when the published pair keeps its promise.
fn span_transition_is_truthful(before: &str, after: &str) -> bool {
    let reds = colour_spans(before, "red");
    let greens = colour_spans(after, "green");
    // No partial span: the whole state is coloured (or nothing is). Nothing to
    // check — the pair says "this state became that state", which is the
    // before/after itself.
    if reds.is_empty() || greens.is_empty() {
        return true;
    }
    // MULTI-SPAN EXCEPTION, declared (plan §7.1, lifted in C2.2). When the two
    // sides carry a different number of spans the assertion is "these N pieces
    // became that one" — a truthful claim that substitution cannot express, and
    // declining it would erase correct narration (two pins fix exactly this
    // shape: `semantics_cli_contract_tests` steps 4 and 9). The guard covers
    // the one-to-one case, which is where the audit's false identities live.
    if reds.len() != greens.len() {
        return true;
    }
    let mut rebuilt = String::with_capacity(before.len());
    let mut last = 0usize;
    for ((start, end, _), (_, _, green_inner)) in reds.iter().zip(greens.iter()) {
        rebuilt.push_str(&before[last..*start]);
        rebuilt.push_str(green_inner);
        last = *end;
    }
    rebuilt.push_str(&before[last..]);
    normalized(&strip_colours(&rebuilt)) == normalized(&strip_colours(after))
}

/// Whole-state fallback for a declined span: still true, just less precise.
fn whole_state_transition(before: &str, after: &str) -> (String, String) {
    (
        format!("{{\\color{{red}}{{{}}}}}", strip_colours(before)),
        format!("{{\\color{{green}}{{{}}}}}", strip_colours(after)),
    )
}

pub(super) fn render_global_transition_latex(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    let local_scope = preferred_local_scope(context, step);

    let (before, after) = if let Some(before_local) = local_scope {
        render_local_scope_transition(
            context,
            step,
            snapshots,
            before_local,
            display_hints,
            style_prefs,
        )
    } else {
        default_transition::render_default_global_transition(
            context,
            step,
            snapshots,
            display_hints,
            style_prefs,
        )
    };

    if span_transition_is_truthful(&before, &after) {
        (before, after)
    } else {
        whole_state_transition(&before, &after)
    }
}

fn preferred_local_scope(context: &Context, step: &Step) -> Option<ExprId> {
    let focus_before = step.before_local().unwrap_or(step.before);
    if step.before_local().is_some() {
        return Some(focus_before);
    }

    match context.get(focus_before) {
        Expr::Function(_, _)
        | Expr::Add(_, _)
        | Expr::Sub(_, _)
        | Expr::Div(_, _)
        | Expr::Pow(_, _) => Some(focus_before),
        _ => None,
    }
}
