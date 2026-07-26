mod additive;
mod additive_render;
mod additive_search;
mod default_transition;
mod direct;
mod scope;

use self::scope::render_local_scope_transition;
use super::renderers::render_with_single_path;
use super::TimelineStepSnapshots;
use crate::runtime::Step;
use cas_ast::{Context, Expr, ExprId, ExprPath};
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
///
/// The invariant is UNIFORM in the number of spans, which is what let the
/// multi-span exception of C1.3 be lifted: collapse each contiguous RUN of
/// coloured spans into a single hole and require the rest — the part the step
/// claims it did not touch — to be **identical** on both sides.
///
///   before: `[H] + [H] - (2+3x)/(1+2x)`   (two adjacent red spans)
///   after:  `[H] - (2+3x)/(1+2x)`         (one green span)
///                → same context, same number of holes: TRUE, "these two
///                  pieces became that one".
///
///   before: `[H] + (6x⁵-120x³)/720`
///   after:  `[H] + (x⁵-20x³)/120`
///                → the context DIFFERS, so the step changed something it did
///                  not mark: FALSE. This is the audit's witness.
///
/// It also subsumes the one-to-one substitution check it replaces: if the
/// untouched part matches and the spans are where the change is, the assertion
/// holds by construction.
fn context_with_holes(latex: &str, colour: &str) -> Option<String> {
    let spans = colour_spans(latex, colour);
    if spans.is_empty() {
        return None;
    }
    // Merge spans separated only by additive/multiplicative glue: they are one
    // manoeuvre, not several.
    let glue_only = |gap: &str| {
        gap.chars()
            .all(|c| c.is_whitespace() || matches!(c, '+' | '-'))
            || gap.trim() == "\\cdot"
            || gap.trim().is_empty()
    };
    let mut rebuilt = String::with_capacity(latex.len());
    let mut last_end = 0usize;
    let mut previous_end: Option<usize> = None;
    for (start, end, _) in &spans {
        let merges = previous_end.is_some_and(|pe| glue_only(&latex[pe..*start]));
        if merges {
            // absorb the glue into the same hole
        } else {
            rebuilt.push_str(&latex[last_end..*start]);
            rebuilt.push('\u{1}');
        }
        last_end = *end;
        previous_end = Some(*end);
    }
    rebuilt.push_str(&latex[last_end..]);
    Some(rebuilt)
}

fn span_transition_is_truthful(before: &str, after: &str) -> bool {
    let (Some(before_context), Some(after_context)) = (
        context_with_holes(before, "red"),
        context_with_holes(after, "green"),
    ) else {
        // No partial span on one of the sides: the pair simply says "this state
        // became that state", which is the before/after itself.
        return true;
    };
    normalized(&strip_colours(&before_context)) == normalized(&strip_colours(&after_context))
}

/// Structural diff between the two states: the MINIMAL set of paths where they
/// differ.
///
/// This is the answer to PATH DRIFT (RC-1). The recorded `step.path()` is valid
/// on the RAW tree and the presentation renders the CANONICALISED one, so the
/// index no longer designates the node it was recorded against — measured at
/// 30 % of steps. Rather than transporting the path through the normalisation
/// (refuted: `Context::add` flattens to n-ary and changes the path's LENGTH),
/// the focus is recomputed from the two states themselves.
///
/// By construction the result is TRUE: everything outside the returned paths is
/// structurally identical on both sides, which is exactly what the guard checks.
/// It also absorbs RC-5 (first-occurrence ambiguity under hash-consing): a diff
/// walks positions, so two identical subtrees at different positions are told
/// apart by where they sit, not by their id.
fn minimal_diff_paths(
    context: &Context,
    before: ExprId,
    after: ExprId,
    prefix: &mut ExprPath,
    out: &mut Vec<ExprPath>,
    budget: &mut usize,
) {
    if before == after {
        return;
    }
    if *budget == 0 {
        out.push(prefix.clone());
        return;
    }
    *budget -= 1;
    let before_children = expr_children(context, before);
    let after_children = expr_children(context, after);
    // Different shape, or different arity: the difference IS this node.
    if before_children.is_none()
        || after_children.is_none()
        || !same_variant(context, before, after)
    {
        out.push(prefix.clone());
        return;
    }
    let (bc, ac) = (before_children.unwrap(), after_children.unwrap());
    if bc.len() != ac.len() {
        out.push(prefix.clone());
        return;
    }
    let differing: Vec<usize> = (0..bc.len()).filter(|&i| bc[i] != ac[i]).collect();
    // More than one child differs: colouring each separately would claim they
    // changed independently. Claim the node instead — true, and less precise.
    if differing.len() != 1 {
        out.push(prefix.clone());
        return;
    }
    let i = differing[0];
    prefix.push(i as u8);
    minimal_diff_paths(context, bc[i], ac[i], prefix, out, budget);
    prefix.pop();
}

fn same_variant(context: &Context, a: ExprId, b: ExprId) -> bool {
    use cas_ast::Expr::*;
    matches!(
        (context.get(a), context.get(b)),
        (Add(_, _), Add(_, _))
            | (Sub(_, _), Sub(_, _))
            | (Mul(_, _), Mul(_, _))
            | (Div(_, _), Div(_, _))
            | (Pow(_, _), Pow(_, _))
            | (Neg(_), Neg(_))
            | (Hold(_), Hold(_))
            | (Function(_, _), Function(_, _))
    )
}

fn expr_children(context: &Context, id: ExprId) -> Option<Vec<ExprId>> {
    use cas_ast::Expr::*;
    Some(match context.get(id) {
        Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Pow(a, b) => vec![*a, *b],
        Neg(a) | Hold(a) => vec![*a],
        Function(_, args) => args.clone(),
        _ => return None,
    })
}

/// Recompute the highlight from the two states, and publish it only if the
/// guard agrees. Returns `None` when the diff is the whole expression (nothing
/// gained over the honest whole-state fallback).
fn content_focus_transition(
    context: &Context,
    snapshots: TimelineStepSnapshots,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> Option<(String, String)> {
    let mut paths = Vec::new();
    let mut prefix = ExprPath::new();
    let mut budget = 64usize;
    minimal_diff_paths(
        context,
        snapshots.global_before_expr,
        snapshots.global_after_expr,
        &mut prefix,
        &mut paths,
        &mut budget,
    );
    let [path] = paths.as_slice() else {
        return None;
    };
    if path.is_empty() {
        return None;
    }
    let before = render_with_single_path(
        context,
        snapshots.global_before_expr,
        path.clone(),
        cas_formatter::HighlightColor::Red,
        display_hints,
        style_prefs,
    );
    let after = render_with_single_path(
        context,
        snapshots.global_after_expr,
        path.clone(),
        cas_formatter::HighlightColor::Green,
        display_hints,
        style_prefs,
    );
    span_transition_is_truthful(&before, &after).then_some((before, after))
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
        return (before, after);
    }
    // The recorded focus lied. Recompute it from the two states before falling
    // back to colouring everything: the structural diff yields a span that is
    // true by construction, so most declines recover their precision.
    if let Some(recovered) =
        content_focus_transition(context, snapshots, display_hints, style_prefs)
    {
        return recovered;
    }
    whole_state_transition(&before, &after)
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
