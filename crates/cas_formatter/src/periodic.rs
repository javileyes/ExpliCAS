//! Rendering of a periodic solution set `baseᵢ + k·period` (`k ∈ ℤ`) — the
//! `SolutionSet::Periodic` variant emitted by the trig equation solver. One base renders as a
//! single family (`{ k·pi : k ∈ ℤ }`); several bases share the period
//! (`{ pi/6 + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }`).

use crate::{DisplayExpr, LaTeXExpr};
use cas_ast::{Context, Expr, ExprId};

fn base_is_zero(ctx: &Context, base: ExprId) -> bool {
    use num_traits::Zero;
    matches!(ctx.get(base), Expr::Number(n) if n.is_zero())
}

/// One `baseᵢ + k·period` family term in text form (`k·pi` when the base is zero).
fn display_family_term(ctx: &Context, base: ExprId, period_term: &str) -> String {
    if base_is_zero(ctx, base) {
        period_term.to_string()
    } else {
        let base_s = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: base
            }
        );
        format!("{base_s} + {period_term}")
    }
}

/// Text form of a periodic set: `{ k·pi : k ∈ ℤ }`, `{ pi/2 + k·pi : k ∈ ℤ }`, or for several
/// families `{ pi/6 + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }`.
pub fn display_periodic_family(ctx: &Context, bases: &[ExprId], period: ExprId) -> String {
    let period_term = format!(
        "k·{}",
        DisplayExpr {
            context: ctx,
            id: period
        }
    );
    let families: Vec<String> = bases
        .iter()
        .map(|&b| display_family_term(ctx, b, &period_term))
        .collect();
    format!("{{ {} : k ∈ ℤ }}", families.join(", "))
}

/// One family term in LaTeX form.
fn latex_family_term(ctx: &Context, base: ExprId, period_term: &str) -> String {
    if base_is_zero(ctx, base) {
        period_term.to_string()
    } else {
        let base_s = LaTeXExpr {
            context: ctx,
            id: base,
        }
        .to_latex();
        format!("{base_s} + {period_term}")
    }
}

/// LaTeX form of a periodic set: `\left\{ k\pi : k \in \mathbb{Z} \right\}` etc.
pub fn latex_periodic_family(ctx: &Context, bases: &[ExprId], period: ExprId) -> String {
    let period_term = format!(
        "k{}",
        LaTeXExpr {
            context: ctx,
            id: period
        }
        .to_latex()
    );
    let families: Vec<String> = bases
        .iter()
        .map(|&b| latex_family_term(ctx, b, &period_term))
        .collect();
    format!(
        r"\left\{{ {} : k \in \mathbb{{Z}} \right\}}",
        families.join(", ")
    )
}

/// One `(a + k·T, b + k·T)` window in text form, with per-endpoint
/// open/closed brackets. A zero endpoint renders as the bare period term.
fn display_window(ctx: &Context, window: &cas_ast::Interval, period_term: &str) -> String {
    let open = match window.min_type {
        cas_ast::BoundType::Open => "(",
        cas_ast::BoundType::Closed => "[",
    };
    let close = match window.max_type {
        cas_ast::BoundType::Open => ")",
        cas_ast::BoundType::Closed => "]",
    };
    let lo = display_family_term(ctx, window.min, period_term);
    let hi = display_family_term(ctx, window.max, period_term);
    format!("{open}{lo}, {hi}{close}")
}

/// Text form of a periodic interval union, mirroring the `Periodic` frame:
/// `{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi) : k ∈ ℤ }`; several windows share
/// the period and join with commas, and mixed closedness renders
/// per-endpoint (`[k·pi, 1/2·pi + k·pi)`).
pub fn display_periodic_interval_union(
    ctx: &Context,
    windows: &[cas_ast::Interval],
    period: ExprId,
) -> String {
    let period_term = format!(
        "k·{}",
        DisplayExpr {
            context: ctx,
            id: period
        }
    );
    let parts: Vec<String> = windows
        .iter()
        .map(|w| display_window(ctx, w, &period_term))
        .collect();
    format!("{{ {} : k ∈ ℤ }}", parts.join(", "))
}

/// One window in LaTeX form (per-endpoint brackets via `\left(`/`\right]`…).
fn latex_window(ctx: &Context, window: &cas_ast::Interval, period_term: &str) -> String {
    let open = match window.min_type {
        cas_ast::BoundType::Open => r"\left(",
        cas_ast::BoundType::Closed => r"\left[",
    };
    let close = match window.max_type {
        cas_ast::BoundType::Open => r"\right)",
        cas_ast::BoundType::Closed => r"\right]",
    };
    let lo = latex_family_term(ctx, window.min, period_term);
    let hi = latex_family_term(ctx, window.max, period_term);
    format!("{open} {lo}, {hi} {close}")
}

/// LaTeX form of a periodic interval union:
/// `\left\{ \left( \frac{\pi}{6} + k2\pi, … \right) : k \in \mathbb{Z} \right\}`.
pub fn latex_periodic_interval_union(
    ctx: &Context,
    windows: &[cas_ast::Interval],
    period: ExprId,
) -> String {
    let period_term = format!(
        "k{}",
        LaTeXExpr {
            context: ctx,
            id: period
        }
        .to_latex()
    );
    let parts: Vec<String> = windows
        .iter()
        .map(|w| latex_window(ctx, w, &period_term))
        .collect();
    format!(
        r"\left\{{ {} : k \in \mathbb{{Z}} \right\}}",
        parts.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Constant, Expr};

    #[test]
    fn periodic_family_text_and_latex() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        // Zero base omits the leading term: { k·pi : k ∈ ℤ }.
        assert_eq!(
            display_periodic_family(&ctx, &[zero], pi),
            "{ k·pi : k ∈ ℤ }"
        );
        assert_eq!(
            latex_periodic_family(&ctx, &[zero], pi),
            r"\left\{ k\pi : k \in \mathbb{Z} \right\}"
        );
        // Nonzero base keeps it.
        let two = ctx.num(2);
        let half_pi = ctx.add(Expr::Div(pi, two));
        let text = display_periodic_family(&ctx, &[half_pi], pi);
        assert!(
            text.starts_with("{ ") && text.contains("+ k·pi : k ∈ ℤ }"),
            "{text}"
        );
        // Two families share the period, joined by a comma.
        let two_fam = display_periodic_family(&ctx, &[zero, half_pi], pi);
        assert!(
            two_fam.contains("k·pi, ") && two_fam.ends_with(" : k ∈ ℤ }"),
            "{two_fam}"
        );
    }

    #[test]
    fn periodic_interval_union_text_mixed_closedness() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let two = ctx.num(2);
        let half = ctx.add(Expr::Div(pi, two));
        // [kπ, π/2 + kπ): closed at the zero base, open at the asymptote.
        let w = cas_ast::Interval {
            min: zero,
            min_type: cas_ast::BoundType::Closed,
            max: half,
            max_type: cas_ast::BoundType::Open,
        };
        assert_eq!(
            display_periodic_interval_union(&ctx, &[w], pi),
            "{ [k·pi, pi / 2 + k·pi) : k ∈ ℤ }"
        );
    }

    #[test]
    fn periodic_interval_union_latex_multiple_windows() {
        let mut ctx = Context::new();
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let six = ctx.num(6);
        let sixth = ctx.add(Expr::Div(pi, six));
        let two = ctx.num(2);
        let half = ctx.add(Expr::Div(pi, two));
        let w1 = cas_ast::Interval {
            min: sixth,
            min_type: cas_ast::BoundType::Open,
            max: half,
            max_type: cas_ast::BoundType::Open,
        };
        let w2 = cas_ast::Interval {
            min: half,
            min_type: cas_ast::BoundType::Closed,
            max: pi,
            max_type: cas_ast::BoundType::Closed,
        };
        let out = latex_periodic_interval_union(&ctx, &[w1, w2], pi);
        assert!(
            out.starts_with(r"\left\{") && out.ends_with(r"\right\}"),
            "{out}"
        );
        assert!(out.contains(r"\left(") && out.contains(r"\right)"), "{out}");
        assert!(out.contains(r"\left[") && out.contains(r"\right]"), "{out}");
        assert!(out.contains(r"k \in \mathbb{Z}"), "{out}");
    }
}
