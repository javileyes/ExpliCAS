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
}
