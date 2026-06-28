//! Rendering of a periodic solution family `base + k·period` (`k ∈ ℤ`) — the
//! `SolutionSet::Periodic` variant emitted by the trig equation solver.

use crate::{DisplayExpr, LaTeXExpr};
use cas_ast::{Context, Expr, ExprId};

fn base_is_zero(ctx: &Context, base: ExprId) -> bool {
    use num_traits::Zero;
    matches!(ctx.get(base), Expr::Number(n) if n.is_zero())
}

/// Text form of a periodic family: `{ k·pi : k ∈ ℤ }` (zero base) or
/// `{ pi/2 + k·pi : k ∈ ℤ }`.
pub fn display_periodic_family(ctx: &Context, base: ExprId, period: ExprId) -> String {
    let term = format!(
        "k·{}",
        DisplayExpr {
            context: ctx,
            id: period
        }
    );
    if base_is_zero(ctx, base) {
        format!("{{ {term} : k ∈ ℤ }}")
    } else {
        let base_s = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: base
            }
        );
        format!("{{ {base_s} + {term} : k ∈ ℤ }}")
    }
}

/// LaTeX form of a periodic family: `\left\{ k\pi : k \in \mathbb{Z} \right\}` etc.
pub fn latex_periodic_family(ctx: &Context, base: ExprId, period: ExprId) -> String {
    let term = format!(
        "k{}",
        LaTeXExpr {
            context: ctx,
            id: period
        }
        .to_latex()
    );
    if base_is_zero(ctx, base) {
        format!(r"\left\{{ {term} : k \in \mathbb{{Z}} \right\}}")
    } else {
        let base_s = LaTeXExpr {
            context: ctx,
            id: base,
        }
        .to_latex();
        format!(r"\left\{{ {base_s} + {term} : k \in \mathbb{{Z}} \right\}}")
    }
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
        assert_eq!(display_periodic_family(&ctx, zero, pi), "{ k·pi : k ∈ ℤ }");
        assert_eq!(
            latex_periodic_family(&ctx, zero, pi),
            r"\left\{ k\pi : k \in \mathbb{Z} \right\}"
        );
        // Nonzero base keeps it: { 1/2·pi + k·pi : k ∈ ℤ }.
        let two = ctx.num(2);
        let half_pi = ctx.add(Expr::Div(pi, two));
        let text = display_periodic_family(&ctx, half_pi, pi);
        assert!(
            text.starts_with("{ ") && text.contains("+ k·pi : k ∈ ℤ }"),
            "{text}"
        );
    }
}
