//! Formatting helpers for domain conditions (`ConditionPredicate`/`ConditionSet`).

use cas_ast::{ConditionPredicate, ConditionSet, Context};

use crate::{DisplayExpr, LaTeXExpr};

/// Render a single condition predicate as plain text.
pub fn condition_predicate_to_display(pred: &ConditionPredicate, ctx: &Context) -> String {
    let expr_str = DisplayExpr {
        context: ctx,
        id: pred.expr_id(),
    }
    .to_string();

    match pred {
        ConditionPredicate::NonZero(_) => format!("{} != 0", expr_str),
        ConditionPredicate::Positive(_) => format!("{} > 0", expr_str),
        ConditionPredicate::NonNegative(_) => format!("{} >= 0", expr_str),
        ConditionPredicate::Defined(_) => format!("defined({})", expr_str),
        ConditionPredicate::InvTrigPrincipalRange { func, .. } => {
            format!("{} in principal range of {}", expr_str, func)
        }
        ConditionPredicate::EqZero(_) => format!("{} = 0", expr_str),
        ConditionPredicate::EqOne(_) => format!("{} = 1", expr_str),
    }
}

/// Render a single condition predicate as LaTeX.
pub fn condition_predicate_to_latex(pred: &ConditionPredicate, ctx: &Context) -> String {
    let expr_latex = LaTeXExpr {
        context: ctx,
        id: pred.expr_id(),
    }
    .to_latex();

    match pred {
        ConditionPredicate::NonZero(_) => format!("{} \\neq 0", expr_latex),
        ConditionPredicate::Positive(_) => format!("{} > 0", expr_latex),
        ConditionPredicate::NonNegative(_) => format!("{} \\geq 0", expr_latex),
        ConditionPredicate::Defined(_) => format!("\\text{{defined}}({})", expr_latex),
        ConditionPredicate::InvTrigPrincipalRange { func, .. } => {
            format!("{} \\in \\text{{principal range of }}{}", expr_latex, func)
        }
        ConditionPredicate::EqZero(_) => format!("{} = 0", expr_latex),
        ConditionPredicate::EqOne(_) => format!("{} = 1", expr_latex),
    }
}

/// Render a condition set as plain text.
pub fn condition_set_to_display(set: &ConditionSet, ctx: &Context) -> String {
    if set.is_empty() {
        "always true".to_string()
    } else {
        set.predicates()
            .iter()
            .map(|p| condition_predicate_to_display(p, ctx))
            .collect::<Vec<_>>()
            .join(" and ")
    }
}

/// Render a condition set as LaTeX.
pub fn condition_set_to_latex(set: &ConditionSet, ctx: &Context) -> String {
    if set.is_empty() {
        r"\text{always true}".to_string()
    } else {
        set.predicates()
            .iter()
            .map(|p| condition_predicate_to_latex(p, ctx))
            .collect::<Vec<_>>()
            .join(r" \land ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_set_display_empty() {
        let ctx = Context::new();
        let set = ConditionSet::empty();
        assert_eq!(condition_set_to_display(&set, &ctx), "always true");
    }

    #[test]
    fn condition_set_latex_nonzero() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let set = ConditionSet::single(ConditionPredicate::NonZero(x));
        assert_eq!(condition_set_to_latex(&set, &ctx), "x \\neq 0");
    }
}
