use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::contains_division_like_term;
use num_bigint::BigInt;

/// Pattern classification for nested fractions.
#[derive(Debug)]
pub(crate) enum NestedFractionPattern {
    /// P1: 1/(a + 1/b) -> b/(a*b + 1)
    OneOverSumWithUnitFraction,
    /// P2: 1/(a + b/c) -> c/(a*c + b)
    OneOverSumWithFraction,
    /// P3: A/(B + C/D) -> A*D/(B*D + C)
    FractionOverSumWithFraction,
    /// P4: (A + 1/B)/C -> (A*B + 1)/(B*C)
    SumWithFractionOverScalar,
    /// Fallback for complex patterns.
    General,
}

/// Find and return the first `Div` node within an expression.
pub(crate) fn find_div_in_expr(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Div(_, _) => Some(id),
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            find_div_in_expr(ctx, *l).or_else(|| find_div_in_expr(ctx, *r))
        }
        Expr::Mul(l, r) => find_div_in_expr(ctx, *l).or_else(|| find_div_in_expr(ctx, *r)),
        Expr::Neg(inner) | Expr::Hold(inner) => find_div_in_expr(ctx, *inner),
        Expr::Pow(b, e) => find_div_in_expr(ctx, *b).or_else(|| find_div_in_expr(ctx, *e)),
        Expr::Function(_, args) => args.iter().find_map(|a| find_div_in_expr(ctx, *a)),
        Expr::Matrix { data, .. } => data.iter().find_map(|e| find_div_in_expr(ctx, *e)),
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => None,
    }
}

/// Classify a nested fraction expression.
pub(crate) fn classify_nested_fraction(
    ctx: &Context,
    expr: ExprId,
) -> Option<NestedFractionPattern> {
    let is_one = |id: ExprId| -> bool {
        matches!(ctx.get(id), Expr::Number(n) if n.is_integer() && *n.numer() == BigInt::from(1))
    };

    let find_fraction_in_add = |id: ExprId| -> Option<ExprId> {
        match ctx.get(id) {
            Expr::Add(l, r) => {
                if matches!(ctx.get(*l), Expr::Div(_, _)) {
                    Some(*l)
                } else if matches!(ctx.get(*r), Expr::Div(_, _)) {
                    Some(*r)
                } else {
                    None
                }
            }
            _ => None,
        }
    };

    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Some(inner_frac) = find_fraction_in_add(*den) {
            if is_one(*num) {
                if let Expr::Div(n, _) = ctx.get(inner_frac) {
                    if is_one(*n) {
                        return Some(NestedFractionPattern::OneOverSumWithUnitFraction);
                    }
                }
                return Some(NestedFractionPattern::OneOverSumWithFraction);
            }
            return Some(NestedFractionPattern::FractionOverSumWithFraction);
        }

        if contains_division_like_term(ctx, *num) && !contains_division_like_term(ctx, *den) {
            return Some(NestedFractionPattern::SumWithFractionOverScalar);
        }

        if contains_division_like_term(ctx, *den) {
            return Some(NestedFractionPattern::General);
        }
    }

    None
}

/// Extract the combined fraction string from an `Add` expression containing a fraction.
/// Example: `1 + 1/x -> "\\frac{x + 1}{x}"` in LaTeX.
pub(crate) fn extract_combined_fraction_str(ctx: &Context, add_expr: ExprId) -> String {
    use cas_formatter::DisplayContext;
    use cas_formatter::LaTeXExprWithHints;

    let hints = DisplayContext::default();
    let to_latex = |id: ExprId| -> String {
        LaTeXExprWithHints {
            context: ctx,
            id,
            hints: &hints,
        }
        .to_latex()
    };

    if let Expr::Add(l, r) = ctx.get(add_expr) {
        let (frac_id, other_id) = if matches!(ctx.get(*l), Expr::Div(_, _)) {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Div(_, _)) {
            (*r, *l)
        } else {
            return "\\text{(combinado)}".to_string();
        };

        if let Expr::Div(frac_num, frac_den) = ctx.get(frac_id) {
            let frac_num_latex = to_latex(*frac_num);
            let frac_den_latex = to_latex(*frac_den);
            let other_latex = to_latex(other_id);

            return format!(
                "\\frac{{{} \\cdot {} + {}}}{{{}}}",
                other_latex, frac_den_latex, frac_num_latex, frac_den_latex
            );
        }
    }

    "\\text{(combinado)}".to_string()
}
