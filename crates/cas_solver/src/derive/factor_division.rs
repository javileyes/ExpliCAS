use cas_ast::{views::as_rational_const, Expr, ExprId};
use num_traits::{One, Signed};

use super::strong_target_match;

pub(crate) fn detect_factor_out_with_division_target(
    ctx: &mut cas_ast::Context,
    target_expr: ExprId,
    candidate_variables: &[String],
) -> Option<String> {
    candidate_variables
        .iter()
        .find(|var_name| extract_factored_division_target(ctx, target_expr, var_name).is_some())
        .cloned()
}

pub(crate) fn extract_factored_division_target(
    ctx: &mut cas_ast::Context,
    target_expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    let var_expr = ctx.var(var_name);

    if !ctx.is_mul_commutative(target_expr) {
        return None;
    }

    let factors = cas_math::trig_roots_flatten::flatten_mul_chain(ctx, target_expr);
    let match_index = factors
        .iter()
        .position(|factor| matches_factored_variable_power(ctx, *factor, var_expr))?;

    let factored_expr = factors[match_index];
    let mut remaining = factors;
    remaining.remove(match_index);
    if remaining.is_empty() {
        return None;
    }

    let mut iter = remaining.into_iter();
    let mut inner = iter.next()?;
    for factor in iter {
        inner = ctx.add(Expr::Mul(inner, factor));
    }

    contains_division_by_factor(ctx, inner, factored_expr).then_some((factored_expr, inner))
}

fn matches_factored_variable_power(
    ctx: &mut cas_ast::Context,
    factor_expr: ExprId,
    var_expr: ExprId,
) -> bool {
    if strong_target_match(ctx, factor_expr, var_expr) {
        return true;
    }

    match ctx.get(factor_expr).clone() {
        Expr::Pow(base, exp) => {
            strong_target_match(ctx, base, var_expr) && is_positive_integer_power(ctx, exp)
        }
        _ => false,
    }
}

fn is_positive_integer_power(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    as_rational_const(ctx, expr, 8)
        .is_some_and(|value| value.is_positive() && value.denom().is_one())
}

fn contains_division_by_factor(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    factor_expr: ExprId,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Div(_, den) => strong_target_match(ctx, den, factor_expr),
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            contains_division_by_factor(ctx, left, factor_expr)
                || contains_division_by_factor(ctx, right, factor_expr)
        }
        Expr::Pow(base, exp) => {
            contains_division_by_factor(ctx, base, factor_expr)
                || contains_division_by_factor(ctx, exp, factor_expr)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            contains_division_by_factor(ctx, inner, factor_expr)
        }
        Expr::Function(_, args) => args
            .into_iter()
            .any(|arg| contains_division_by_factor(ctx, arg, factor_expr)),
        Expr::Matrix { data, .. } => data
            .into_iter()
            .any(|item| contains_division_by_factor(ctx, item, factor_expr)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{detect_factor_out_with_division_target, extract_factored_division_target};

    #[test]
    fn detects_tabulated_factored_with_division_targets() {
        let cases = [
            ("x*(a + b + c/x)", "x", &["x", "c / x"][..]),
            ("y*(a*y + b + c/y)", "y", &["y", "c / y"][..]),
            (
                "x^2*(a*x^2 + b*x + c + d/x^2)",
                "x",
                &["x^2", "d / x^2"][..],
            ),
            (
                "x^3*(a*x^4 + b*x^2 + c + d/x^3)",
                "x",
                &["x^3", "d / x^3"][..],
            ),
        ];

        for (target, expected_var, rendered_fragments) in cases {
            let mut ctx = cas_ast::Context::new();
            let target = cas_parser::parse(target, &mut ctx).expect("target");
            let vars = ["x".to_string(), "y".to_string()];
            let detected =
                detect_factor_out_with_division_target(&mut ctx, target, &vars).expect("detect");
            assert_eq!(detected, expected_var);

            let (factor, inner) =
                extract_factored_division_target(&mut ctx, target, expected_var).expect("extract");
            let rendered = format!(
                "{} || {}",
                cas_formatter::render_expr(&ctx, factor),
                cas_formatter::render_expr(&ctx, inner)
            );
            for fragment in rendered_fragments {
                assert!(
                    rendered.contains(fragment)
                        || rendered.contains(&fragment.replace("^2", "^(2)").replace("^3", "^(3)")),
                    "expected rendered target `{rendered}` to contain `{fragment}`"
                );
            }
        }
    }

    #[test]
    fn rejects_plain_factored_target_without_division() {
        let mut ctx = cas_ast::Context::new();
        let target = cas_parser::parse("x*(a + b + c)", &mut ctx).expect("target");
        assert!(extract_factored_division_target(&mut ctx, target, "x").is_none());
    }
}
