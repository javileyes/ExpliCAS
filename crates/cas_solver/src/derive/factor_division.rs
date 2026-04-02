use cas_ast::{Expr, ExprId};

use super::strong_target_match;

pub(crate) fn detect_factor_out_with_division_target(
    ctx: &mut cas_ast::Context,
    target_expr: ExprId,
    candidate_variables: &[String],
) -> Option<String> {
    candidate_variables
        .iter()
        .find(|var_name| extract_factored_inner_target(ctx, target_expr, var_name).is_some())
        .cloned()
}

pub(crate) fn extract_factored_inner_target(
    ctx: &mut cas_ast::Context,
    target_expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let var_expr = ctx.var(var_name);

    if !ctx.is_mul_commutative(target_expr) {
        return None;
    }

    let factors = cas_math::trig_roots_flatten::flatten_mul_chain(ctx, target_expr);
    let match_index = factors
        .iter()
        .position(|factor| strong_target_match(ctx, *factor, var_expr))?;

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

    contains_division_by_var(ctx, inner, var_expr).then_some(inner)
}

fn contains_division_by_var(ctx: &mut cas_ast::Context, expr: ExprId, var_expr: ExprId) -> bool {
    match ctx.get(expr).clone() {
        Expr::Div(_, den) => strong_target_match(ctx, den, var_expr),
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            contains_division_by_var(ctx, left, var_expr)
                || contains_division_by_var(ctx, right, var_expr)
        }
        Expr::Pow(base, exp) => {
            contains_division_by_var(ctx, base, var_expr)
                || contains_division_by_var(ctx, exp, var_expr)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_division_by_var(ctx, inner, var_expr),
        Expr::Function(_, args) => args
            .into_iter()
            .any(|arg| contains_division_by_var(ctx, arg, var_expr)),
        Expr::Matrix { data, .. } => data
            .into_iter()
            .any(|item| contains_division_by_var(ctx, item, var_expr)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{detect_factor_out_with_division_target, extract_factored_inner_target};

    #[test]
    fn detects_factored_with_division_target_shape() {
        let mut ctx = cas_ast::Context::new();
        let target = cas_parser::parse("x*(a + b + c/x)", &mut ctx).expect("target");
        let vars = ["x".to_string(), "y".to_string()];
        let detected =
            detect_factor_out_with_division_target(&mut ctx, target, &vars).expect("detect");
        assert_eq!(detected, "x");
    }

    #[test]
    fn extracts_inner_target_after_factored_variable() {
        let mut ctx = cas_ast::Context::new();
        let target = cas_parser::parse("x*(a + b + c/x)", &mut ctx).expect("target");
        let inner = extract_factored_inner_target(&mut ctx, target, "x").expect("inner");
        let rendered = cas_formatter::render_expr(&ctx, inner);
        assert!(rendered.contains("c / x") || rendered.contains("c/x"));
    }

    #[test]
    fn rejects_plain_factored_target_without_division() {
        let mut ctx = cas_ast::Context::new();
        let target = cas_parser::parse("x*(a + b + c)", &mut ctx).expect("target");
        assert!(extract_factored_inner_target(&mut ctx, target, "x").is_none());
    }
}
