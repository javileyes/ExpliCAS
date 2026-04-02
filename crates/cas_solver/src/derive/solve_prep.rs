use super::strong_target_match;
use cas_ast::{Context, Expr, ExprId};
use cas_solver_core::quadratic_coeffs::{
    extract_quadratic_coefficients, extract_simplified_nonzero_quadratic_coefficients_with_state,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeriveSolvePrepRewriteKind {
    CompleteSquare,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeriveSolvePrepRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) assume_nonzero_expr: ExprId,
    pub(crate) kind: DeriveSolvePrepRewriteKind,
}

impl DeriveSolvePrepRewriteKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::CompleteSquare => "Complete the square to rewrite the quadratic",
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::CompleteSquare => "Complete the Square",
        }
    }
}

pub(crate) fn try_rewrite_solve_prep_target_aware(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
    shared_vars: &[String],
) -> Option<DeriveSolvePrepRewrite> {
    for var_name in shared_vars {
        let Some((candidate, leading_coeff)) =
            try_build_complete_square_candidate(ctx, expr, var_name)
        else {
            continue;
        };

        if strong_target_match(ctx, candidate, target_expr)
            || simplified_difference_matches_zero(ctx, candidate, target_expr)
        {
            return Some(DeriveSolvePrepRewrite {
                rewritten: target_expr,
                assume_nonzero_expr: leading_coeff,
                kind: DeriveSolvePrepRewriteKind::CompleteSquare,
            });
        }
    }

    None
}

fn try_build_complete_square_candidate(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let (a, b, c) = extract_simplified_nonzero_quadratic_coefficients_with_state(
        ctx,
        expr,
        var,
        extract_quadratic_coefficients,
        run_default_simplify,
        is_zero_expr,
    )?;

    if is_zero_expr(ctx, b) {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let var_expr = ctx.var(var);

    let two_a_raw = ctx.add(Expr::Mul(two, a));
    let two_a = run_default_simplify(ctx, two_a_raw);
    let shift_raw = ctx.add(Expr::Div(b, two_a));
    let shift = run_default_simplify(ctx, shift_raw);
    let completed_binomial_raw = ctx.add(Expr::Add(var_expr, shift));
    let completed_binomial = run_default_simplify(ctx, completed_binomial_raw);
    let square = ctx.add(Expr::Pow(completed_binomial, two));
    let scaled_square_raw = ctx.add(Expr::Mul(a, square));
    let scaled_square = run_default_simplify(ctx, scaled_square_raw);

    let b_squared = ctx.add(Expr::Pow(b, two));
    let four_a_raw = ctx.add(Expr::Mul(four, a));
    let four_a = run_default_simplify(ctx, four_a_raw);
    let correction_raw = ctx.add(Expr::Div(b_squared, four_a));
    let correction = run_default_simplify(ctx, correction_raw);
    let tail_raw = ctx.add(Expr::Sub(c, correction));
    let tail = run_default_simplify(ctx, tail_raw);
    let candidate_raw = ctx.add(Expr::Add(scaled_square, tail));
    let candidate = run_default_simplify(ctx, candidate_raw);

    Some((candidate, a))
}

fn run_default_simplify(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) =
        simplifier.simplify_with_stats(expr, crate::SimplifyOptions::default());
    std::mem::swap(&mut simplifier.context, ctx);
    rewritten
}

fn simplified_difference_matches_zero(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    let zero = ctx.num(0);
    let difference = ctx.add(Expr::Sub(left, right));
    let simplified = run_default_simplify(ctx, difference);
    strong_target_match(ctx, simplified, zero)
}

fn is_zero_expr(ctx: &mut Context, expr: ExprId) -> bool {
    let zero = ctx.num(0);
    let simplified = run_default_simplify(ctx, expr);
    strong_target_match(ctx, simplified, zero)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_monic_quadratic_to_completed_square_target() {
        let mut ctx = Context::new();
        let source = cas_parser::parse("x^2 + 6*x + 5", &mut ctx).expect("parse source");
        let target = cas_parser::parse("(x+3)^2 - 4", &mut ctx).expect("parse target");

        let rewrite =
            try_rewrite_solve_prep_target_aware(&mut ctx, source, target, &["x".to_string()])
                .expect("must rewrite to completed square");

        assert_eq!(rewrite.rewritten, target);
        assert_eq!(rewrite.kind, DeriveSolvePrepRewriteKind::CompleteSquare);
    }

    #[test]
    fn rewrites_symbolic_leading_coeff_quadratic_to_completed_square_target() {
        let mut ctx = Context::new();
        let source = cas_parser::parse("a*x^2 + b*x + c", &mut ctx).expect("parse source");
        let target =
            cas_parser::parse("a*(x + b/(2*a))^2 + c - b^2/(4*a)", &mut ctx).expect("parse target");

        let rewrite = try_rewrite_solve_prep_target_aware(
            &mut ctx,
            source,
            target,
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "x".to_string(),
            ],
        )
        .expect("must rewrite to completed square");

        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_negative_leading_coeff_quadratic_to_completed_square_target() {
        let mut ctx = Context::new();
        let source = cas_parser::parse("-x^2 + 4*x + 1", &mut ctx).expect("parse source");
        let target = cas_parser::parse("-(x-2)^2 + 5", &mut ctx).expect("parse target");

        let rewrite =
            try_rewrite_solve_prep_target_aware(&mut ctx, source, target, &["x".to_string()])
                .expect("must rewrite negative-leading quadratic to completed square");

        assert_eq!(rewrite.rewritten, target);
        assert_eq!(rewrite.kind, DeriveSolvePrepRewriteKind::CompleteSquare);
    }

    #[test]
    fn rewrites_alternate_variable_quadratic_to_completed_square_target() {
        let mut ctx = Context::new();
        let source = cas_parser::parse("3*y^2 - 12*y + 7", &mut ctx).expect("parse source");
        let target = cas_parser::parse("3*(y-2)^2 - 5", &mut ctx).expect("parse target");

        let rewrite =
            try_rewrite_solve_prep_target_aware(&mut ctx, source, target, &["y".to_string()])
                .expect("must rewrite alternate-variable quadratic to completed square");

        assert_eq!(rewrite.rewritten, target);
        assert_eq!(rewrite.kind, DeriveSolvePrepRewriteKind::CompleteSquare);
    }

    #[test]
    fn rewrites_symbolic_leading_coeff_quadratic_in_y_to_completed_square_target() {
        let mut ctx = Context::new();
        let source = cas_parser::parse("a*y^2 + b*y + c", &mut ctx).expect("parse source");
        let target =
            cas_parser::parse("a*(y + b/(2*a))^2 + c - b^2/(4*a)", &mut ctx).expect("parse target");

        let rewrite = try_rewrite_solve_prep_target_aware(
            &mut ctx,
            source,
            target,
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "y".to_string(),
            ],
        )
        .expect("must rewrite symbolic y-quadratic to completed square");

        assert_eq!(rewrite.rewritten, target);
        assert_eq!(rewrite.kind, DeriveSolvePrepRewriteKind::CompleteSquare);
    }
}
