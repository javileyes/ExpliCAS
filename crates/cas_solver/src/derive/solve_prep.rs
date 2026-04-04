use super::strong_target_match;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expand_ops::expand;
use cas_math::expr_extract::extract_i64_multiplier_and_base_factors;
use cas_math::expr_nary::build_balanced_mul;
use cas_math::poly_compare::poly_eq;
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

        if completed_square_matches_target(ctx, candidate, target_expr, var_name) {
            return Some(DeriveSolvePrepRewrite {
                rewritten: target_expr,
                assume_nonzero_expr: leading_coeff,
                kind: DeriveSolvePrepRewriteKind::CompleteSquare,
            });
        }

        let Some((negative_leading_candidate, positive_leading_coeff)) =
            try_build_negative_leading_complete_square_candidate(ctx, expr, var_name)
        else {
            continue;
        };

        if completed_square_matches_target(ctx, negative_leading_candidate, target_expr, var_name) {
            return Some(DeriveSolvePrepRewrite {
                rewritten: target_expr,
                assume_nonzero_expr: positive_leading_coeff,
                kind: DeriveSolvePrepRewriteKind::CompleteSquare,
            });
        }
    }

    None
}

fn completed_square_matches_target(
    ctx: &mut Context,
    candidate: ExprId,
    target_expr: ExprId,
    var_name: &str,
) -> bool {
    strong_target_match(ctx, candidate, target_expr)
        || poly_eq(ctx, candidate, target_expr)
        || quadratic_coefficients_match_for_var(ctx, candidate, target_expr, var_name)
        || simplified_difference_matches_zero(ctx, candidate, target_expr)
}

fn quadratic_coefficients_match_for_var(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    let left_expanded = expand(ctx, left);
    let right_expanded = expand(ctx, right);
    let Some((left_a, left_b, left_c)) =
        extract_quadratic_coefficients(ctx, left_expanded, var_name)
    else {
        return false;
    };
    let Some((right_a, right_b, right_c)) =
        extract_quadratic_coefficients(ctx, right_expanded, var_name)
    else {
        return false;
    };

    coeffs_match(ctx, left_a, right_a)
        && coeffs_match(ctx, left_b, right_b)
        && coeffs_match(ctx, left_c, right_c)
}

fn coeffs_match(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    strong_target_match(ctx, left, right)
        || poly_eq(ctx, left, right)
        || simplified_difference_matches_zero(ctx, left, right)
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

fn try_build_negative_leading_complete_square_candidate(
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
    let positive_a = positive_part_of_negative_expr(ctx, a)?;

    if is_zero_expr(ctx, b) {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let var_expr = ctx.var(var);
    let neg_b_raw = ctx.add(Expr::Neg(b));
    let neg_b = run_default_simplify(ctx, neg_b_raw);

    let two_a_raw = ctx.add(Expr::Mul(two, positive_a));
    let two_a = run_default_simplify(ctx, two_a_raw);
    let shift_raw = ctx.add(Expr::Div(neg_b, two_a));
    let shift = run_default_simplify(ctx, shift_raw);
    let completed_binomial_raw = ctx.add(Expr::Add(var_expr, shift));
    let completed_binomial = run_default_simplify(ctx, completed_binomial_raw);
    let square = ctx.add(Expr::Pow(completed_binomial, two));
    let scaled_square_raw = ctx.add(Expr::Mul(positive_a, square));
    let scaled_square = run_default_simplify(ctx, scaled_square_raw);
    let neg_scaled_square_raw = ctx.add(Expr::Neg(scaled_square));
    let neg_scaled_square = run_default_simplify(ctx, neg_scaled_square_raw);

    let b_squared = ctx.add(Expr::Pow(b, two));
    let four_a_raw = ctx.add(Expr::Mul(four, positive_a));
    let four_a = run_default_simplify(ctx, four_a_raw);
    let correction_raw = ctx.add(Expr::Div(b_squared, four_a));
    let correction = run_default_simplify(ctx, correction_raw);
    let tail_raw = ctx.add(Expr::Add(c, correction));
    let tail = run_default_simplify(ctx, tail_raw);
    let candidate_raw = ctx.add(Expr::Add(neg_scaled_square, tail));
    let candidate = run_default_simplify(ctx, candidate_raw);

    Some((candidate, positive_a))
}

fn positive_part_of_negative_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(*inner),
        _ => {
            let (multiplier, factors) = extract_i64_multiplier_and_base_factors(ctx, expr);
            if multiplier >= 0 {
                return None;
            }

            let mut rebuilt: Vec<ExprId> = Vec::with_capacity(factors.len() + 1);
            if multiplier != -1 {
                rebuilt.push(ctx.num(-multiplier));
            }
            rebuilt.extend(factors);
            Some(build_balanced_mul(ctx, &rebuilt))
        }
    }
}

fn run_default_simplify(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(
        expr,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
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
    fn rewrites_tabulated_completed_square_targets_aware() {
        let cases = [
            ("x^2 + 6*x + 5", "(x+3)^2 - 4", &["x"][..]),
            (
                "a*x^2 + b*x + c",
                "a*(x + b/(2*a))^2 + c - b^2/(4*a)",
                &["a", "b", "c", "x"][..],
            ),
            ("x^2 + 2*b*x + c", "(x+b)^2 + c - b^2", &["b", "c", "x"][..]),
            (
                "a*y^2 + b*y + c",
                "a*(y + b/(2*a))^2 + c - b^2/(4*a)",
                &["a", "b", "c", "y"][..],
            ),
            (
                "a*x^2 - b*x + c",
                "a*(x - b/(2*a))^2 + c - b^2/(4*a)",
                &["a", "b", "c", "x"][..],
            ),
            (
                "-a*x^2 + b*x + c",
                "-a*(x - b/(2*a))^2 + c + b^2/(4*a)",
                &["a", "b", "c", "x"][..],
            ),
            ("x^2 + 3*x + 1", "(x+3/2)^2 - 5/4", &["x"][..]),
            (
                "(a/2)*x^2 + b*x + c",
                "(a/2)*(x + b/a)^2 + c - b^2/(2*a)",
                &["a", "b", "c", "x"][..],
            ),
            (
                "(a/2)*y^2 - b*y + c",
                "(a/2)*(y - b/a)^2 + c - b^2/(2*a)",
                &["a", "b", "c", "y"][..],
            ),
        ];

        for (source_text, target_text, vars) in cases {
            let mut ctx = Context::new();
            let source = cas_parser::parse(source_text, &mut ctx).expect("parse source");
            let target = cas_parser::parse(target_text, &mut ctx).expect("parse target");
            let shared_vars: Vec<String> = vars.iter().map(|name| (*name).to_string()).collect();

            let rewrite = try_rewrite_solve_prep_target_aware(
                &mut ctx,
                source,
                target,
                &shared_vars,
            )
            .unwrap_or_else(|| {
                panic!("must rewrite solve-prep source `{source_text}` to target `{target_text}`")
            });

            assert_eq!(rewrite.rewritten, target);
            assert_eq!(rewrite.kind, DeriveSolvePrepRewriteKind::CompleteSquare);
        }
    }
}
