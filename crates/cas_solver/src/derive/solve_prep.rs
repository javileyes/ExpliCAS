use super::strong_target_match;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expand_ops::expand;
use cas_math::expr_extract::extract_i64_multiplier_and_base_factors;
use cas_math::expr_nary::build_balanced_mul;
use cas_math::expr_predicates::contains_named_var;
use cas_math::poly_compare::poly_eq;
use cas_math::trig_roots_flatten::{flatten_add_sub_chain, flatten_mul_chain};
use cas_solver_core::quadratic_coeffs::{
    extract_quadratic_coefficients, extract_simplified_nonzero_quadratic_coefficients_with_state,
};
use num_rational::BigRational;

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
        if !target_looks_like_completed_square(ctx, target_expr, var_name) {
            continue;
        }

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

fn target_looks_like_completed_square(
    ctx: &mut Context,
    target_expr: ExprId,
    var_name: &str,
) -> bool {
    let terms = flatten_add_sub_chain(ctx, target_expr);
    if terms.len() < 2 {
        return false;
    }

    terms
        .into_iter()
        .any(|term| contains_completed_square_factor(ctx, term, var_name))
}

fn contains_completed_square_factor(ctx: &mut Context, expr: ExprId, var_name: &str) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            is_square_exponent(ctx, *exp) && is_affine_binomial_in_var(ctx, *base, var_name)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            contains_completed_square_factor(ctx, *inner, var_name)
        }
        Expr::Mul(_, _) if ctx.is_mul_commutative(expr) => flatten_mul_chain(ctx, expr)
            .into_iter()
            .any(|factor| contains_completed_square_factor(ctx, factor, var_name)),
        _ => false,
    }
}

fn is_affine_binomial_in_var(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            contains_named_var(ctx, *left, var_name) ^ contains_named_var(ctx, *right, var_name)
        }
        _ => false,
    }
}

fn is_square_exponent(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(n) if *n == BigRational::from_integer(2.into())
    )
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
    let b_squared = ctx.add(Expr::Pow(b, two));

    if let Some(double_a) = extract_positive_half_scaled_base_expr(ctx, a) {
        let half_a = ctx.add(Expr::Div(double_a, two));
        let shift = ctx.add(Expr::Div(b, double_a));
        let completed_binomial = ctx.add(Expr::Add(var_expr, shift));
        let square = ctx.add(Expr::Pow(completed_binomial, two));
        let scaled_square = ctx.add(Expr::Mul(half_a, square));
        let two_double_a = ctx.add(Expr::Mul(two, double_a));
        let correction = ctx.add(Expr::Div(b_squared, two_double_a));
        let tail = ctx.add(Expr::Sub(c, correction));

        return Some((ctx.add(Expr::Add(scaled_square, tail)), double_a));
    }

    let two_a_raw = ctx.add(Expr::Mul(two, a));
    let two_a = run_default_simplify(ctx, two_a_raw);
    let shift_raw = ctx.add(Expr::Div(b, two_a));
    let shift = run_default_simplify(ctx, shift_raw);
    let completed_binomial_raw = ctx.add(Expr::Add(var_expr, shift));
    let completed_binomial = run_default_simplify(ctx, completed_binomial_raw);
    let square = ctx.add(Expr::Pow(completed_binomial, two));
    let scaled_square_raw = ctx.add(Expr::Mul(a, square));
    let scaled_square = run_default_simplify(ctx, scaled_square_raw);

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

fn is_positive_one_half_expr(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(n) if *n == BigRational::new(1.into(), 2.into())
    )
}

fn extract_positive_half_scaled_base_expr(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if matches!(ctx.get(*den), Expr::Number(n) if *n == BigRational::from_integer(2.into())) => {
            Some(*num)
        }
        Expr::Mul(lhs, rhs) if is_positive_one_half_expr(ctx, *lhs) => Some(*rhs),
        Expr::Mul(lhs, rhs) if is_positive_one_half_expr(ctx, *rhs) => Some(*lhs),
        _ => None,
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

    type SolvePrepRewriteCase = (&'static str, &'static str, &'static [&'static str]);

    fn assert_tabulated_solve_prep_rewrites(cases: &[SolvePrepRewriteCase]) {
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

    #[test]
    fn rewrites_tabulated_completed_square_monic_targets_aware() {
        assert_tabulated_solve_prep_rewrites(&[
            ("x^2 + 6*x + 5", "(x+3)^2 - 4", &["x"][..]),
            ("x^2 + 2*b*x + c", "(x+b)^2 + c - b^2", &["b", "c", "x"][..]),
            ("x^2 + 3*x + 1", "(x+3/2)^2 - 5/4", &["x"][..]),
        ]);
    }

    #[test]
    fn rewrites_tabulated_completed_square_symbolic_positive_targets_aware() {
        assert_tabulated_solve_prep_rewrites(&[(
            "a*x^2 + b*x + c",
            "a*(x + b/(2*a))^2 + c - b^2/(4*a)",
            &["a", "b", "c", "x"][..],
        )]);
    }

    #[test]
    fn rewrites_tabulated_completed_square_negative_linear_targets_aware() {
        assert_tabulated_solve_prep_rewrites(&[(
            "a*x^2 - b*x + c",
            "a*(x - b/(2*a))^2 + c - b^2/(4*a)",
            &["a", "b", "c", "x"][..],
        )]);
    }

    #[test]
    fn builds_symbolic_negative_leading_completed_square_candidate() {
        let mut ctx = Context::new();
        let source =
            cas_parser::parse("-a*x^2 + b*x + c", &mut ctx).expect("parse solve-prep source");
        let target = cas_parser::parse("-a*(x - b/(2*a))^2 + c + b^2/(4*a)", &mut ctx)
            .expect("parse solve-prep target");
        let expected_positive_a = cas_parser::parse("a", &mut ctx).expect("parse expected coeff");

        let (candidate, positive_a) =
            try_build_negative_leading_complete_square_candidate(&mut ctx, source, "x")
                .expect("must build negative-leading solve-prep candidate");

        assert_eq!(positive_a, expected_positive_a);
        assert!(
            simplified_difference_matches_zero(&mut ctx, candidate, target),
            "expected symbolic negative-leading candidate to match target"
        );
    }

    #[test]
    fn rewrites_tabulated_completed_square_fractional_targets_aware() {
        assert_tabulated_solve_prep_rewrites(&[(
            "(a/2)*x^2 + b*x + c",
            "(a/2)*(x + b/a)^2 + c - b^2/(2*a)",
            &["a", "b", "c", "x"][..],
        )]);
    }

    #[test]
    fn rejects_targets_that_are_not_completed_square_forms() {
        let cases = [
            ("x^2 + 2*x + 1", "(x + 1)^2", &["x"][..]),
            ("x^2 + 2*x + 1", "x*(x + 2) + 1", &["x"][..]),
            (
                "a*y^2 + b*y + c",
                "y*(a*y + b + c/y)",
                &["a", "b", "c", "y"][..],
            ),
        ];

        for (source_text, target_text, vars) in cases {
            let mut ctx = Context::new();
            let source = cas_parser::parse(source_text, &mut ctx).expect("parse source");
            let target = cas_parser::parse(target_text, &mut ctx).expect("parse target");
            let shared_vars: Vec<String> = vars.iter().map(|name| (*name).to_string()).collect();

            assert!(
                try_rewrite_solve_prep_target_aware(&mut ctx, source, target, &shared_vars)
                    .is_none(),
                "solve-prep must not claim `{source_text}` -> `{target_text}`"
            );
        }
    }
}
