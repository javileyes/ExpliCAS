use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::factoring_support::{
    try_rewrite_automatic_factor_expr, try_rewrite_difference_of_squares_product_expr,
    try_rewrite_factor_common_integer_from_add_expr,
    try_rewrite_factor_common_integer_from_add_expr_with_policy,
    try_rewrite_factor_difference_squares_nary_expr, try_rewrite_factor_function_expr,
    try_rewrite_sum_three_cubes_zero_expr, AutomaticFactorRewriteKind,
    DifferenceOfSquaresProductRewriteKind, FactorCommonIntegerFromAddPolicy,
    FactorDifferenceSquaresNaryRewriteKind, FactorFunctionRewriteKind,
    SumThreeCubesZeroRewriteKind,
};
use cas_math::polynomial::Polynomial;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn format_factor_common_integer_from_add_desc(gcd_int: &BigInt) -> String {
    format!("Factor out {}", gcd_int)
}

fn format_difference_of_squares_product_desc(
    kind: DifferenceOfSquaresProductRewriteKind,
) -> &'static str {
    match kind {
        DifferenceOfSquaresProductRewriteKind::Basic => "(a-b)(a+b) = a² - b²",
        DifferenceOfSquaresProductRewriteKind::NaryConjugateProduct => {
            "(U+V)(U-V) = U² - V² (conjugate product)"
        }
        DifferenceOfSquaresProductRewriteKind::NaryScan => "(a-b)(a+b)·… = (a²-b²)·… (n-ary scan)",
    }
}

fn format_factor_function_desc(kind: FactorFunctionRewriteKind) -> &'static str {
    match kind {
        FactorFunctionRewriteKind::Factorization => "Factorization",
    }
}

fn format_factor_difference_squares_nary_desc(
    kind: FactorDifferenceSquaresNaryRewriteKind,
) -> &'static str {
    match kind {
        FactorDifferenceSquaresNaryRewriteKind::Empty => "Factor difference of squares (Empty)",
        FactorDifferenceSquaresNaryRewriteKind::Nary => "Factor difference of squares (N-ary)",
    }
}

fn format_automatic_factor_desc(kind: AutomaticFactorRewriteKind) -> &'static str {
    match kind {
        AutomaticFactorRewriteKind::ReducedSize => "Automatic Factorization (Reduced Size)",
        AutomaticFactorRewriteKind::DiffSquares => "Automatic Factorization (Diff Squares)",
        AutomaticFactorRewriteKind::AlternatingCubicVandermonde => {
            "Automatic Factorization (Alternating Cubic Vandermonde)"
        }
    }
}

fn format_sum_three_cubes_zero_desc(kind: SumThreeCubesZeroRewriteKind) -> &'static str {
    match kind {
        SumThreeCubesZeroRewriteKind::ZeroSumIdentity => "x³ + y³ + z³ = 3xyz (when x + y + z = 0)",
    }
}

// DifferenceOfSquaresRule: Expands conjugate products
// (a - b) * (a + b) → a² - b²
// Now supports N-ary sums: (U + V)(U - V) → U² - V²
// Also scans n-ary product chains: (a+b) * (a-b) * f(x) → (a²-b²) * f(x)
// Phase: CORE | POST (structural simplification, not expansion)
define_rule!(
    DifferenceOfSquaresRule,
    "Difference of Squares (Product to Difference)",
    None,
    PhaseMask::CORE | PhaseMask::POST,
    |ctx, expr| {
        let rewrite = try_rewrite_difference_of_squares_product_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_difference_of_squares_product_desc(rewrite.kind)),
        )
    }
);

define_rule!(
    FactorRule,
    "Factor Polynomial",
    Some(crate::target_kind::TargetKindSet::FUNCTION), // Target Function expressions specifically
    PhaseMask::CORE | PhaseMask::POST,
    |ctx, expr| {
        let rewrite = try_rewrite_factor_function_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_factor_function_desc(rewrite.kind)))
    }
);

define_rule!(
    FactorDifferenceSquaresRule,
    "Factor Difference of Squares",
    |ctx, expr| {
        let rewrite = try_rewrite_factor_difference_squares_nary_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_factor_difference_squares_nary_desc(rewrite.kind)),
        )
    }
);

define_rule!(
    AutomaticFactorRule,
    "Automatic Factorization",
    |ctx, expr| {
        let rewrite = try_rewrite_automatic_factor_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_automatic_factor_desc(rewrite.kind)))
    }
);

// FactorCommonIntegerFromAdd: Factor out common integer GCD from sum terms
// Example: 2*√2 - 2 → 2*(√2 - 1)
// Phase: POST (runs after rationalization to clean up results)
define_rule!(
    FactorCommonIntegerFromAdd,
    "Factor Common Integer",
    None,
    PhaseMask::POST,
    |ctx, expr, parent_ctx| {
        if let Some(rewrite) = try_rewrite_factor_common_integer_from_add_expr(ctx, expr) {
            let rewritten = rewrite.rewritten;
            let desc = format_factor_common_integer_from_add_desc(&rewrite.gcd_int);
            return Some(Rewrite::new(rewritten).desc(desc).local(expr, rewritten));
        }

        if should_factor_variable_common_integer_in_compact_power_product(ctx, expr, parent_ctx) {
            if let Some(rewrite) = try_rewrite_factor_common_integer_from_add_expr_with_policy(
                ctx,
                expr,
                FactorCommonIntegerFromAddPolicy {
                    allow_variable_terms: true,
                },
            ) {
                let rewritten = rewrite.rewritten;
                let desc = format_factor_common_integer_from_add_desc(&rewrite.gcd_int);
                return Some(Rewrite::new(rewritten).desc(desc).local(expr, rewritten));
            }
        }

        if should_factor_variable_common_integer_in_constant_reciprocal_denominator(
            ctx, expr, parent_ctx,
        ) {
            if let Some((rewritten, gcd_int)) =
                try_factor_low_degree_polynomial_integer_content(ctx, expr)
            {
                let desc = format_factor_common_integer_from_add_desc(&gcd_int);
                return Some(Rewrite::new(rewritten).desc(desc).local(expr, rewritten));
            }
        }

        None
    }
);

fn try_factor_low_degree_polynomial_integer_content(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, BigInt)> {
    let variables = cas_ast::collect_variables(ctx, expr);
    let var = variables.iter().next()?;
    if variables.len() != 1 {
        return None;
    }

    let poly = Polynomial::from_expr(ctx, expr, var.as_str()).ok()?;
    if !(1..=2).contains(&poly.degree()) {
        return None;
    }
    if is_integer_scaled_primitive_low_degree_polynomial(ctx, expr, var.as_str()) {
        return None;
    }

    let gcd_int = polynomial_integer_content_gcd(&poly)?;
    let divisor = BigRational::from_integer(gcd_int.clone());
    let inner_poly = poly.div_scalar(&divisor);
    let inner = inner_poly.to_expr(ctx);
    let gcd_expr = ctx.add(cas_ast::Expr::Number(divisor));
    let rewritten = ctx.add(cas_ast::Expr::Mul(gcd_expr, inner));
    Some((rewritten, gcd_int))
}

fn is_integer_scaled_primitive_low_degree_polynomial(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    var: &str,
) -> bool {
    let cas_ast::Expr::Mul(left, right) = ctx.get(expr) else {
        return false;
    };

    for (scale, body) in [(*left, *right), (*right, *left)] {
        let Some(scale_value) = cas_ast::views::as_rational_const(ctx, scale, 4) else {
            continue;
        };
        if !scale_value.is_integer() || scale_value.abs() <= BigRational::one() {
            continue;
        }

        let Ok(body_poly) = Polynomial::from_expr(ctx, body, var) else {
            continue;
        };
        if !(1..=2).contains(&body_poly.degree()) {
            continue;
        }
        if polynomial_integer_content_gcd(&body_poly).is_none() {
            return true;
        }
    }

    false
}

fn polynomial_integer_content_gcd(poly: &Polynomial) -> Option<BigInt> {
    let mut gcd = None;

    for coeff in &poly.coeffs {
        if coeff.is_zero() {
            continue;
        }
        if !coeff.is_integer() {
            return None;
        }

        let coeff_abs = coeff.to_integer().abs();
        gcd = Some(match gcd {
            Some(existing) => gcd_bigint(existing, coeff_abs),
            None => coeff_abs,
        });
    }

    gcd.filter(|value| *value > BigInt::one())
}

fn gcd_bigint(mut left: BigInt, mut right: BigInt) -> BigInt {
    left = left.abs();
    right = right.abs();
    while !right.is_zero() {
        let next = (&left % &right).abs();
        left = right;
        right = next;
    }
    left
}

fn should_factor_variable_common_integer_in_constant_reciprocal_denominator(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    if parent_ctx.is_expand_mode() || parent_ctx.is_auto_expand() || parent_ctx.is_solve_context() {
        return false;
    }

    if !is_low_degree_univariate_polynomial(ctx, expr) {
        return false;
    }

    parent_ctx.has_ancestor_matching(ctx, |c, ancestor| {
        matches!(
            c.get(ancestor),
            cas_ast::Expr::Div(num, den)
                if *den == expr && cas_ast::views::as_rational_const(c, *num, 8).is_some()
        )
    })
}

fn is_low_degree_univariate_polynomial(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    let variables = cas_ast::collect_variables(ctx, expr);
    let Some(var) = variables.iter().next() else {
        return false;
    };

    variables.len() == 1
        && matches!(
            Polynomial::from_expr(ctx, expr, var.as_str()),
            Ok(poly) if (1..=2).contains(&poly.degree())
        )
}

fn should_factor_variable_common_integer_in_compact_power_product(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    if parent_ctx.is_expand_mode() || parent_ctx.is_auto_expand() || parent_ctx.is_solve_context() {
        return false;
    }

    if cas_ast::collect_variables(ctx, expr).is_empty() {
        return false;
    }

    parent_ctx.has_ancestor_matching(ctx, |c, ancestor| {
        if !matches!(c.get(ancestor), cas_ast::Expr::Mul(_, _)) {
            return false;
        }

        let factors = cas_math::expr_nary::mul_leaves(c, ancestor);
        factors.contains(&expr)
            && factors
                .iter()
                .any(|&factor| is_compact_low_degree_polynomial_power(c, factor))
    })
}

fn is_compact_low_degree_polynomial_power(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    let cas_ast::Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    let Some(power) = cas_math::numeric::as_i64(ctx, *exp) else {
        return false;
    };
    if !(2..=8).contains(&power) || cas_ast::count_nodes(ctx, *base) > 25 {
        return false;
    }

    let variables = cas_ast::collect_variables(ctx, *base);
    let Some(var) = variables.iter().next() else {
        return false;
    };
    variables.len() == 1
        && matches!(
            Polynomial::from_expr(ctx, *base, var.as_str()),
            Ok(poly) if (1..=2).contains(&poly.degree())
        )
}

// SumThreeCubesZeroRule: Simplifies x³ + y³ + z³ → 3xyz when x + y + z = 0
// Classic identity: x³ + y³ + z³ - 3xyz = (x+y+z)(x²+y²+z²-xy-yz-zx)
// When x+y+z = 0, we get x³ + y³ + z³ = 3xyz
//
// This handles cyclic differences: (a-b)³ + (b-c)³ + (c-a)³ = 3(a-b)(b-c)(c-a)
// because (a-b) + (b-c) + (c-a) = 0 always
define_rule!(
    SumThreeCubesZeroRule,
    "Sum of Three Cubes (Zero Sum Identity)",
    |ctx, expr| {
        let rewrite = try_rewrite_sum_three_cubes_zero_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_sum_three_cubes_zero_desc(rewrite.kind))
                .local(expr, rewrite.rewritten),
        )
    }
);

#[cfg(test)]
mod tests;
