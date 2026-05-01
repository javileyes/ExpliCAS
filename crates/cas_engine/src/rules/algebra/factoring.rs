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
        let rewrite = try_rewrite_factor_common_integer_from_add_expr(ctx, expr).or_else(|| {
            should_factor_variable_common_integer_in_compact_power_product(ctx, expr, parent_ctx)
                .then(|| {
                    try_rewrite_factor_common_integer_from_add_expr_with_policy(
                        ctx,
                        expr,
                        FactorCommonIntegerFromAddPolicy {
                            allow_variable_terms: true,
                        },
                    )
                })
                .flatten()
        })?;
        let rewritten = rewrite.rewritten;
        let desc = format_factor_common_integer_from_add_desc(&rewrite.gcd_int);
        Some(Rewrite::new(rewritten).desc(desc).local(expr, rewritten))
    }
);

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
