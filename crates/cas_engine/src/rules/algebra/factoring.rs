use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::factoring_support::{
    try_rewrite_automatic_factor_expr, try_rewrite_difference_of_squares_product_expr,
    try_rewrite_factor_common_integer_from_add_expr,
    try_rewrite_factor_difference_squares_nary_expr, try_rewrite_factor_function_expr,
    try_rewrite_sum_three_cubes_zero_expr,
};
use num_bigint::BigInt;

fn format_factor_common_integer_from_add_desc(gcd_int: &BigInt) -> String {
    format!("Factor out {}", gcd_int)
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
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    FactorRule,
    "Factor Polynomial",
    Some(crate::target_kind::TargetKindSet::FUNCTION), // Target Function expressions specifically
    PhaseMask::CORE | PhaseMask::POST,
    |ctx, expr| {
        let rewrite = try_rewrite_factor_function_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    FactorDifferenceSquaresRule,
    "Factor Difference of Squares",
    |ctx, expr| {
        let rewrite = try_rewrite_factor_difference_squares_nary_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    AutomaticFactorRule,
    "Automatic Factorization",
    |ctx, expr| {
        let rewrite = try_rewrite_automatic_factor_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
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
    |ctx, expr| {
        let rewrite = try_rewrite_factor_common_integer_from_add_expr(ctx, expr)?;
        let rewritten = rewrite.rewritten;
        let desc = format_factor_common_integer_from_add_desc(&rewrite.gcd_int);
        Some(Rewrite::new(rewritten).desc(desc).local(expr, rewritten))
    }
);

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
                .desc(rewrite.desc)
                .local(expr, rewrite.rewritten),
        )
    }
);

#[cfg(test)]
mod tests;
