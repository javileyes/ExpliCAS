use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;

#[cfg(test)]
use cas_math::expr_relations::{
    conjugate_add_sub_pair as is_conjugate_pair,
    conjugate_nary_add_sub_pair as is_nary_conjugate_pair,
};
use cas_math::factoring_support::{
    rewrite_factor_common_integer_from_add_expr, try_rewrite_automatic_factor_expr,
    try_rewrite_difference_of_squares_product_expr,
    try_rewrite_factor_difference_squares_nary_expr, try_rewrite_factor_function_expr,
    try_rewrite_sum_three_cubes_zero_expr,
};

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
        let rewrite = rewrite_factor_common_integer_from_add_expr(ctx, expr)?;
        let rewritten = rewrite.0;
        Some(
            Rewrite::new(rewritten)
                .desc(rewrite.1)
                .local(expr, rewritten),
        )
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
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_is_nary_conjugate_pair_sophie_germain() {
        let mut ctx = cas_ast::Context::new();

        // Parse both sides of the product
        let l = parse("a^2 + 2*b^2 + 2*a*b", &mut ctx).expect("parse L");
        let r = parse("a^2 + 2*b^2 - 2*a*b", &mut ctx).expect("parse R");

        println!(
            "L = {}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: l
            }
        );
        println!(
            "R = {}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: r
            }
        );

        let result = is_nary_conjugate_pair(&mut ctx, l, r);

        println!("Result: {:?}", result);

        assert!(result.is_some(), "Should detect conjugate pair");

        if let Some((u, v)) = result {
            println!(
                "U = {}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: u
                }
            );
            println!(
                "V = {}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: v
                }
            );
        }
    }

    #[test]
    fn test_is_conjugate_pair_simple() {
        let mut ctx = cas_ast::Context::new();

        let l = parse("x + 1", &mut ctx).expect("parse L");
        let r = parse("x - 1", &mut ctx).expect("parse R");

        let result = is_conjugate_pair(&ctx, l, r);

        assert!(result.is_some(), "Should detect simple conjugate pair");
    }

    #[test]
    fn test_difference_of_squares_rule_on_product() {
        use crate::parent_context::ParentContext;
        use crate::rule::Rule;

        let mut ctx = cas_ast::Context::new();

        // Parse the full product
        let expr =
            parse("(a^2 + 2*b^2 + 2*a*b)*(a^2 + 2*b^2 - 2*a*b)", &mut ctx).expect("parse product");

        println!(
            "Product = {}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: expr
            }
        );

        // Apply the rule directly
        let rule = DifferenceOfSquaresRule;
        let parent_ctx = ParentContext::root();
        let result = rule.apply(&mut ctx, expr, &parent_ctx);

        println!("Rule result: {:?}", result.is_some());

        assert!(
            result.is_some(),
            "DifferenceOfSquaresRule should match the product"
        );

        if let Some(rewrite) = result {
            println!(
                "Rewrite: {} -> {}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: expr
                },
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                },
            );
        }
    }

    #[test]
    fn test_difference_of_squares_reordered_terms() {
        use crate::parent_context::ParentContext;
        use crate::rule::Rule;

        let mut ctx = cas_ast::Context::new();

        // Parse with the order as appears after canonicalization
        // Note: The REPL shows (a² + 2·b² - 2*a·b)·(a² + 2·b² + 2·a·b)
        let expr =
            parse("(a^2 + 2*b^2 - 2*a*b)*(a^2 + 2*b^2 + 2*a*b)", &mut ctx).expect("parse product");

        println!(
            "Product (reordered) = {}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: expr
            }
        );

        // Apply the rule directly
        let rule = DifferenceOfSquaresRule;
        let parent_ctx = ParentContext::root();
        let result = rule.apply(&mut ctx, expr, &parent_ctx);

        println!("Rule result: {:?}", result.is_some());

        // This should also match because (U-V)(U+V) is the same as (U+V)(U-V)
        assert!(
            result.is_some(),
            "DifferenceOfSquaresRule should match the reordered product"
        );

        if let Some(rewrite) = result {
            println!(
                "Rewrite: {} -> {}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: expr
                },
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                },
            );
        }
    }

    #[test]
    fn test_simplifier_applies_difference_of_squares() {
        use crate::parent_context::ParentContext;
        use crate::rule::Rule;
        use crate::Simplifier;

        // Create simplifier with default rules (includes DifferenceOfSquaresRule)
        let mut simplifier = Simplifier::with_default_rules();

        // Parse the product
        let expr = parse(
            "(a^2 + 2*b^2 + 2*a*b)*(a^2 + 2*b^2 - 2*a*b)",
            &mut simplifier.context,
        )
        .expect("parse");

        println!(
            "Input: {}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: expr
            }
        );

        // Run simplifier
        let (result, steps) = simplifier.simplify(expr);

        println!(
            "Output: {}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        println!("Number of steps: {}", steps.len());
        for step in &steps {
            println!("  Step: {}", step.rule_name);
        }

        // Now try to apply DifferenceOfSquaresRule directly to the OUTPUT
        // to see if it would have matched if given a chance
        let rule = DifferenceOfSquaresRule;
        let parent_ctx = ParentContext::root();
        let rule_result = rule.apply(&mut simplifier.context, result, &parent_ctx);

        println!(
            "DifferenceOfSquaresRule direct application to OUTPUT: {:?}",
            rule_result.is_some()
        );

        // Inspect the structure of result
        // Extract ExprIds first to avoid borrow conflicts
        let factors = {
            match simplifier.context.get(result) {
                cas_ast::Expr::Mul(l, r) => Some((*l, *r)),
                _ => None,
            }
        };

        if let Some((l, r)) = factors {
            // Verify that we can identify this as a conjugate pair and the rule applies
            let conjugate = is_nary_conjugate_pair(&mut simplifier.context, l, r);
            // After the fix, the conjugate pair should be recognized
            assert!(
                conjugate.is_some(),
                "is_nary_conjugate_pair should recognize the canonicalized conjugate pair"
            );
        }

        // Verify that DifferenceOfSquaresRule was applied (indicated by step name)
        let dos_applied = steps
            .iter()
            .any(|s| s.rule_name.starts_with("Difference of Squares"));
        assert!(
            dos_applied,
            "DifferenceOfSquaresRule should be applied during simplification"
        );
    }
}
