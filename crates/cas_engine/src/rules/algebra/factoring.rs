use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::count_nodes;
use cas_ast::Expr;

use cas_math::expr_relations::{
    conjugate_add_sub_pair as is_conjugate_pair,
    conjugate_nary_add_sub_pair as is_nary_conjugate_pair, is_structurally_zero,
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
        // Match Mul(l, r) where l and r are conjugate binomials
        if let Expr::Mul(l, r) = ctx.get(expr) {
            let l = *l;
            let r = *r;

            // Try fast binary matcher first
            if let Some((a, b)) = is_conjugate_pair(ctx, l, r) {
                // Create a² - b²
                let two = ctx.num(2);
                let a_squared = ctx.add(Expr::Pow(a, two));
                let b_squared = ctx.add(Expr::Pow(b, two));
                let new_expr = ctx.add(Expr::Sub(a_squared, b_squared));

                return Some(Rewrite::new(new_expr).desc("(a-b)(a+b) = a² - b²"));
            }

            // Try N-ary matcher for sums with 3+ terms
            if let Some((u, v)) = is_nary_conjugate_pair(ctx, l, r) {
                // Create U² - V²
                let two = ctx.num(2);
                let u_squared = ctx.add(Expr::Pow(u, two));
                let v_squared = ctx.add(Expr::Pow(v, two));
                let new_expr = ctx.add(Expr::Sub(u_squared, v_squared));

                return Some(
                    Rewrite::new(new_expr).desc("(U+V)(U-V) = U² - V² (conjugate product)"),
                );
            }

            // N-ary product scan: flatten Mul chains and look for conjugate factor pairs.
            // E.g. (sqrt(5)+sqrt(2)) * (sqrt(5)-sqrt(2)) * (x+1)^6 → 3 * (x+1)^6
            let factors = flatten_mul_factors(ctx, expr);
            if factors.len() >= 3 {
                // Scan all pairs for conjugate matches (O(n²), but n is small)
                for i in 0..factors.len() {
                    for j in (i + 1)..factors.len() {
                        let fi = factors[i];
                        let fj = factors[j];

                        // Try binary conjugate
                        let conjugate = is_conjugate_pair(ctx, fi, fj)
                            .or_else(|| is_nary_conjugate_pair(ctx, fi, fj));

                        if let Some((a, b)) = conjugate {
                            // Build a² - b²
                            let two = ctx.num(2);
                            let a_sq = ctx.add(Expr::Pow(a, two));
                            let b_sq = ctx.add(Expr::Pow(b, two));
                            let dos = ctx.add(Expr::Sub(a_sq, b_sq));

                            // Rebuild product: dos * remaining factors
                            let mut result = dos;
                            for (k, &fk) in factors.iter().enumerate() {
                                if k != i && k != j {
                                    result = ctx.add(Expr::Mul(result, fk));
                                }
                            }

                            return Some(
                                Rewrite::new(result).desc("(a-b)(a+b)·… = (a²-b²)·… (n-ary scan)"),
                            );
                        }
                    }
                }
            }
        }
        None
    }
);

/// Flatten a binary Mul chain into a list of leaf factors.
/// E.g. Mul(Mul(a, b), c) → [a, b, c]
fn flatten_mul_factors(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Vec<cas_ast::ExprId> {
    let mut factors = Vec::new();
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        if let Expr::Mul(l, r) = ctx.get(e) {
            stack.push(*r);
            stack.push(*l);
        } else {
            factors.push(e);
        }
    }
    factors
}

define_rule!(
    FactorRule,
    "Factor Polynomial",
    Some(crate::target_kind::TargetKindSet::FUNCTION), // Target Function expressions specifically
    PhaseMask::CORE | PhaseMask::POST,
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let name = ctx.sym_name(*fn_id);
            if name == "factor" && args.len() == 1 {
                let arg = args[0];
                // Use the general factor entry point which tries polynomial then diff squares
                let new_expr = crate::factor::factor(ctx, arg);
                if new_expr != arg {
                    // Wrap in __hold() to prevent other rules from undoing the factorization
                    // (e.g., DifferenceOfSquaresRule converts (a-b)(a+b) back to a²-b²)
                    let held = cas_ast::hold::wrap_hold(ctx, new_expr);
                    return Some(Rewrite::new(held).desc("Factorization"));
                }
            }
        }
        None
    }
);

define_rule!(
    FactorDifferenceSquaresRule,
    "Factor Difference of Squares",
    |ctx, expr| {
        // match Expr::Add(l, r)
        if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
            return None;
        }

        // 2. Flatten the chain
        let terms = crate::nary::add_leaves(ctx, expr);

        // 3. Separate into potential squares and negative squares
        // We look for pairs (A, B) where A is a square and B is a negative square (B = -C^2)
        // i.e. A - C^2.

        // Optimization: We only need to find ONE pair that works.
        // We can iterate O(N^2) or sort? O(N^2) is fine for small N.

        for i in 0..terms.len() {
            for j in 0..terms.len() {
                if i == j {
                    continue;
                }

                let t1 = terms[i];
                let t2 = terms[j];

                // Check if t1 + t2 forms a difference of squares
                // We construct a temporary Add(t1, t2) and call factor_difference_squares
                // This reuses the existing logic (including get_square_root and is_pythagorean)
                let pair = ctx.add(Expr::Add(t1, t2));

                if let Some(factored) = crate::factor::factor_difference_squares(ctx, pair) {
                    // Complexity check
                    let old_count = count_nodes(ctx, pair);
                    let new_count = count_nodes(ctx, factored);

                    // Smart Check:
                    // If the result is a Mul, it means we factored into (A-B)(A+B).
                    // This usually increases complexity and blocks cancellation (due to DistributeRule guard).
                    // So we require STRICT reduction (<).
                    //
                    // If the result is NOT a Mul, it means we used a Pythagorean identity (sin^2+cos^2=1).
                    // The result is just (A-B). This is a simplification we want, even if size is same.
                    // So we allow SAME size (<=).

                    let is_mul = matches!(ctx.get(factored), Expr::Mul(_, _));
                    let allowed = if is_mul {
                        new_count < old_count
                    } else {
                        new_count <= old_count
                    };

                    if allowed {
                        // Found a pair!
                        // Construct the new expression: Factored + (Terms - {t1, t2})
                        let mut new_terms = Vec::new();
                        new_terms.push(factored);
                        for (k, &term) in terms.iter().enumerate() {
                            if k != i && k != j {
                                new_terms.push(term);
                            }
                        }

                        // Rebuild Add chain
                        if new_terms.is_empty() {
                            return Some(
                                Rewrite::new(ctx.num(0))
                                    .desc("Factor difference of squares (Empty)"),
                            );
                        }

                        let mut new_expr = new_terms[0];
                        for t in new_terms.iter().skip(1) {
                            new_expr = ctx.add(Expr::Add(new_expr, *t));
                        }

                        return Some(
                            Rewrite::new(new_expr).desc("Factor difference of squares (N-ary)"),
                        );
                    }
                }
            }
        }
        None
    }
);

define_rule!(
    AutomaticFactorRule,
    "Automatic Factorization",
    |ctx, expr| {
        // Only try to factor if it's an Add or Sub (polynomial-like)
        match ctx.get(expr) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {}
            _ => return None,
        }

        // Try factor_polynomial first
        if let Some(new_expr) = crate::factor::factor_polynomial(ctx, expr) {
            if new_expr != expr {
                // Complexity check: Only accept if it strictly reduces size
                // This prevents loops with ExpandRule which usually increases size (or keeps it same)
                let old_count = count_nodes(ctx, expr);
                let new_count = count_nodes(ctx, new_expr);

                if new_count < old_count {
                    // Check for structural equality (though unlikely if count reduced)
                    if crate::ordering::compare_expr(ctx, new_expr, expr)
                        == std::cmp::Ordering::Equal
                    {
                        return None;
                    }
                    return Some(
                        Rewrite::new(new_expr).desc("Automatic Factorization (Reduced Size)"),
                    );
                }
            }
        }

        // Try difference of squares
        // Note: Diff squares usually increases size: a^2 - b^2 (5) -> (a-b)(a+b) (7)
        // So this will rarely trigger with strict size check unless terms simplify further.
        // e.g. x^4 - 1 -> (x^2-1)(x^2+1) -> (x-1)(x+1)(x^2+1).
        // x^4 - 1 (5 nodes). (x-1)(x+1)(x^2+1) (many nodes).
        // So auto-factoring diff squares is risky for loops.
        // We'll skip it for now in AutomaticFactorRule, or only if it reduces size.
        if let Some(new_expr) = crate::factor::factor_difference_squares(ctx, expr) {
            if new_expr != expr {
                let old_count = count_nodes(ctx, expr);
                let new_count = count_nodes(ctx, new_expr);
                if new_count < old_count {
                    return Some(
                        Rewrite::new(new_expr).desc("Automatic Factorization (Diff Squares)"),
                    );
                }
            }
        }

        None
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
        use cas_math::numeric::gcd_rational;
        use num_rational::BigRational;
        use num_traits::{One, Signed};

        // Only match simple binary Add(a, b)
        let (l, r) = match ctx.get(expr) {
            Expr::Add(l, r) => (*l, *r),
            _ => return None,
        };

        // Extract integer coefficient from a term
        fn get_int_coef(ctx: &cas_ast::Context, term: cas_ast::ExprId) -> Option<BigRational> {
            match ctx.get(term) {
                Expr::Number(n) if n.is_integer() => Some(n.clone()),
                Expr::Mul(a, b) => {
                    if let Expr::Number(n) = ctx.get(*a) {
                        if n.is_integer() {
                            return Some(n.clone());
                        }
                    }
                    if let Expr::Number(n) = ctx.get(*b) {
                        if n.is_integer() {
                            return Some(n.clone());
                        }
                    }
                    None
                }
                Expr::Neg(inner) => get_int_coef(ctx, *inner).map(|c| -c),
                _ => None,
            }
        }

        // Get coefficients
        let coef_l = get_int_coef(ctx, l);
        let coef_r = get_int_coef(ctx, r);

        let coef_l = coef_l?;
        let coef_r = coef_r?;

        if cas_math::expr_predicates::contains_variable(ctx, l)
            || cas_math::expr_predicates::contains_variable(ctx, r)
        {
            return None;
        }

        // Compute GCD of absolute values
        let gcd = gcd_rational(coef_l.abs(), coef_r.abs());
        if gcd <= BigRational::one() {
            return None;
        }

        // Check GCD is at least 2
        let gcd_int = gcd.to_integer();
        if gcd_int <= num_bigint::BigInt::from(1) {
            return None;
        }

        // Divide coefficients by GCD
        fn divide_term(
            ctx: &mut cas_ast::Context,
            term: cas_ast::ExprId,
            gcd: &BigRational,
        ) -> cas_ast::ExprId {
            match ctx.get(term) {
                Expr::Number(n) => {
                    let new_n = n / gcd;
                    ctx.add(Expr::Number(new_n))
                }
                Expr::Mul(a, b) => {
                    let (a, b) = (*a, *b);
                    if let Expr::Number(n) = ctx.get(a) {
                        let new_n = n / gcd;
                        if new_n.is_one() {
                            return b;
                        }
                        let num = ctx.add(Expr::Number(new_n));
                        return ctx.add_raw(Expr::Mul(num, b));
                    }
                    if let Expr::Number(n) = ctx.get(b) {
                        let new_n = n / gcd;
                        if new_n.is_one() {
                            return a;
                        }
                        let num = ctx.add(Expr::Number(new_n));
                        return ctx.add_raw(Expr::Mul(a, num));
                    }
                    term
                }
                Expr::Neg(inner) => {
                    let inner = *inner;
                    let divided = divide_term(ctx, inner, gcd);
                    ctx.add(Expr::Neg(divided))
                }
                _ => term,
            }
        }

        let new_l = divide_term(ctx, l, &gcd);
        let new_r = divide_term(ctx, r, &gcd);
        let inner = ctx.add_raw(Expr::Add(new_l, new_r));
        let gcd_expr = ctx.add(Expr::Number(gcd.clone()));
        let new_expr = ctx.add_raw(Expr::Mul(gcd_expr, inner));

        // Note: We intentionally allow node count to increase for GCD factoring
        // because 2*(3 + 2*√5) is mathematically cleaner than 6 + 4*√5

        Some(
            Rewrite::new(new_expr)
                .desc_lazy(|| format!("Factor out {}", gcd_int))
                .local(expr, new_expr),
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
        // Flatten the sum
        let terms = crate::nary::add_leaves(ctx, expr);

        // We need at least 3 cube terms (and no more for the pure identity)
        // For safety, only match exactly 3 cubes with no other terms
        if terms.len() != 3 {
            return None;
        }

        // Extract bases from cubes: term must be Pow(base, 3)
        let mut bases: Vec<cas_ast::ExprId> = Vec::new();
        for &term in &terms {
            let (base, is_neg) = match ctx.get(term) {
                Expr::Pow(b, e) => {
                    let (b, e) = (*b, *e);
                    if let Expr::Number(n) = ctx.get(e) {
                        if n.is_integer() && n.to_integer() == num_bigint::BigInt::from(3) {
                            (b, false)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                Expr::Neg(inner) => {
                    let inner = *inner;
                    // Handle -(x^3) form
                    if let Expr::Pow(b, e) = ctx.get(inner) {
                        let (b, e) = (*b, *e);
                        if let Expr::Number(n) = ctx.get(e) {
                            if n.is_integer() && n.to_integer() == num_bigint::BigInt::from(3) {
                                (b, true)
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                _ => return None,
            };

            // For negative cubes (-x³), we need to negate the base
            if is_neg {
                let neg_base = ctx.add(Expr::Neg(base));
                bases.push(neg_base);
            } else {
                bases.push(base);
            }
        }

        // Check if bases sum to zero structurally
        // Build the sum x + y + z and try to simplify it
        let sum_bases = {
            let xy = ctx.add(Expr::Add(bases[0], bases[1]));
            ctx.add(Expr::Add(xy, bases[2]))
        };

        // Try to prove the sum is zero by simplifying
        if !is_structurally_zero(ctx, sum_bases) {
            return None;
        }

        // Sum of bases is zero! Apply the identity: x³ + y³ + z³ = 3xyz
        let three = ctx.num(3);
        // Build 3 * x * y * z
        let xy = cas_math::expr_rewrite::smart_mul(ctx, bases[0], bases[1]);
        let xyz = cas_math::expr_rewrite::smart_mul(ctx, xy, bases[2]);
        let inner_result = cas_math::expr_rewrite::smart_mul(ctx, three, xyz);
        // Wrap in __hold to prevent DistributeRule from expanding
        let result = cas_ast::hold::wrap_hold(ctx, inner_result);

        Some(
            Rewrite::new(result)
                .desc("x³ + y³ + z³ = 3xyz (when x + y + z = 0)")
                .local(expr, result),
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
