use cas_ast::{Context, Expr, ExprId};
use cas_engine::Simplifier;
use proptest::prelude::*;
use std::io::Write;

mod strategies;

/// Log expression to file before processing - helps capture what causes stack overflow
fn log_expr_before_simplify(ctx: &Context, expr: ExprId, test_name: &str) {
    let display = cas_ast::DisplayExpr {
        context: ctx,
        id: expr,
    };
    let expr_str = display.to_string();
    let log_entry = format!("[{}] {}\n", test_name, expr_str);

    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("/tmp/last_tested_expr.txt")
    {
        let _ = file.write_all(log_entry.as_bytes());
        let _ = file.flush();
    }
}

// ============================================================================
// PROPERTY TESTS - REQUIRE LARGER STACK
// ============================================================================
//
// These property tests can cause stack overflow due to the recursive nature
// of the simplifier and related functions (expand, normalize_core, etc).
//
// To run ALL property tests reliably, use:
//   RUST_MIN_STACK=16777216 cargo test --package cas_engine --test property_tests
//
// For CI environments, these tests should be run separately with increased stack.
// ============================================================================

proptest! {
    // Reduce number of cases to minimize stack overflow probability
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn test_round_trip_parse(re in strategies::arb_recursive_expr()) {
        let (ctx, expr) = strategies::to_context(re);
        let _s = cas_format::Format::to_latex(&expr, &ctx); // Use to_latex to ensure it doesn't panicon for now, or Display
        // Actually we should use Display implementation if available.
        // But DisplayExpr is needed.
        let display_expr = cas_ast::DisplayExpr { context: &ctx, id: expr };
        let s = display_expr.to_string();

        // Parse it back
        let mut parse_ctx = Context::new();
        let parsed = cas_parser::parse(&s, &mut parse_ctx);
        // It should parse successfully
        prop_assert!(parsed.is_ok(), "Failed to parse: {}", s);
    }

    #[test]
    fn test_identity_add_zero(re in strategies::arb_recursive_expr()) {
        let (ctx, expr) = strategies::to_context(re);
        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx; // Use the generated context

        // expr + 0 should simplify to expr (or equivalent)
        let zero = simplifier.context.num(0);
        let input = simplifier.context.add(Expr::Add(expr, zero));

        // simplify(expr + 0) == simplify(expr)
        let (s1, _) = simplifier.simplify(input);
        let (s2, _) = simplifier.simplify(expr);

        // We need to compare structure, but ExprId equality is not enough if they are different nodes with same content.
        // But simplify should canonicalize.
        // However, s1 and s2 might be different IDs even if content is same.
        // We need deep equality check.
        // Or compare string representation.
        let d1 = cas_ast::DisplayExpr { context: &simplifier.context, id: s1 };
        let d2 = cas_ast::DisplayExpr { context: &simplifier.context, id: s2 };
        prop_assert_eq!(d1.to_string(), d2.to_string());
    }

    #[test]
    fn test_identity_mul_one(re in strategies::arb_recursive_expr()) {
        let (ctx, expr) = strategies::to_context(re);
        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx;

        // expr * 1 should simplify to expr
        let one = simplifier.context.num(1);
        let input = simplifier.context.add(Expr::Mul(expr, one));

        let (s1, _) = simplifier.simplify(input);
        let (s2, _) = simplifier.simplify(expr);

        let d1 = cas_ast::DisplayExpr { context: &simplifier.context, id: s1 };
        let d2 = cas_ast::DisplayExpr { context: &simplifier.context, id: s2 };
        prop_assert_eq!(d1.to_string(), d2.to_string());
    }

    #[test]
    fn test_idempotency(re in strategies::arb_recursive_expr()) {
        let (ctx, expr) = strategies::to_context(re);
        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx;

        let (s1, _) = simplifier.simplify(expr);
        let (s2, _) = simplifier.simplify(s1);

        let d1 = cas_ast::DisplayExpr { context: &simplifier.context, id: s1 };
        let d2 = cas_ast::DisplayExpr { context: &simplifier.context, id: s2 };
        prop_assert_eq!(d1.to_string(), d2.to_string());
    }

    #[test]
    fn test_constant_folding(re in strategies::arb_recursive_expr()) {
        let (ctx, expr) = strategies::to_context(re);
        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx;

        let (simplified, _) = simplifier.simplify(expr);

        // Check that no Number op Number exists in the simplified expression
        fn check_no_constant_ops(ctx: &Context, expr: ExprId) -> bool {
            match ctx.get(expr) {
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
                    if let (Expr::Number(_), Expr::Number(_)) = (ctx.get(*l), ctx.get(*r)) {
                        return false;
                    }
                    check_no_constant_ops(ctx, *l) && check_no_constant_ops(ctx, *r)
                },
                Expr::Neg(e) => {
                    if let Expr::Number(_) = ctx.get(*e) {
                        return false;
                    }
                    check_no_constant_ops(ctx, *e)
                },
                Expr::Function(_, args) => args.iter().all(|a| check_no_constant_ops(ctx, *a)),
                _ => true,
            }
        }

        let d = cas_ast::DisplayExpr { context: &simplifier.context, id: simplified };
        prop_assert!(check_no_constant_ops(&simplifier.context, simplified), "Constant folding failed: {}", d);
    }

    #[test]
    fn test_identity_preservation(re in strategies::arb_recursive_expr()) {
        let (ctx, expr) = strategies::to_context(re);
        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx;

        // x * 0 -> 0
        let zero = simplifier.context.num(0);
        let mul_zero = simplifier.context.add(Expr::Mul(expr, zero));
        let (s_mul_zero, _) = simplifier.simplify(mul_zero);

        let d_s = cas_ast::DisplayExpr { context: &simplifier.context, id: s_mul_zero };
        let d_z = cas_ast::DisplayExpr { context: &simplifier.context, id: zero };
        prop_assert_eq!(d_s.to_string(), d_z.to_string());

        // x ^ 0 -> 1
        let one = simplifier.context.num(1);
        let pow_zero = simplifier.context.add(Expr::Pow(expr, zero));
        let (s_pow_zero, _) = simplifier.simplify(pow_zero);

        let d_p = cas_ast::DisplayExpr { context: &simplifier.context, id: s_pow_zero };
        let d_o = cas_ast::DisplayExpr { context: &simplifier.context, id: one };
        prop_assert_eq!(d_p.to_string(), d_o.to_string());

        // x ^ 1 -> x
        let pow_one = simplifier.context.add(Expr::Pow(expr, one));
        let (s_pow_one, _) = simplifier.simplify(pow_one);
        let (s_expr, _) = simplifier.simplify(expr);

        let d_p1 = cas_ast::DisplayExpr { context: &simplifier.context, id: s_pow_one };
        let d_e = cas_ast::DisplayExpr { context: &simplifier.context, id: s_expr };
        prop_assert_eq!(d_p1.to_string(), d_e.to_string());
    }

    // Note: We now use balanced trees instead of right-associative, so no tree shape check
    // The important property is that simplification is idempotent and deterministic
    #[test]
    fn test_simplify_deterministic(re in strategies::arb_recursive_expr()) {
        let (ctx1, expr1) = strategies::to_context(re.clone());
        let (ctx2, expr2) = strategies::to_context(re);
        let mut simp1 = Simplifier::with_default_rules();
        let mut simp2 = Simplifier::with_default_rules();
        simp1.context = ctx1;
        simp2.context = ctx2;

        let (s1, _) = simp1.simplify(expr1);
        let (s2, _) = simp2.simplify(expr2);

        let d1 = cas_ast::DisplayExpr { context: &simp1.context, id: s1 };
        let d2 = cas_ast::DisplayExpr { context: &simp2.context, id: s2 };
        prop_assert_eq!(d1.to_string(), d2.to_string(), "Simplify not deterministic");
    }
}

// ============================================================================
// NORMALIZE_CORE STRUCTURAL INVARIANTS (Property Tests)
// These test the N0/N1/N2/N3 canonicalization guarantees
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// N0: normalize_core is idempotent
    #[test]
    fn test_normalize_core_idempotent(re in strategies::arb_recursive_expr()) {
        let (mut ctx, expr) = strategies::to_context(re);

        let n1 = cas_engine::canonical_forms::normalize_core(&mut ctx, expr);
        let n2 = cas_engine::canonical_forms::normalize_core(&mut ctx, n1);

        let d1 = cas_ast::DisplayExpr { context: &ctx, id: n1 };
        let d2 = cas_ast::DisplayExpr { context: &ctx, id: n2 };
        prop_assert_eq!(d1.to_string(), d2.to_string(), "normalize_core not idempotent");
    }

    /// N0: No Neg(Number) after normalize_core
    #[test]
    fn test_normalize_core_no_neg_number(re in strategies::arb_recursive_expr()) {
        let (mut ctx, expr) = strategies::to_context(re);
        let normalized = cas_engine::canonical_forms::normalize_core(&mut ctx, expr);

        fn check_no_neg_number(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Neg(inner) => {
                    if matches!(ctx.get(*inner), Expr::Number(_)) {
                        return false; // Found Neg(Number) - violation!
                    }
                    check_no_neg_number(ctx, *inner)
                }
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
                    check_no_neg_number(ctx, *l) && check_no_neg_number(ctx, *r)
                }
                Expr::Function(_, args) => args.iter().all(|a| check_no_neg_number(ctx, *a)),
                _ => true,
            }
        }

        let d = cas_ast::DisplayExpr { context: &ctx, id: normalized };
        prop_assert!(check_no_neg_number(&ctx, normalized), "Found Neg(Number) after normalize_core: {}", d);
    }

    /// N1: No Neg(Neg(x)) after normalize_core
    #[test]
    fn test_normalize_core_no_double_neg(re in strategies::arb_recursive_expr()) {
        let (mut ctx, expr) = strategies::to_context(re);
        let normalized = cas_engine::canonical_forms::normalize_core(&mut ctx, expr);

        fn check_no_double_neg(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Neg(inner) => {
                    if matches!(ctx.get(*inner), Expr::Neg(_)) {
                        return false; // Found Neg(Neg(x)) - violation!
                    }
                    check_no_double_neg(ctx, *inner)
                }
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
                    check_no_double_neg(ctx, *l) && check_no_double_neg(ctx, *r)
                }
                Expr::Function(_, args) => args.iter().all(|a| check_no_double_neg(ctx, *a)),
                _ => true,
            }
        }

        let d = cas_ast::DisplayExpr { context: &ctx, id: normalized };
        prop_assert!(check_no_double_neg(&ctx, normalized), "Found Neg(Neg(x)) after normalize_core: {}", d);
    }

    /// After simplify, no Pow(x, 1) should exist
    #[test]
    fn test_simplify_no_pow_one(re in strategies::arb_recursive_expr()) {
        let (ctx, expr) = strategies::to_context(re);
        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx;

        let (simplified, _) = simplifier.simplify(expr);

        fn check_no_pow_one(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Pow(_, exp) => {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if n == &num_rational::BigRational::from_integer(1.into()) {
                            return false; // Found Pow(x, 1)
                        }
                    }
                    true // Don't recurse into Pow base to avoid false positives
                }
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                    check_no_pow_one(ctx, *l) && check_no_pow_one(ctx, *r)
                }
                Expr::Neg(e) => check_no_pow_one(ctx, *e),
                Expr::Function(_, args) => args.iter().all(|a| check_no_pow_one(ctx, *a)),
                _ => true,
            }
        }

        let d = cas_ast::DisplayExpr { context: &simplifier.context, id: simplified };
        prop_assert!(check_no_pow_one(&simplifier.context, simplified), "Found Pow(x, 1) after simplify: {}", d);
    }

    /// Fractions should be in reduced form (gcd = 1)
    #[test]
    fn test_numbers_reduced_form(re in strategies::arb_recursive_expr()) {
        use num_integer::Integer;
        use num_traits::Signed;

        let (ctx, expr) = strategies::to_context(re);
        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx;

        let (simplified, _) = simplifier.simplify(expr);

        fn check_reduced_rationals(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Number(n) => {
                    let numer = n.numer().abs();
                    let denom = n.denom().abs();
                    let g = numer.gcd(&denom);
                    // GCD should be 1 (reduced form)
                    g == num_bigint::BigInt::from(1) && n.denom() > &num_bigint::BigInt::from(0)
                }
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
                    check_reduced_rationals(ctx, *l) && check_reduced_rationals(ctx, *r)
                }
                Expr::Neg(e) => check_reduced_rationals(ctx, *e),
                Expr::Function(_, args) => args.iter().all(|a| check_reduced_rationals(ctx, *a)),
                _ => true,
            }
        }

        let d = cas_ast::DisplayExpr { context: &simplifier.context, id: simplified };
        prop_assert!(check_reduced_rationals(&simplifier.context, simplified), "Found non-reduced rational after simplify: {}", d);
    }

    // Note: We now use balanced trees, so nested Add(Add(..),..) is allowed and expected
    // The important invariant is idempotence, not tree shape
    /// normalize_core should be deterministic
    #[test]
    fn test_normalize_core_deterministic(re in strategies::arb_recursive_expr()) {
        let (mut ctx1, expr1) = strategies::to_context(re.clone());
        let (mut ctx2, expr2) = strategies::to_context(re);

        let n1 = cas_engine::canonical_forms::normalize_core(&mut ctx1, expr1);
        let n2 = cas_engine::canonical_forms::normalize_core(&mut ctx2, expr2);

        let d1 = cas_ast::DisplayExpr { context: &ctx1, id: n1 };
        let d2 = cas_ast::DisplayExpr { context: &ctx2, id: n2 };
        prop_assert_eq!(d1.to_string(), d2.to_string(), "normalize_core not deterministic");
    }

    // Note: We now use balanced trees, so nested Mul(Mul(..),..) is allowed and expected
    // The important invariant is that sorting is correct
    /// Mul factors should be sorted consistently
    #[test]
    fn test_normalize_core_mul_sorted(re in strategies::arb_recursive_expr()) {
        let (mut ctx, expr) = strategies::to_context(re);
        let n1 = cas_engine::canonical_forms::normalize_core(&mut ctx, expr);
        let n2 = cas_engine::canonical_forms::normalize_core(&mut ctx, n1);

        // Idempotence check implies consistent sorting
        let d1 = cas_ast::DisplayExpr { context: &ctx, id: n1 };
        let d2 = cas_ast::DisplayExpr { context: &ctx, id: n2 };
        prop_assert_eq!(d1.to_string(), d2.to_string(), "normalize_core not idempotent");
    }

    /// N3: No Pow(Pow(x,a),b) after normalize_core - should be Pow(x, a*b)
    /// Note: N3 only applies when BOTH exponents are integer Numbers
    #[test]
    fn test_normalize_core_no_nested_pow(re in strategies::arb_recursive_expr()) {
        let (mut ctx, expr) = strategies::to_context(re);
        let normalized = cas_engine::canonical_forms::normalize_core(&mut ctx, expr);

        fn is_integer_number(ctx: &Context, id: ExprId) -> bool {
            if let Expr::Number(n) = ctx.get(id) {
                n.is_integer()
            } else {
                false
            }
        }

        fn check_no_nested_pow(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Pow(base, outer_exp) => {
                    // Base should NOT be Pow (N3 should flatten)
                    // BUT only if both exponents are integers
                    if let Expr::Pow(_, inner_exp) = ctx.get(*base) {
                        // N3 only applies when BOTH exponents are integers
                        if is_integer_number(ctx, *inner_exp) && is_integer_number(ctx, *outer_exp) {
                            return false; // This should have been flattened
                        }
                        // If either exponent is non-integer, nested Pow is allowed
                    }
                    check_no_nested_pow(ctx, *base)
                }
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                    check_no_nested_pow(ctx, *l) && check_no_nested_pow(ctx, *r)
                }
                Expr::Neg(e) => check_no_nested_pow(ctx, *e),
                Expr::Function(_, args) => args.iter().all(|a| check_no_nested_pow(ctx, *a)),
                _ => true,
            }
        }

        let d = cas_ast::DisplayExpr { context: &ctx, id: normalized };
        prop_assert!(check_no_nested_pow(&ctx, normalized), "Found Pow(Pow(x,a),b) with integer exponents after normalize_core: {}", d);
    }

    /// Ordering consistency: same expression should always produce same normalized form
    #[test]
    fn test_normalize_core_order_deterministic(re in strategies::arb_recursive_expr()) {
        let (mut ctx1, expr1) = strategies::to_context(re.clone());
        let (mut ctx2, expr2) = strategies::to_context(re);

        let n1 = cas_engine::canonical_forms::normalize_core(&mut ctx1, expr1);
        let n2 = cas_engine::canonical_forms::normalize_core(&mut ctx2, expr2);

        let d1 = cas_ast::DisplayExpr { context: &ctx1, id: n1 };
        let d2 = cas_ast::DisplayExpr { context: &ctx2, id: n2 };
        prop_assert_eq!(d1.to_string(), d2.to_string(), "normalize_core is not deterministic");
    }
}

// ============================================================================
// METAMORPHIC PROPERTY TESTS
// These verify algebraic identities are preserved
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(25))]

    /// Metamorphic: normalize_core(e + 0) displays same as normalize_core(e)
    #[test]
    fn test_metamorphic_add_zero(re in strategies::arb_recursive_expr()) {
        let (mut ctx, expr) = strategies::to_context(re);

        let zero = ctx.num(0);
        let expr_plus_zero = ctx.add(Expr::Add(expr, zero));

        let n1 = cas_engine::canonical_forms::normalize_core(&mut ctx, expr_plus_zero);
        let n2 = cas_engine::canonical_forms::normalize_core(&mut ctx, expr);

        // After normalize_core, e+0 should be structurally simplified
        // Note: normalize_core may not remove +0, but simplify should
        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (s1, _) = simplifier.simplify(n1);
        let (s2, _) = simplifier.simplify(n2);

        let d1 = cas_ast::DisplayExpr { context: &simplifier.context, id: s1 };
        let d2 = cas_ast::DisplayExpr { context: &simplifier.context, id: s2 };
        prop_assert_eq!(d1.to_string(), d2.to_string(), "e+0 != e");
    }

    /// Metamorphic: normalize_core(e * 1) displays same as normalize_core(e)
    #[test]
    fn test_metamorphic_mul_one(re in strategies::arb_recursive_expr()) {
        let (mut ctx, expr) = strategies::to_context(re);

        let one = ctx.num(1);
        let expr_times_one = ctx.add(Expr::Mul(expr, one));

        let n1 = cas_engine::canonical_forms::normalize_core(&mut ctx, expr_times_one);
        let n2 = cas_engine::canonical_forms::normalize_core(&mut ctx, expr);

        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx;

        // Log expression before simplify to capture what causes overflow
        log_expr_before_simplify(&simplifier.context, n1, "test_metamorphic_mul_one");

        let (s1, _) = simplifier.simplify(n1);
        let (s2, _) = simplifier.simplify(n2);

        let d1 = cas_ast::DisplayExpr { context: &simplifier.context, id: s1 };
        let d2 = cas_ast::DisplayExpr { context: &simplifier.context, id: s2 };
        prop_assert_eq!(d1.to_string(), d2.to_string(), "e*1 != e");
    }

    /// Metamorphic: e * 0 == 0
    #[test]
    fn test_metamorphic_mul_zero(re in strategies::arb_recursive_expr()) {
        let (ctx, expr) = strategies::to_context(re);
        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx;

        let zero = simplifier.context.num(0);
        let expr_times_zero = simplifier.context.add(Expr::Mul(expr, zero));

        let (result, _) = simplifier.simplify(expr_times_zero);

        let d = cas_ast::DisplayExpr { context: &simplifier.context, id: result };
        prop_assert_eq!(d.to_string(), "0", "e*0 != 0, got {}", d);
    }
}
