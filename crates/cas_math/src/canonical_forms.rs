use cas_ast::{Context, Expr, ExprId};

/// Detects if an expression is in a canonical (elegant) form that should not be expanded.
/// These forms are mathematically clean and expanding them would only create unnecessary complexity.
pub fn is_canonical_form(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        // Case 1: (product)^n where product is factored elegantly
        Expr::Pow(base, exp) => {
            is_product_of_factors(ctx, *base) && is_small_positive_integer(ctx, *exp)
        }

        // Case 2: Product of conjugates without power (e.g., (x+y)*(x-y))
        // This is already in difference of squares form, expanding serves no purpose
        Expr::Mul(l, r) => is_conjugate(ctx, *l, *r),

        // Case 3: Functions containing powers or products that should be preserved
        // Examples: sqrt((x-1)^2), sqrt((x-2)(x+2)), abs((x+y)^3)
        Expr::Function(fn_id, args)
            if (ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt)
                || ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Abs))
                && args.len() == 1 =>
        {
            let inner = args[0];
            match ctx.get(inner) {
                // Protect sqrt(x^2), sqrt((x-1)^2), etc.
                Expr::Pow(base, exp) => {
                    // Any power of 2, or product raised to a power
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into())
                        {
                            return true; // sqrt(anything^2) should use SimplifySqrtSquareRule
                        }
                    }
                    // Also protect if base is a product
                    is_product_of_factors(ctx, *base) && is_small_positive_integer(ctx, *exp)
                }
                // Protect sqrt((x-2)(x+2)), abs((x+1)(x-1)), etc.
                Expr::Mul(l, r) => is_conjugate(ctx, *l, *r) || is_product_of_factors(ctx, inner),
                _ => false,
            }
        }

        _ => false,
    }
}

/// Checks if base is already a product in elegant factored form
fn is_product_of_factors(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        // (a + b) * (a - b) - difference of squares form
        Expr::Mul(l, r) => {
            // Check if this is a conjugate pair (difference of squares pattern)
            if is_conjugate(ctx, *l, *r) {
                return true;
            }

            // Only protect if at least one factor is a binomial (Add/Sub)
            // This prevents protecting simple monomials like (3*x)^3
            // which should expand to 27*x^3
            let l_is_binomial = matches!(ctx.get(*l), Expr::Add(_, _) | Expr::Sub(_, _));
            let r_is_binomial = matches!(ctx.get(*r), Expr::Add(_, _) | Expr::Sub(_, _));

            if (l_is_binomial || r_is_binomial)
                && is_linear_or_simple(ctx, *l)
                && is_linear_or_simple(ctx, *r)
            {
                return true;
            }

            // Recursive: check if it's a product of multiple factors
            is_product_of_factors(ctx, *l) || is_product_of_factors(ctx, *r)
        }
        _ => false,
    }
}

/// Check if expression is linear (degree 1) or simple (constant, variable)
fn is_linear_or_simple(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => true,
        Expr::Add(a, b) | Expr::Sub(a, b) => {
            // x + 1, x - 2, etc.
            matches!(
                (ctx.get(*a), ctx.get(*b)),
                (Expr::Variable(_), Expr::Number(_))
                    | (Expr::Number(_), Expr::Variable(_))
                    | (Expr::Neg(_), Expr::Variable(_))
                    | (Expr::Variable(_), Expr::Neg(_))
            )
        }
        Expr::Neg(inner) => is_linear_or_simple(ctx, *inner),
        _ => false,
    }
}

/// Check if exponent is a small positive integer (2, 3, etc.)
fn is_small_positive_integer(ctx: &Context, exp: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(exp) {
        if n.is_integer() && *n > num_rational::BigRational::from_integer(1.into()) {
            return true;
        }
    }
    false
}

/// Checks for conjugate pairs: (A+B) and (A-B)
/// Order-invariant: handles (x+1)(x-1), (x-1)(x+1), (-1+x)(1+x), etc.
fn is_conjugate(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    crate::expr_relations::is_conjugate_binomial(ctx, a, b)
}

// ============================================================================
// normalize_core: Canonical normalization to prevent infinite loops
// ============================================================================

// NOTE: N2 (flatten/sort/compress products) is not yet implemented because the
// full implementation caused stack overflow in some tests. The helpers below are
// preserved for future use when we implement a more careful version.
//
// TODO: Implement N2 with proper termination checks:
// - Track "already normalized" state to avoid re-processing
// - Use explicit worklist instead of recursion
// - Only compress when it actually reduces complexity

/// PERF: Quick check whether an expression *might* contain patterns that
/// `normalize_core` actually transforms (Neg(Neg(…)), Neg(Number(…)), or
/// Pow(Pow(…,int),int)). If none are present, we can skip the entire
/// HashMap+worklist traversal.
///
/// This is conservative — it scans the top few levels only (depth ≤ budget)
/// to avoid becoming expensive itself. If the budget is exhausted we return
/// `true` ("might need normalization") to stay correct.
fn needs_normalization(ctx: &Context, expr: ExprId, budget: &mut u32) -> bool {
    if *budget == 0 {
        return true; // budget exhausted → conservative
    }
    *budget -= 1;
    match ctx.get(expr) {
        // Atoms never need normalization
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,

        // Neg: normalizable if inner is Neg or Number
        Expr::Neg(inner) => {
            matches!(ctx.get(*inner), Expr::Neg(_) | Expr::Number(_))
                || needs_normalization(ctx, *inner, budget)
        }

        // Pow: normalizable if base is Pow(_, int) and exp is int
        Expr::Pow(base, exp) => {
            if let Expr::Pow(_, inner_exp) = ctx.get(*base) {
                if let Expr::Number(a) = ctx.get(*inner_exp) {
                    if a.is_integer() {
                        if let Expr::Number(b) = ctx.get(*exp) {
                            if b.is_integer() {
                                return true; // Pow(Pow(x,int),int)
                            }
                        }
                    }
                }
            }
            needs_normalization(ctx, *base, budget) || needs_normalization(ctx, *exp, budget)
        }

        // Binary ops: recurse into children
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            needs_normalization(ctx, *l, budget) || needs_normalization(ctx, *r, budget)
        }

        Expr::Function(_, args) => args.iter().any(|a| needs_normalization(ctx, *a, budget)),
        Expr::Matrix { data, .. } => data.iter().any(|e| needs_normalization(ctx, *e, budget)),
        Expr::Hold(inner) => needs_normalization(ctx, *inner, budget),
    }
}

/// Normalize an expression to canonical form to prevent infinite loops.
///
/// Applies the following normalization rules:
/// - **N1**: Sign absorption - `Neg(Neg(x))` → `x`
/// - **N2**: Flatten products, sort factors, compress exponents (`x * x` → `x^2`)
/// - **N3**: Compress nested powers - `Pow(Pow(x, a), b)` → `Pow(x, a*b)` if a,b are integers
///
/// This should be called after each successful rewrite to ensure expressions
/// stay in a consistent form that rules can match reliably.
///
/// IMPLEMENTATION: Uses iterative worklist to avoid stack overflow on deep expressions.
pub fn normalize_core(ctx: &mut Context, expr: ExprId) -> ExprId {
    // PERF: Early exit if nothing to normalize (atoms, simple expressions)
    let mut budget = 256;
    if !needs_normalization(ctx, expr, &mut budget) {
        return expr;
    }
    let mut cache = std::collections::HashMap::new();
    normalize_core_inner(ctx, expr, &mut cache)
}

/// Like `normalize_core`, but reuses a caller-provided cache HashMap.
/// PERF: Avoids per-call HashMap allocation when called in a tight loop.
pub fn normalize_core_with_cache(
    ctx: &mut Context,
    expr: ExprId,
    cache: &mut std::collections::HashMap<ExprId, ExprId>,
) -> ExprId {
    // PERF: Early exit if nothing to normalize
    let mut budget = 256;
    if !needs_normalization(ctx, expr, &mut budget) {
        return expr;
    }
    cache.clear();
    normalize_core_inner(ctx, expr, cache)
}

/// Inner implementation shared by both `normalize_core` and `normalize_core_with_cache`.
fn normalize_core_inner(
    ctx: &mut Context,
    expr: ExprId,
    cache: &mut std::collections::HashMap<ExprId, ExprId>,
) -> ExprId {
    use num_traits::ToPrimitive;

    // Worklist: (node, children_processed)
    // When children_processed=false, push children first
    // When children_processed=true, normalize the node itself
    let mut stack: Vec<(ExprId, bool)> = vec![(expr, false)];

    while let Some((id, children_done)) = stack.pop() {
        // If already cached, skip
        if cache.contains_key(&id) {
            continue;
        }

        let node = ctx.get(id).clone();

        if !children_done {
            // First visit: push self back with children_done=true, then push children
            stack.push((id, true));

            match &node {
                Expr::Neg(inner) => stack.push((*inner, false)),
                Expr::Pow(base, exp) => {
                    stack.push((*exp, false));
                    stack.push((*base, false));
                }
                Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Add(l, r) | Expr::Sub(l, r) => {
                    stack.push((*r, false));
                    stack.push((*l, false));
                }
                Expr::Function(_, args) => {
                    for arg in args.iter().rev() {
                        stack.push((*arg, false));
                    }
                }
                Expr::Matrix { data, .. } => {
                    for elem in data.iter().rev() {
                        stack.push((*elem, false));
                    }
                }
                // Atoms: no children
                _ => {}
            }
        } else {
            // Second visit: children are cached, now normalize this node
            let result = match &node {
                // N0: Neg(Number(n)) → Number(-n)
                // N1: Neg(Neg(x)) → x
                Expr::Neg(inner) => {
                    let inner_norm = *cache.get(inner).unwrap_or(inner);

                    // N0: Neg(Number(n)) → Number(-n)
                    if let Expr::Number(n) = ctx.get(inner_norm) {
                        ctx.add(Expr::Number(-n.clone()))
                    }
                    // N1: Neg(Neg(x)) → x
                    else if let Expr::Neg(double_inner) = ctx.get(inner_norm).clone() {
                        // Neg(Neg(x)) → x (recursively normalizes double_inner)
                        // We need to look up double_inner in cache
                        *cache.get(&double_inner).unwrap_or(&double_inner)
                    } else if inner_norm == *inner {
                        id
                    } else {
                        ctx.add(Expr::Neg(inner_norm))
                    }
                }

                // N3: Pow(Pow(x, a), b) → Pow(x, a*b) if both are integers
                Expr::Pow(base, exp) => {
                    let base_norm = *cache.get(base).unwrap_or(base);
                    let exp_norm = *cache.get(exp).unwrap_or(exp);

                    // Check for Pow(Pow(x, a), b)
                    let result =
                        if let Expr::Pow(inner_base, inner_exp) = ctx.get(base_norm).clone() {
                            if let (Expr::Number(a), Expr::Number(b)) =
                                (ctx.get(inner_exp), ctx.get(exp_norm))
                            {
                                if a.is_integer() && b.is_integer() {
                                    if let (Some(a_i), Some(b_i)) =
                                        (a.to_integer().to_i64(), b.to_integer().to_i64())
                                    {
                                        let product = a_i * b_i;
                                        let new_exp = ctx.num(product);
                                        Some(ctx.add(Expr::Pow(inner_base, new_exp)))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                    result.unwrap_or_else(|| {
                        if base_norm == *base && exp_norm == *exp {
                            id
                        } else {
                            ctx.add(Expr::Pow(base_norm, exp_norm))
                        }
                    })
                }

                // N2: Mul normalization - currently just recurse
                Expr::Mul(l, r) => {
                    let l_norm = *cache.get(l).unwrap_or(l);
                    let r_norm = *cache.get(r).unwrap_or(r);
                    if l_norm == *l && r_norm == *r {
                        id
                    } else {
                        ctx.add_raw(Expr::Mul(l_norm, r_norm))
                    }
                }

                Expr::Div(n, d) => {
                    let n_norm = *cache.get(n).unwrap_or(n);
                    let d_norm = *cache.get(d).unwrap_or(d);
                    if n_norm == *n && d_norm == *d {
                        id
                    } else {
                        ctx.add(Expr::Div(n_norm, d_norm))
                    }
                }

                Expr::Add(l, r) => {
                    let l_norm = *cache.get(l).unwrap_or(l);
                    let r_norm = *cache.get(r).unwrap_or(r);
                    // Use add_raw to preserve term order (e.g., polynomial descending degree)
                    if l_norm == *l && r_norm == *r {
                        id
                    } else {
                        ctx.add_raw(Expr::Add(l_norm, r_norm))
                    }
                }

                Expr::Sub(l, r) => {
                    let l_norm = *cache.get(l).unwrap_or(l);
                    let r_norm = *cache.get(r).unwrap_or(r);
                    if l_norm == *l && r_norm == *r {
                        id
                    } else {
                        ctx.add(Expr::Sub(l_norm, r_norm))
                    }
                }

                Expr::Function(name, args) => {
                    let args_norm: Vec<_> =
                        args.iter().map(|a| *cache.get(a).unwrap_or(a)).collect();
                    if args_norm == *args {
                        id
                    } else {
                        ctx.add(Expr::Function(*name, args_norm))
                    }
                }

                Expr::Matrix { data, rows, cols } => {
                    let data_norm: Vec<_> =
                        data.iter().map(|e| *cache.get(e).unwrap_or(e)).collect();
                    if data_norm == *data {
                        id
                    } else {
                        ctx.add(Expr::Matrix {
                            data: data_norm,
                            rows: *rows,
                            cols: *cols,
                        })
                    }
                }

                // Atoms: no normalization needed
                Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => id,

                // Hold: normalize inner but keep hold wrapper
                Expr::Hold(inner) => {
                    let inner_norm = *cache.get(inner).unwrap_or(inner);
                    if inner_norm == *inner {
                        id
                    } else {
                        ctx.add(Expr::Hold(inner_norm))
                    }
                }
            };

            cache.insert(id, result);
        }
    }

    // Return the normalized result for the root expression
    *cache.get(&expr).unwrap_or(&expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn test_canonical_difference_of_squares_squared() {
        let mut ctx = Context::new();
        // ((x+1)*(x-1))^2
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));
        let x_minus_1 = ctx.add(Expr::Sub(x, one));
        let product = ctx.add(Expr::Mul(x_plus_1, x_minus_1));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(product, two));

        assert!(
            is_canonical_form(&ctx, expr),
            "((x+1)*(x-1))^2 should be canonical"
        );
    }

    #[test]
    fn test_simple_binomial_not_canonical() {
        let mut ctx = Context::new();
        // (x+1)^2 - should expand for educational purposes
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(x_plus_1, two));

        assert!(
            !is_canonical_form(&ctx, expr),
            "(x+1)^2 should NOT be canonical (should expand)"
        );
    }

    #[test]
    fn test_x_squared_minus_one_squared() {
        let mut ctx = Context::new();
        // (x^2-1)^2 can be expanded
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let x_sq = ctx.add(Expr::Pow(x, two));
        let x_sq_minus_1 = ctx.add(Expr::Sub(x_sq, one));
        let expr = ctx.add(Expr::Pow(x_sq_minus_1, two));

        // This is NOT a product, so not canonical
        assert!(!is_canonical_form(&ctx, expr));
    }

    #[test]
    fn test_canonicalized_conjugate_add_add() {
        let mut ctx = Context::new();
        // (-1 + x) * (1 + x) should be detected as conjugates
        let x = ctx.var("x");
        let one = ctx.num(1);
        let neg_one = ctx.num(-1);

        // (-1 + x) * (1 + x)
        let left = ctx.add(Expr::Add(neg_one, x));
        let right = ctx.add(Expr::Add(one, x));
        let product = ctx.add(Expr::Mul(left, right));

        assert!(
            is_canonical_form(&ctx, product),
            "(-1+x)*(1+x) should be canonical"
        );
    }

    #[test]
    fn test_normalize_core_neg_neg() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        // Neg(Neg(x)) → x
        let neg_x = ctx.add(Expr::Neg(x));
        let neg_neg_x = ctx.add(Expr::Neg(neg_x));

        let normalized = normalize_core(&mut ctx, neg_neg_x);
        assert_eq!(normalized, x, "Neg(Neg(x)) should normalize to x");
    }

    #[test]
    fn test_normalize_core_pow_pow() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);

        // Pow(Pow(x, 2), 3) → Pow(x, 6)
        let x_sq = ctx.add(Expr::Pow(x, two));
        let x_sq_cubed = ctx.add(Expr::Pow(x_sq, three));

        let normalized = normalize_core(&mut ctx, x_sq_cubed);

        // Should be Pow(x, 6)
        if let Expr::Pow(base, exp) = ctx.get(normalized) {
            assert_eq!(*base, x, "Base should be x");
            if let Expr::Number(n) = ctx.get(*exp) {
                assert_eq!(
                    *n,
                    num_rational::BigRational::from_integer(6.into()),
                    "Exponent should be 6"
                );
            } else {
                panic!("Expected Number exponent");
            }
        } else {
            panic!("Expected Pow");
        }
    }
}
