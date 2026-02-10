//! Light rationalization rules for simple surd denominators.

use crate::build::mul2_raw;
use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{count_nodes, Context, DisplayExpr, Expr, ExprId};

// ========== Light Rationalization for Single Numeric Surd Denominators ==========
// Transforms: num / (k * √n) → (num * √n) / (k * n)
// Only applies when:
// - denominator contains exactly one numeric square root
// - base of the root is a positive integer
// - no variables inside the radical

define_rule!(
    RationalizeSingleSurdRule,
    "Rationalize Single Surd",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::as_rational_const;
        use num_rational::BigRational;
        use num_traits::ToPrimitive;

        // Only match Div expressions - use zero-clone helper
        let (num, den) = crate::helpers::as_div(ctx, expr)?;

        // Check denominator for Pow(Number(n), 1/2) patterns
        // We need to find exactly one surd in the denominator factors

        // Helper to check if an expression is a numeric square root
        fn is_numeric_sqrt(ctx: &Context, id: ExprId) -> Option<i64> {
            if let Expr::Pow(base, exp) = ctx.get(id) {
                // Check exponent is 1/2 (using robust detection)
                let exp_val = as_rational_const(ctx, *exp, 8)?;
                let half = BigRational::new(1.into(), 2.into());
                if exp_val != half {
                    return None;
                }
                // Check base is a positive integer
                if let Expr::Number(n) = ctx.get(*base) {
                    if n.is_integer() {
                        return n.numer().to_i64().filter(|&x| x > 0);
                    }
                }
            }
            None
        }

        // Try different denominator patterns
        let (sqrt_n_value, other_den_factors): (i64, Vec<ExprId>) = match ctx.get(den) {
            // Case 1: Denominator is just √n
            Expr::Pow(_, _) => {
                if let Some(n) = is_numeric_sqrt(ctx, den) {
                    (n, vec![])
                } else {
                    return None;
                }
            }

            // Case 2: Denominator is k * √n or √n * k (one level of Mul)
            Expr::Mul(l, r) => {
                let (l, r) = (*l, *r);
                if let Some(n) = is_numeric_sqrt(ctx, l) {
                    // √n * k form
                    (n, vec![r])
                } else if let Some(n) = is_numeric_sqrt(ctx, r) {
                    // k * √n form
                    (n, vec![l])
                } else {
                    // Check if either side is a Mul containing √n (two levels)
                    // For simplicity, we only handle shallow cases
                    return None;
                }
            }

            // Case 3: Function("sqrt", [n])
            Expr::Function(name, ref args)
                if ctx.is_builtin(*name, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                if let Expr::Number(n) = ctx.get(args[0]) {
                    if n.is_integer() {
                        if let Some(n_int) = n.numer().to_i64().filter(|&x| x > 0) {
                            (n_int, vec![])
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None; // Variable inside sqrt
                }
            }

            _ => return None,
        };

        // Build the rationalized form: (num * √n) / (other_den * n)
        let n_expr = ctx.num(sqrt_n_value);
        let half = ctx.rational(1, 2);
        let sqrt_n = ctx.add(Expr::Pow(n_expr, half));

        // New numerator: num * √n
        let new_num = mul2_raw(ctx, num, sqrt_n);

        // New denominator: other_den_factors * n
        let n_in_den = ctx.num(sqrt_n_value);
        let new_den = if other_den_factors.is_empty() {
            n_in_den
        } else {
            let mut den_product = other_den_factors[0];
            for &f in &other_den_factors[1..] {
                den_product = mul2_raw(ctx, den_product, f);
            }
            mul2_raw(ctx, den_product, n_in_den)
        };

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        // Optional: Check node count didn't explode (shouldn't for this simple transform)
        if count_nodes(ctx, new_expr) > count_nodes(ctx, expr) + 10 {
            return None;
        }

        Some(Rewrite::new(new_expr).desc_lazy(|| {
            format!(
                "{} / {} -> {} / {}",
                DisplayExpr {
                    context: ctx,
                    id: num
                },
                DisplayExpr {
                    context: ctx,
                    id: den
                },
                DisplayExpr {
                    context: ctx,
                    id: new_num
                },
                DisplayExpr {
                    context: ctx,
                    id: new_den
                }
            )
        }))
    }
);

// ========== Distribute Numeric Fraction Into Sum ==========
// After canonicalization, (c₁·A + c₂·B) / d  becomes  Mul(Number(1/d), Add(c₁·A, c₂·B)).
// This rule matches that canonicalized form and distributes:
//   Mul(1/d, Add(c₁·A, c₂·B))  →  (c₁/d)·A + (c₂/d)·B
// when all resulting coefficients are integers (or when GCD simplification is possible).
//
// Examples (after canonicalization):
//   1/2 * (2x + 4y)      →  x + 2y
//   1/3 * (6a + 3b)      →  2a + b
//   1/2 * (2·√3 + 2·x²)  →  √3 + x²
//
// Only applies when:
// - outer factor is a non-integer rational Number(p/q) with q > 1
// - inner expression is Add
// - all term coefficients are divisible by q/gcd(all_coeffs*p, q)
// - distributing actually simplifies (doesn't just rearrange)

define_rule!(
    DivScalarIntoAddRule,
    "Distribute Division Into Sum",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        use num_rational::BigRational;
        use num_traits::{One, Zero};

        // Match Mul(Number(frac), Add(...)) or Mul(Add(...), Number(frac))
        let (frac_val, add_id) = match ctx.get(expr) {
            Expr::Mul(l, r) => {
                let (l, r) = (*l, *r);
                match (ctx.get(l), ctx.get(r)) {
                    (Expr::Number(n), Expr::Add(_, _)) => (n.clone(), r),
                    (Expr::Add(_, _), Expr::Number(n)) => (n.clone(), l),
                    _ => return None,
                }
            }
            _ => return None,
        };

        // Only handle non-integer fractions (p/q where q > 1)
        // Integer multipliers (like 2 * (x+y)) don't need this rule
        if frac_val.is_integer() || frac_val.is_zero() {
            return None;
        }

        // Collect all additive terms (flatten)
        fn collect_add_terms(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
            match ctx.get(expr) {
                Expr::Add(l, r) => {
                    collect_add_terms(ctx, *l, terms);
                    collect_add_terms(ctx, *r, terms);
                }
                _ => terms.push(expr),
            }
        }

        let mut terms = Vec::new();
        collect_add_terms(ctx, add_id, &mut terms);

        if terms.len() < 2 {
            return None;
        }

        // Extract the rational coefficient of each term.
        fn extract_rational_coeff(ctx: &Context, term: ExprId, one_id: ExprId) -> (BigRational, ExprId) {
            match ctx.get(term) {
                Expr::Number(n) => {
                    return (n.clone(), one_id);
                }
                Expr::Neg(inner) => {
                    let (c, rest) = extract_rational_coeff(ctx, *inner, one_id);
                    return (-c, rest);
                }
                Expr::Mul(l, r) => {
                    if let Expr::Number(n) = ctx.get(*l) {
                        return (n.clone(), *r);
                    }
                    if let Expr::Number(n) = ctx.get(*r) {
                        return (n.clone(), *l);
                    }
                }
                _ => {}
            }
            (BigRational::from_integer(1.into()), term)
        }

        let one_id = ctx.num(1);

        let mut coeffs: Vec<BigRational> = Vec::with_capacity(terms.len());
        let mut rests: Vec<ExprId> = Vec::with_capacity(terms.len());
        for &t in &terms {
            let (c, r) = extract_rational_coeff(ctx, t, one_id);
            coeffs.push(c);
            rests.push(r);
        }

        // Multiply each coefficient by the outer fraction and check if all results simplify
        let new_coeffs: Vec<BigRational> = coeffs.iter().map(|c| c * &frac_val).collect();

        // Check if at least one coefficient simplified to integer (became simpler)
        // If all new coefficients are non-integer fractions, distributing doesn't help
        let any_simpler = new_coeffs.iter().any(|c| c.is_integer());
        if !any_simpler {
            return None;
        }

        // Build new terms with the distributed coefficients
        let mut new_terms: Vec<ExprId> = Vec::with_capacity(terms.len());
        for (new_c, rest) in new_coeffs.iter().zip(rests.iter()) {
            let new_term = if new_c == &BigRational::from_integer(1.into()) {
                *rest
            } else if new_c == &BigRational::from_integer((-1).into()) {
                ctx.add(Expr::Neg(*rest))
            } else if new_c.is_zero() {
                continue; // Skip zero terms
            } else {
                let c_expr = ctx.add(Expr::Number(new_c.clone()));
                if matches!(ctx.get(*rest), Expr::Number(n) if n.is_one()) {
                    c_expr
                } else {
                    crate::build::mul2_raw(ctx, c_expr, *rest)
                }
            };
            new_terms.push(new_term);
        }

        if new_terms.is_empty() {
            return Some(Rewrite::new(ctx.num(0)).desc("All terms cancel"));
        }

        // Build new sum
        let mut new_sum = new_terms[0];
        for &t in &new_terms[1..] {
            new_sum = ctx.add(Expr::Add(new_sum, t));
        }

        Some(Rewrite::new(new_sum).desc("Divide common factor from sum and denominator"))
    }
);
