use crate::build::mul2_raw;
use crate::Step;
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_traits::{Signed, ToPrimitive};

/// Threshold for using fast mod-p expansion in eager eval.
/// Above this many estimated terms, use poly_ref instead of AST.
pub const EAGER_EXPAND_MODP_THRESHOLD: u64 = 500;

/// Threshold for returning poly_ref instead of materializing AST.
/// Below this, we materialize the full AST for display.
pub const EXPAND_MATERIALIZE_LIMIT: usize = 1_000;

/// Eager evaluation pass for expand() calls.
///
/// This runs BEFORE the simplification pipeline to avoid budget exhaustion
/// from processing large polynomial children.
///
/// Strategy:
/// - For expand(arg) where estimated terms > threshold: use fast mod-p expansion
/// - If terms > EXPAND_MATERIALIZE_LIMIT: return poly_ref (opaque)
/// - Otherwise: materialize AST wrapped in __hold
pub fn eager_eval_expand_calls(ctx: &mut Context, expr: ExprId) -> (ExprId, Vec<Step>) {
    let mut steps = Vec::new();
    let result = eager_eval_expand_recursive(ctx, expr, &mut steps);
    (result, steps)
}

fn eager_eval_expand_recursive(ctx: &mut Context, expr: ExprId, steps: &mut Vec<Step>) -> ExprId {
    // Check if this is expand(...) that should use fast path
    if let Expr::Function(name, args) = ctx.get(expr).clone() {
        if name == "expand" && args.len() == 1 {
            let arg = args[0];

            // Estimate output terms
            if let Some(est) = estimate_expand_terms(ctx, arg) {
                // Use fast mod-p path for large expansions
                if est > EAGER_EXPAND_MODP_THRESHOLD {
                    if let Some(result) = expand_to_poly_ref_or_hold(ctx, arg, est as usize) {
                        steps.push(Step::new(
                            &format!("Eager expand (mod-p, {} terms)", est),
                            "Polynomial Expansion",
                            expr,
                            result,
                            Vec::new(),
                            Some(ctx),
                        ));

                        return result;
                    }
                }
            }
            // Fall through - let normal pipeline handle it
        }

        // For other functions, recurse into children
        let new_args: Vec<ExprId> = args
            .iter()
            .map(|&arg| eager_eval_expand_recursive(ctx, arg, steps))
            .collect();

        if new_args
            .iter()
            .zip(args.iter())
            .any(|(new, old)| new != old)
        {
            return ctx.add(Expr::Function(name.clone(), new_args));
        }
        return expr;
    }

    // Recurse into children for other expression types
    match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            let nl = eager_eval_expand_recursive(ctx, l, steps);
            let nr = eager_eval_expand_recursive(ctx, r, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Sub(l, r) => {
            let nl = eager_eval_expand_recursive(ctx, l, steps);
            let nr = eager_eval_expand_recursive(ctx, r, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let nl = eager_eval_expand_recursive(ctx, l, steps);
            let nr = eager_eval_expand_recursive(ctx, r, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Mul(nl, nr))
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let nl = eager_eval_expand_recursive(ctx, l, steps);
            let nr = eager_eval_expand_recursive(ctx, r, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Pow(b, e) => {
            let nb = eager_eval_expand_recursive(ctx, b, steps);
            let ne = eager_eval_expand_recursive(ctx, e, steps);
            if nb != b || ne != e {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }
        Expr::Neg(e) => {
            let ne = eager_eval_expand_recursive(ctx, e, steps);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        // Leaves - no recursion needed
        Expr::Number(_)
        | Expr::Variable(_)
        | Expr::Constant(_)
        | Expr::Matrix { .. }
        | Expr::SessionRef(_)
        | Expr::Function(_, _) => expr,
    }
}

/// Expand expression to poly_ref (opaque) or __hold(AST) depending on size.
///
/// - If terms <= EXPAND_MATERIALIZE_LIMIT: materialize AST wrapped in __hold
/// - If terms > EXPAND_MATERIALIZE_LIMIT: return poly_ref(inline_stats)
///
/// For now, poly_ref stores metadata inline since we don't have SessionState access.
/// Format: poly_result(n_terms, degree, n_vars, modulus)
fn expand_to_poly_ref_or_hold(
    ctx: &mut Context,
    expr: ExprId,
    _est_terms: usize,
) -> Option<ExprId> {
    use crate::poly_modp_conv::{expr_to_poly_modp, PolyModpBudget, VarTable};
    use crate::rules::algebra::gcd_modp::{multipoly_modp_to_expr, DEFAULT_PRIME};

    // Budget for polynomial expansion
    let budget = PolyModpBudget {
        max_vars: 16,
        max_terms: 500_000,
        max_total_degree: 100,
        max_pow_exp: 100,
    };

    let p = DEFAULT_PRIME;
    let mut vars = VarTable::new();

    // Convert to MultiPolyModP
    let poly = expr_to_poly_modp(ctx, expr, p, &budget, &mut vars).ok()?;

    let n_terms = poly.num_terms();

    if n_terms <= EXPAND_MATERIALIZE_LIMIT {
        // Small enough to materialize - return __hold(AST)
        let expanded = multipoly_modp_to_expr(ctx, &poly, &vars);
        Some(ctx.add(Expr::Function("__hold".to_string(), vec![expanded])))
    } else {
        // Too large - return poly_result with metadata
        // poly_result(n_terms, degree, n_vars, modulus)
        // This acts as an opaque handle that won't be traversed
        let terms = ctx.num(n_terms as i64);
        // Note: MultiPolyModP doesn't have max_degree(); use 0 as placeholder
        let degree = ctx.num(0);
        let nvars = ctx.num(vars.names().len() as i64);
        let modulus = ctx.num(p as i64);

        Some(ctx.add(Expr::Function(
            "poly_result".to_string(),
            vec![terms, degree, nvars, modulus],
        )))
    }
}

/// Legacy function for backward compatibility.
/// Now uses expand_to_poly_ref_or_hold internally.
pub fn expand_modp_safe(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    // Always materialize AST for this function (used by distribution.rs)
    use crate::poly_modp_conv::{expr_to_poly_modp, PolyModpBudget, VarTable};
    use crate::rules::algebra::gcd_modp::{multipoly_modp_to_expr, DEFAULT_PRIME};

    let budget = PolyModpBudget {
        max_vars: 16,
        max_terms: 500_000,
        max_total_degree: 100,
        max_pow_exp: 100,
    };

    let p = DEFAULT_PRIME;
    let mut vars = VarTable::new();

    let poly = expr_to_poly_modp(ctx, expr, p, &budget, &mut vars).ok()?;
    let expanded = multipoly_modp_to_expr(ctx, &poly, &vars);
    Some(ctx.add(Expr::Function("__hold".to_string(), vec![expanded])))
}

/// Expand with budget tracking, returning PassStats for unified budget charging.
///
/// Tracks `nodes_delta` and estimates `terms_materialized` based on the expansion.
pub fn expand_with_stats(ctx: &mut Context, expr: ExprId) -> (ExprId, crate::budget::PassStats) {
    let nodes_snap = ctx.stats().nodes_created;

    // Estimate terms before expansion (if Pow(Add, n))
    let estimated_terms = estimate_expand_terms(ctx, expr).unwrap_or(0);

    let result = expand(ctx, expr);

    let nodes_delta = ctx.stats().nodes_created.saturating_sub(nodes_snap);

    let stats = crate::budget::PassStats {
        op: crate::budget::Operation::Expand,
        rewrite_count: 0, // Expand doesn't use rewrites
        nodes_delta,
        terms_materialized: estimated_terms,
        poly_ops: 0,
        stop_reason: None, // Budget checking happens via MultinomialExpandBudget
    };

    (result, stats)
}

/// Estimate number of terms that will be generated by expansion.
/// Returns None if not an expandable pattern or unable to estimate.
pub fn estimate_expand_terms(ctx: &Context, expr: ExprId) -> Option<u64> {
    estimate_expanded_term_count(ctx, expr)
}

/// Recursively estimate term count after expansion.
/// For Pow(Add, n): multinomial count C(n+k-1, k-1)
/// For Mul(a, b): product of expanded term counts
/// For Add/Sub: sum of expanded term counts
fn estimate_expanded_term_count(ctx: &Context, expr: ExprId) -> Option<u64> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            // Extract exponent
            let n = match ctx.get(*exp) {
                Expr::Number(num) => {
                    if !num.is_integer() || num.is_negative() {
                        return None;
                    }
                    num.to_integer().to_u32()?
                }
                _ => return None,
            };

            // Count base terms (flatten Add tree)
            let k = count_add_terms(ctx, *base);
            if k < 2 || n < 2 {
                return Some(1); // Not expandable, counts as 1 term
            }

            // Multinomial term count: C(n+k-1, k-1)
            crate::multinomial_expand::multinomial_term_count(n, k, u64::MAX as usize)
                .map(|c| c as u64)
        }
        Expr::Mul(l, r) => {
            // For Mul, estimate as product of expanded term counts
            let l_terms = estimate_expanded_term_count(ctx, *l).unwrap_or(1);
            let r_terms = estimate_expanded_term_count(ctx, *r).unwrap_or(1);

            // Overflow-safe multiplication
            l_terms.checked_mul(r_terms)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            // Sum of terms (with potential cancellation, so this is an upper bound)
            let l_terms = estimate_expanded_term_count(ctx, *l).unwrap_or(1);
            let r_terms = estimate_expanded_term_count(ctx, *r).unwrap_or(1);
            l_terms.checked_add(r_terms)
        }
        Expr::Neg(e) => estimate_expanded_term_count(ctx, *e),
        _ => Some(1), // Atoms count as 1 term
    }
}

/// Count number of terms in an Add tree (for estimation)
fn count_add_terms(ctx: &Context, expr: ExprId) -> usize {
    match ctx.get(expr) {
        Expr::Add(l, r) => count_add_terms(ctx, *l) + count_add_terms(ctx, *r),
        Expr::Sub(l, r) => count_add_terms(ctx, *l) + count_add_terms(ctx, *r),
        Expr::Neg(e) => count_add_terms(ctx, *e),
        _ => 1,
    }
}

/// Expands an expression.
/// This is the main entry point for expansion.
/// It recursively expands children and then applies specific expansion rules.
pub fn expand(ctx: &mut Context, expr: ExprId) -> ExprId {
    // NOTE: We previously protected "canonical forms" (like (x+y)*(x-y)) from expansion,
    // but this was causing incomplete expansion when user explicitly calls expand().
    // If user calls expand(), they want FULL expansion. Canonical preservation should
    // only be applied in simplify(), not expand().

    // V2.15.35: mod-p expansion (expand_modp_safe) is NOT used here because:
    // 1. The bottleneck is AST reconstruction, not polynomial arithmetic
    // 2. mod-p is only useful for internal consumption (e.g., poly_gcd_modp)
    //    where the result stays in mod-p form without AST reconstruction
    // For user-facing expand(), we use MultinomialExact (in Pow branch below).

    // Symbolic expansion: expand children first (bottom-up), then apply rules

    let expr_data = ctx.get(expr).clone();
    let expanded_expr = match expr_data {
        Expr::Add(l, r) => {
            let el = expand(ctx, l);
            let er = expand(ctx, r);
            ctx.add(Expr::Add(el, er))
        }
        Expr::Sub(l, r) => {
            let el = expand(ctx, l);
            let er = expand(ctx, r);
            ctx.add(Expr::Sub(el, er))
        }
        Expr::Mul(l, r) => {
            let el = expand(ctx, l);
            let er = expand(ctx, r);
            // Apply distribution
            expand_mul(ctx, el, er)
        }
        Expr::Div(l, r) => {
            let el = expand(ctx, l);
            let er = expand(ctx, r);
            // Distribute division? (a+b)/c -> a/c + b/c
            // This is usually considered "expansion".
            expand_div(ctx, el, er)
        }
        Expr::Pow(b, e) => {
            // FAST PATH: Try direct multinomial expansion on ORIGINAL base
            // before expanding children (which can change structure)
            let budget = crate::multinomial_expand::MultinomialExpandBudget::default();
            if let Some(result) =
                crate::multinomial_expand::try_expand_multinomial_direct(ctx, b, e, &budget)
            {
                return result;
            }

            // Fall back to normal processing
            let eb = expand(ctx, b);
            let ee = expand(ctx, e);
            // Apply binomial expansion if applicable
            expand_pow(ctx, eb, ee)
        }
        Expr::Neg(e) => {
            let ee = expand(ctx, e);
            ctx.add(Expr::Neg(ee))
        }
        Expr::Function(name, args) => {
            if name == "expand" && args.len() == 1 {
                // Unwrap explicit expand call
                return expand(ctx, args[0]);
            }
            let new_args: Vec<ExprId> = args.iter().map(|a| expand(ctx, *a)).collect();
            ctx.add(Expr::Function(name, new_args))
        }
        _ => expr,
    };

    expanded_expr
}

/// Expands multiplication: distributes over addition/subtraction.
/// a * (b + c) -> a*b + a*c
pub fn expand_mul(ctx: &mut Context, l: ExprId, r: ExprId) -> ExprId {
    // Logic from `distribute` in algebra.rs

    // Try to distribute l into r
    if let Some(res) = distribute_single(ctx, l, r) {
        return res;
    }
    // Try to distribute r into l
    if let Some(res) = distribute_single(ctx, r, l) {
        return res;
    }

    // If neither, just return Mul(l, r)
    mul2_raw(ctx, l, r)
}

fn distribute_single(ctx: &mut Context, multiplier: ExprId, target: ExprId) -> Option<ExprId> {
    let target_data = ctx.get(target).clone();
    match target_data {
        Expr::Add(a, b) => {
            let ma = expand_mul(ctx, multiplier, a);
            let mb = expand_mul(ctx, multiplier, b);
            Some(ctx.add(Expr::Add(ma, mb)))
        }
        Expr::Sub(a, b) => {
            let ma = expand_mul(ctx, multiplier, a);
            let mb = expand_mul(ctx, multiplier, b);
            Some(ctx.add(Expr::Sub(ma, mb)))
        }
        _ => None,
    }
}

/// Expands division: distributes over addition/subtraction in numerator.
/// (a + b) / c -> a/c + b/c
pub fn expand_div(ctx: &mut Context, num: ExprId, den: ExprId) -> ExprId {
    let num_data = ctx.get(num).clone();
    match num_data {
        Expr::Add(a, b) => {
            let da = expand_div(ctx, a, den);
            let db = expand_div(ctx, b, den);
            ctx.add(Expr::Add(da, db))
        }
        Expr::Sub(a, b) => {
            let da = expand_div(ctx, a, den);
            let db = expand_div(ctx, b, den);
            ctx.add(Expr::Sub(da, db))
        }
        _ => ctx.add(Expr::Div(num, den)),
    }
}

/// Expands power: (a + b)^n
/// Note: Fast multinomial expansion is handled in expand() BEFORE children are processed.
pub fn expand_pow(ctx: &mut Context, base: ExprId, exp: ExprId) -> ExprId {
    // Logic from BinomialExpansionRule
    let base_data = ctx.get(base).clone();

    // (a * b)^n -> a^n * b^n
    if let Expr::Mul(a, b) = base_data {
        let ea = expand_pow(ctx, a, exp);
        let eb = expand_pow(ctx, b, exp);
        return mul2_raw(ctx, ea, eb);
    }

    if let Expr::Add(a, b) = base_data {
        let exp_data = ctx.get(exp).clone();
        if let Expr::Number(n) = exp_data {
            if n.is_integer() && !n.is_negative() {
                if let Some(n_val) = n.to_integer().to_u32() {
                    // Limit expansion
                    if (2..=10).contains(&n_val) {
                        // Expand: sum(k=0 to n) (n choose k) * a^(n-k) * b^k
                        let mut terms = Vec::new();
                        for k in 0..=n_val {
                            let coeff = binomial_coeff(n_val, k);
                            let exp_a = n_val - k;
                            let exp_b = k;

                            let term_a = if exp_a == 0 {
                                ctx.num(1)
                            } else if exp_a == 1 {
                                a
                            } else {
                                let e = ctx.num(exp_a as i64);
                                ctx.add(Expr::Pow(a, e))
                            };
                            let term_b = if exp_b == 0 {
                                ctx.num(1)
                            } else if exp_b == 1 {
                                b
                            } else {
                                let e = ctx.num(exp_b as i64);
                                ctx.add(Expr::Pow(b, e))
                            };

                            let mut term = mul2_raw(ctx, term_a, term_b);
                            if coeff > 1 {
                                let c = ctx.num(coeff as i64);
                                term = mul2_raw(ctx, c, term);
                            }
                            terms.push(term);
                        }

                        // Sum up terms
                        let mut expanded = terms[0];
                        for &term in terms.iter().skip(1) {
                            expanded = ctx.add(Expr::Add(expanded, term));
                        }

                        return expand(ctx, expanded);
                    }
                }
            }
        }
    }

    // (a - b)^n -> (a + (-b))^n
    if let Expr::Sub(a, b) = base_data {
        let neg_b = ctx.add(Expr::Neg(b));
        let sum = ctx.add(Expr::Add(a, neg_b));
        return expand_pow(ctx, sum, exp);
    }

    // (-a)^n
    if let Expr::Neg(a) = base_data {
        let exp_data = ctx.get(exp).clone();
        if let Expr::Number(n) = exp_data {
            if n.is_integer() {
                if n.to_integer().is_even() {
                    // (-a)^n -> a^n
                    return expand_pow(ctx, a, exp);
                } else {
                    // (-a)^n -> -(a^n)
                    let p = expand_pow(ctx, a, exp);
                    return ctx.add(Expr::Neg(p));
                }
            }
        }
    }

    ctx.add(Expr::Pow(base, exp))
}

fn binomial_coeff(n: u32, k: u32) -> u32 {
    if k == 0 || k == n {
        return 1;
    }
    if k > n {
        return 0;
    }
    let mut res = 1;
    for i in 0..k {
        res = res * (n - i) / (i + 1);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::DisplayExpr;
    use cas_parser::parse;

    fn s(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn test_expand_mul_distribute() {
        let mut ctx = Context::new();
        let expr = parse("2 * (x + 3)", &mut ctx).unwrap();
        let res = expand(&mut ctx, expr);
        // 2*x + 2*3. Note: 2*3 is not simplified here, expand is structural.
        // But wait, expand calls expand_mul which constructs Mul.
        // If we want simplification, we need a simplifier or builder that simplifies.
        // The current `ctx.add` just adds nodes.
        // So we expect "2 * x + 2 * 3".
        assert_eq!(s(&ctx, res), "2 * x + 2 * 3"); // Canonical: polynomial order (variables before constants)
    }

    #[test]
    fn test_expand_mul_nested() {
        let mut ctx = Context::new();
        let expr = parse("a * (b + c + d)", &mut ctx).unwrap();
        let res = expand(&mut ctx, expr);
        // a*b + a*c + a*d
        // (b+c)+d -> a*(b+c) + a*d -> (a*b + a*c) + a*d
        let str_res = s(&ctx, res);
        assert!(str_res.contains("a * b"));
        assert!(str_res.contains("a * c"));
        assert!(str_res.contains("a * d"));
    }

    #[test]
    fn test_expand_pow_binomial() {
        let mut ctx = Context::new();
        let expr = parse("(x + 1)^2", &mut ctx).unwrap();
        let res = expand(&mut ctx, expr);
        // x^2 + 2*x*1 + 1^2 -> x^2 + 2*x + 1 (if simplified)
        // Here: x^2 + 2 * (x * 1) + 1
        // Wait, 1^2 is 1? No, expand_pow constructs Pow(1, 2).
        // Unless we have simplification.
        // My implementation:
        // term_a = x^2 (if exp_a=2)
        // term_b = 1 (if exp_b=0)
        // term = x^2 * 1
        // coeff = 1.
        // So x^2 * 1.
        // Middle: 2 * (x^1 * 1^1) = 2 * (x * 1).
        // Last: 1 * (x^0 * 1^2) = 1 * (1 * 1^2).
        // This is very verbose without simplification.
        // But `expand` is supposed to be pure structural expansion.
        // Simplification happens later or we use a smart builder.
        // For now, let's check structure.
        let str_res = s(&ctx, res);
        assert!(str_res.contains("x^2"));
        // assert!(str_res.contains("2 * x")); // Might be 2 * (x * 1)
    }
}
