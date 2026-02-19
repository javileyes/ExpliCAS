use crate::build::mul2_raw;
use crate::trig_roots_flatten::flatten_add_sub_chain;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};
use smallvec::SmallVec;

/// A group of terms that cancelled to zero (e.g., 5 + (-5) -> 0).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CancelledGroup {
    /// The canonical key (term part) that identified this group.
    pub key: ExprId,
    /// Original terms from the input expression that cancelled.
    pub original_terms: SmallVec<[ExprId; 4]>,
    /// Whether this is a pure constant cancellation (prioritized for focus).
    pub is_constant: bool,
}

/// A group of terms that were combined (e.g., x + x -> 2x).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CombinedGroup {
    /// The canonical key (term part) that identifies this group.
    pub key: ExprId,
    /// Original terms from the input expression.
    pub original_terms: SmallVec<[ExprId; 4]>,
    /// The combined result term.
    pub combined_term: ExprId,
}

/// Internal result for collect core.
#[derive(Debug, Clone)]
pub struct CollectCoreResult {
    pub new_expr: ExprId,
    pub cancelled: Vec<CancelledGroup>,
    pub combined: Vec<CombinedGroup>,
}

/// Core collect-like-terms implementation without domain semantics.
pub fn collect_impl(ctx: &mut Context, expr: ExprId) -> CollectCoreResult {
    // CRITICAL: Do NOT collect non-commutative expressions (e.g., matrices).
    if !ctx.is_mul_commutative(expr) {
        return CollectCoreResult {
            new_expr: expr,
            cancelled: vec![],
            combined: vec![],
        };
    }

    // Flatten terms.
    let terms = flatten_add_sub_chain(ctx, expr);

    // Group terms by non-coefficient part, tracking original terms.
    let mut groups: Vec<(BigRational, ExprId, SmallVec<[ExprId; 4]>)> = Vec::new();

    for term in terms {
        let (coeff, term_part) = extract_numerical_coeff(ctx, term);

        let mut found = false;
        for (g_coeff, g_term, g_originals) in groups.iter_mut() {
            if are_structurally_equal(ctx, *g_term, term_part) {
                *g_coeff += coeff.clone();
                g_originals.push(term);
                found = true;
                break;
            }
        }

        if !found {
            let mut originals = SmallVec::new();
            originals.push(term);
            groups.push((coeff, term_part, originals));
        }
    }

    // Sort groups to ensure canonical order.
    groups.sort_by(|a, b| compare_expr(ctx, a.1, b.1));

    let mut cancelled = Vec::new();
    let mut combined = Vec::new();
    let mut new_terms = Vec::new();

    for (coeff, term_part, originals) in groups {
        if coeff.is_zero() {
            let is_constant = is_one_term(ctx, term_part);
            cancelled.push(CancelledGroup {
                key: term_part,
                original_terms: originals,
                is_constant,
            });
            continue;
        }

        let term = if is_one_term(ctx, term_part) {
            ctx.add(Expr::Number(coeff.clone()))
        } else if coeff.is_one() {
            term_part
        } else if coeff == BigRational::from_integer((-1).into()) {
            ctx.add(Expr::Neg(term_part))
        } else {
            let coeff_expr = ctx.add(Expr::Number(coeff.clone()));
            mul2_raw(ctx, coeff_expr, term_part)
        };

        if originals.len() > 1 {
            combined.push(CombinedGroup {
                key: term_part,
                original_terms: originals,
                combined_term: term,
            });
        }

        new_terms.push(term);
    }

    // Sort terms by global canonical order to match engine canonicalization.
    new_terms.sort_by(|a, b| compare_expr(ctx, *a, *b));

    let new_expr = if new_terms.is_empty() {
        ctx.num(0)
    } else {
        // Right-associative Add chain.
        let mut result = match new_terms.last() {
            Some(r) => *r,
            None => ctx.num(0),
        };
        for t in new_terms.iter().rev().skip(1) {
            result = ctx.add(Expr::Add(*t, result));
        }
        result
    };

    CollectCoreResult {
        new_expr,
        cancelled,
        combined,
    }
}

fn extract_numerical_coeff(ctx: &mut Context, expr: ExprId) -> (BigRational, ExprId) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Number(n) => (n, ctx.num(1)),
        Expr::Neg(e) => {
            let (c, t) = extract_numerical_coeff(ctx, e);
            (-c, t)
        }
        Expr::Mul(_, _) => {
            // Flatten nested Mul chains to extract all numeric factors.
            let mut factors = Vec::new();
            let mut stack = vec![expr];
            while let Some(id) = stack.pop() {
                if let Expr::Mul(a, b) = ctx.get(id) {
                    stack.push(*a);
                    stack.push(*b);
                } else {
                    factors.push(id);
                }
            }

            let mut coeff = BigRational::one();
            let mut non_numeric: Vec<ExprId> = Vec::new();
            for factor in factors {
                if let Expr::Number(n) = ctx.get(factor) {
                    coeff *= n.clone();
                } else {
                    non_numeric.push(factor);
                }
            }

            if non_numeric.is_empty() {
                (coeff, ctx.num(1))
            } else if coeff.is_one() {
                (BigRational::one(), expr)
            } else {
                // LIFO traversal can reverse order; sort canonically for idempotence.
                non_numeric.sort_by(|a, b| compare_expr(ctx, *a, *b));
                let mut core = non_numeric[0];
                for &factor in &non_numeric[1..] {
                    core = mul2_raw(ctx, core, factor);
                }
                (coeff, core)
            }
        }
        Expr::Add(_, _)
        | Expr::Sub(_, _)
        | Expr::Div(_, _)
        | Expr::Pow(_, _)
        | Expr::Function(_, _)
        | Expr::Constant(_)
        | Expr::Variable(_)
        | Expr::Matrix { .. }
        | Expr::SessionRef(_)
        | Expr::Hold(_) => (BigRational::one(), expr),
    }
}

fn are_structurally_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    compare_expr(ctx, a, b) == std::cmp::Ordering::Equal
}

fn is_one_term(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_one()
    } else {
        false
    }
}

/// Simplify numeric sums in exponents throughout an expression tree.
/// e.g., x^(1/2 + 1/3) -> x^(5/6)
pub fn simplify_numeric_exponents(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        Expr::Pow(base, exp) => {
            let new_base = simplify_numeric_exponents(ctx, base);

            if let Some(sum) = try_sum_numeric_fractions(ctx, exp) {
                let new_exp = ctx.add(Expr::Number(sum));
                ctx.add(Expr::Pow(new_base, new_exp))
            } else {
                let new_exp = simplify_numeric_exponents(ctx, exp);
                if new_base != base || new_exp != exp {
                    ctx.add(Expr::Pow(new_base, new_exp))
                } else {
                    expr
                }
            }
        }
        Expr::Add(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Sub(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                mul2_raw(ctx, nl, nr)
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Neg(e) => {
            let ne = simplify_numeric_exponents(ctx, e);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        Expr::Function(name, args) => {
            let mut changed = false;
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|a| {
                    let na = simplify_numeric_exponents(ctx, *a);
                    if na != *a {
                        changed = true;
                    }
                    na
                })
                .collect();
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }
        Expr::Hold(inner) => {
            let ne = simplify_numeric_exponents(ctx, inner);
            if ne != inner {
                ctx.add(Expr::Hold(ne))
            } else {
                expr
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let new_data: Vec<ExprId> = data
                .iter()
                .map(|a| {
                    let na = simplify_numeric_exponents(ctx, *a);
                    if na != *a {
                        changed = true;
                    }
                    na
                })
                .collect();
            if changed {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                expr
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
    }
}

/// Try to extract and sum all numeric fractions from an Add chain.
fn try_sum_numeric_fractions(ctx: &Context, exp: ExprId) -> Option<BigRational> {
    let mut addends: Vec<BigRational> = Vec::new();
    let mut stack = vec![exp];

    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Add(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Number(n) => {
                addends.push(n.clone());
            }
            Expr::Div(num, den) => {
                if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
                    if !d.is_zero() {
                        addends.push(n / d);
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }

    if addends.len() >= 2 {
        Some(addends.iter().sum())
    } else {
        None
    }
}
