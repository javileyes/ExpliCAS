use crate::expr_nary::{AddView, Sign};
use crate::expr_predicates::is_half_expr;
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;

/// Result of Dirichlet kernel detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirichletKernelResult {
    pub n: usize,         // The n in the sum (highest cosine multiple).
    pub base_var: ExprId, // The base variable x.
}

/// Try to detect Dirichlet kernel identity pattern:
/// 1 + 2*cos(x) + 2*cos(2x) + ... + 2*cos(nx) - sin((n+1/2)x)/sin(x/2) = 0
pub fn try_dirichlet_kernel_identity(ctx: &Context, expr: ExprId) -> Option<DirichletKernelResult> {
    let view = AddView::from_expr(ctx, expr);

    let mut has_one = false;
    let mut cosine_multiples: Vec<(usize, ExprId)> = Vec::new(); // (k, base_var)
    let mut sin_ratio: Option<(ExprId, ExprId)> = None; // (numerator arg, denominator arg)
    let mut sin_ratio_is_negative = false;

    for &(term, sign) in &view.terms {
        let is_positive = sign == Sign::Pos;
        let term_data = ctx.get(term).clone();

        // Check for constant 1.
        if let Expr::Number(n) = &term_data {
            if n.is_one() && is_positive {
                has_one = true;
                continue;
            }
        }

        // Check for 2*cos(k*x).
        if let Some((k, base)) = extract_cosine_multiple(ctx, term) {
            if is_positive {
                cosine_multiples.push((k, base));
            }
            continue;
        }

        // Check for sin(a)/sin(b) ratio.
        if let Some((num_arg, den_arg)) = extract_sin_ratio(ctx, term) {
            sin_ratio = Some((num_arg, den_arg));
            sin_ratio_is_negative = !is_positive; // Should be subtracted (negative).
        }
    }

    // Verify pattern: need 1, consecutive cosine multiples, and matching sin ratio.
    if !has_one || cosine_multiples.is_empty() {
        return None;
    }

    // Sort cosines by their multiple.
    cosine_multiples.sort_by_key(|(k, _)| *k);

    // Check if we have 1, 2, 3, ..., n.
    let n = cosine_multiples.len();
    for (i, (k, _)) in cosine_multiples.iter().enumerate() {
        if *k != i + 1 {
            return None;
        }
    }

    let base_var = cosine_multiples[0].1;

    // Verify sin ratio matches expected form: sin((n+1/2)*x)/sin(x/2).
    if let Some((num_arg, den_arg)) = sin_ratio {
        if sin_ratio_is_negative
            && is_half_angle(ctx, den_arg, base_var)
            && is_half_integer_multiple(ctx, num_arg, base_var, n)
        {
            return Some(DirichletKernelResult { n, base_var });
        }
    }

    None
}

// nary-lint: allow-binary (structural pattern match for 2*cos(k*x))
fn extract_cosine_multiple(ctx: &Context, expr: ExprId) -> Option<(usize, ExprId)> {
    // Pattern: 2 * cos(k * x) or cos(...) * 2
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check if one side is 2.
        let (two_side, other_side) = if is_number(ctx, *l, 2) {
            (Some(*l), *r)
        } else if is_number(ctx, *r, 2) {
            (Some(*r), *l)
        } else {
            return None;
        };

        if two_side.is_some() {
            if let Expr::Function(fn_id, args) = ctx.get(other_side) {
                let name = ctx.sym_name(*fn_id);
                if name == "cos" && args.len() == 1 {
                    return extract_multiple_of_var(ctx, args[0]);
                }
            }
        }
    }
    None
}

/// Extract sin(a)/sin(b) pattern, returning (a, b).
fn extract_sin_ratio(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Function(num_fn_id, num_args) = ctx.get(*num) {
            let num_name = ctx.sym_name(*num_fn_id);
            if num_name == "sin" && num_args.len() == 1 {
                if let Expr::Function(den_fn_id, den_args) = ctx.get(*den) {
                    let den_name = ctx.sym_name(*den_fn_id);
                    if den_name == "sin" && den_args.len() == 1 {
                        return Some((num_args[0], den_args[0]));
                    }
                }
            }
        }
    }
    None
}

// nary-lint: allow-binary (structural pattern match for half-angle)
fn is_half_angle(ctx: &Context, expr: ExprId, base_var: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Div(num, den) => *num == base_var && is_number(ctx, *den, 2),
        Expr::Mul(l, r) => {
            // Check for (1/2)*x or x*(1/2).
            (is_half_expr(ctx, *l) && *r == base_var) || (is_half_expr(ctx, *r) && *l == base_var)
        }
        _ => false,
    }
}

// nary-lint: allow-binary (structural pattern match for half-integer multiples)
fn is_half_integer_multiple(ctx: &Context, expr: ExprId, base_var: ExprId, n: usize) -> bool {
    let expected_num = 2 * n + 1; // (n+1/2) = (2n+1)/2

    match ctx.get(expr) {
        Expr::Div(num, den) => {
            if !is_number(ctx, *den, 2) {
                return false;
            }
            if let Expr::Mul(l, r) = ctx.get(*num) {
                (is_number(ctx, *l, expected_num as i32) && *r == base_var)
                    || (is_number(ctx, *r, expected_num as i32) && *l == base_var)
            } else {
                false
            }
        }
        Expr::Mul(l, r) => {
            // Check for ((2n+1)/2)*x pattern.
            let half_mult = num_rational::BigRational::new((expected_num as i64).into(), 2.into());
            if let Expr::Number(val) = ctx.get(*l) {
                if *val == half_mult && *r == base_var {
                    return true;
                }
            }
            if let Expr::Number(val) = ctx.get(*r) {
                if *val == half_mult && *l == base_var {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

// nary-lint: allow-binary (structural pattern match for k*x extraction)
fn extract_multiple_of_var(ctx: &Context, expr: ExprId) -> Option<(usize, ExprId)> {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            // k * x
            if let Expr::Number(n) = ctx.get(*l) {
                if let Some(k) = n.to_integer().to_u64_digits().1.first() {
                    if n.is_integer() && n.numer() > &0.into() {
                        return Some((*k as usize, *r));
                    }
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if let Some(k) = n.to_integer().to_u64_digits().1.first() {
                    if n.is_integer() && n.numer() > &0.into() {
                        return Some((*k as usize, *l));
                    }
                }
            }
            None
        }
        // Just x means k=1.
        Expr::Variable(_) => Some((1, expr)),
        _ => None,
    }
}

fn is_number(ctx: &Context, expr: ExprId, expected: i32) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n == num_rational::BigRational::from_integer(expected.into())
    } else {
        false
    }
}
