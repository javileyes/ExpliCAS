// ========== Numeric Evaluation ==========

use cas_ast::{Context, Expr, ExprId};

/// Default depth limit for numeric evaluation.
/// Prevents stack overflow on deeply nested expressions.
pub const DEFAULT_NUMERIC_EVAL_DEPTH: usize = 50;

/// Extract a rational constant from an expression, handling multiple representations.
/// Uses default depth limit (50) to prevent stack overflow.
///
/// Supports (all must be purely numeric - returns None if any variable/function present):
/// - `Number(n)` - direct rational
/// - `Div(a, b)` - fraction (recursive)
/// - `Neg(a)` - negation (recursive)
/// - `Mul(a, b)` - product (recursive)
/// - `Add(a, b)` - sum (recursive)
/// - `Sub(a, b)` - difference (recursive)
///
/// This is the canonical helper for numeric evaluation. Used by:
/// - `SemanticEqualityChecker::try_evaluate_numeric`
/// - `EvaluatePowerRule` for exponent matching
pub fn as_rational_const(ctx: &Context, expr: ExprId) -> Option<num_rational::BigRational> {
    as_rational_const_depth(ctx, expr, DEFAULT_NUMERIC_EVAL_DEPTH)
}

/// Extract a rational constant with explicit depth limit.
/// Returns None if depth is exhausted (prevents stack overflow on deep expressions).
pub fn as_rational_const_depth(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
) -> Option<num_rational::BigRational> {
    use num_traits::Zero;

    if depth == 0 {
        return None; // Depth budget exhausted
    }

    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),

        Expr::Div(num, den) => {
            let n = as_rational_const_depth(ctx, *num, depth - 1)?;
            let d = as_rational_const_depth(ctx, *den, depth - 1)?;
            if !d.is_zero() {
                Some(n / d)
            } else {
                None
            }
        }

        Expr::Neg(inner) => {
            let val = as_rational_const_depth(ctx, *inner, depth - 1)?;
            Some(-val)
        }

        Expr::Mul(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv * rv)
        }

        Expr::Add(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv + rv)
        }

        Expr::Sub(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv - rv)
        }

        // Variables, Constants, Functions, Pow, Matrix -> not purely numeric
        _ => None,
    }
}

/// Check if an expression contains an integral (for auto-context detection).
///
/// Searches the expression tree for `integrate(...)` function calls.
/// Uses iterative traversal to avoid stack overflow on deep expressions.
pub fn contains_integral(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];

    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Function(name, args) => {
                let fn_name = ctx.sym_name(*name);
                if fn_name == "integrate" || fn_name == "int" {
                    return true;
                }
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => {
                stack.push(*inner);
            }
            Expr::Matrix { data, .. } => {
                for elem in data {
                    stack.push(*elem);
                }
            }
            Expr::Div(num, den) => {
                stack.push(*num);
                stack.push(*den);
            }
            // Leaf nodes: nothing to push
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

/// Check if an expression contains the imaginary unit `i` anywhere.
/// Check if an expression contains the imaginary unit `i` or imaginary-producing expressions.
/// Detects: Constant::I, sqrt(-1), (-1)^(1/2), and similar patterns.
/// Uses iterative traversal to avoid stack overflow on deep expressions.
pub fn contains_i(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];

    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Constant(c) if *c == cas_ast::Constant::I => {
                return true;
            }
            // Check for sqrt(-1) pattern
            Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == "sqrt" && args.len() == 1 => {
                if is_negative_one(ctx, args[0]) {
                    return true;
                }
                // Still need to traverse the arg for nested i
                stack.push(args[0]);
            }
            // Check for (-1)^(1/2) pattern
            Expr::Pow(base, exp) => {
                if is_negative_one(ctx, *base) && is_one_half(ctx, *exp) {
                    return true;
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => {
                stack.push(*inner);
            }
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Matrix { data, .. } => {
                for elem in data {
                    stack.push(*elem);
                }
            }
            Expr::Div(num, den) => {
                stack.push(*num);
                stack.push(*den);
            }
            // Leaf nodes: nothing to push
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

/// Check if an expression represents -1
fn is_negative_one(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == num_rational::BigRational::from_integer((-1).into()),
        Expr::Neg(inner) => {
            matches!(
                ctx.get(*inner),
                Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
            )
        }
        _ => false,
    }
}

/// Check if an expression represents 1/2
fn is_one_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == num_rational::BigRational::new(1.into(), 2.into()),
        Expr::Div(num, den) => {
            matches!((ctx.get(*num), ctx.get(*den)),
                (Expr::Number(n), Expr::Number(d))
                if *n == num_rational::BigRational::from_integer(1.into())
                && *d == num_rational::BigRational::from_integer(2.into())
            )
        }
        _ => false,
    }
}
