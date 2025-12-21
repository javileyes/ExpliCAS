use crate::{Constant, Context, Expr, ExprId};
use std::cmp::Ordering;

/// Iterative expression comparison using explicit stack.
/// This prevents stack overflow on deeply nested expressions.
pub fn compare_expr(context: &Context, a: ExprId, b: ExprId) -> Ordering {
    use Expr::*;

    let mut stack: Vec<(ExprId, ExprId)> = vec![(a, b)];

    while let Some((x, y)) = stack.pop() {
        if x == y {
            continue;
        }

        let ex = context.get(x);
        let ey = context.get(y);

        let kx = get_rank(ex);
        let ky = get_rank(ey);
        if kx != ky {
            return kx.cmp(&ky);
        }

        match (ex, ey) {
            (Number(n1), Number(n2)) => {
                let c = n1.cmp(n2);
                if c != Ordering::Equal {
                    return c;
                }
            }
            (Constant(c1), Constant(c2)) => {
                let c = compare_constant(c1, c2);
                if c != Ordering::Equal {
                    return c;
                }
            }
            (Variable(v1), Variable(v2)) => {
                let c = v1.cmp(v2);
                if c != Ordering::Equal {
                    return c;
                }
            }
            (SessionRef(s1), SessionRef(s2)) => {
                let c = s1.cmp(s2);
                if c != Ordering::Equal {
                    return c;
                }
            }
            (Function(n1, args1), Function(n2, args2)) => {
                let c = n1.cmp(n2);
                if c != Ordering::Equal {
                    return c;
                }
                let c = args1.len().cmp(&args2.len());
                if c != Ordering::Equal {
                    return c;
                }
                // Push args in reverse order so first arg is compared first
                for (a1, a2) in args1.iter().zip(args2.iter()).rev() {
                    stack.push((*a1, *a2));
                }
            }
            (Neg(e1), Neg(e2)) => {
                stack.push((*e1, *e2));
            }
            (Pow(b1, e1), Pow(b2, e2)) => {
                // Push exponent first (will be popped last, so base compared first)
                stack.push((*e1, *e2));
                stack.push((*b1, *b2));
            }
            (Add(l1, r1), Add(l2, r2))
            | (Sub(l1, r1), Sub(l2, r2))
            | (Mul(l1, r1), Mul(l2, r2))
            | (Div(l1, r1), Div(l2, r2)) => {
                // Push right first so left is compared first
                stack.push((*r1, *r2));
                stack.push((*l1, *l2));
            }
            (
                Matrix {
                    rows: r1,
                    cols: c1,
                    data: d1,
                },
                Matrix {
                    rows: r2,
                    cols: c2,
                    data: d2,
                },
            ) => {
                let c = (r1, c1).cmp(&(r2, c2));
                if c != Ordering::Equal {
                    return c;
                }
                let c = d1.len().cmp(&d2.len());
                if c != Ordering::Equal {
                    return c;
                }
                // Compare by ExprId index to avoid deep recursion
                for (e1, e2) in d1.iter().zip(d2.iter()) {
                    let c = e1.index().cmp(&e2.index());
                    if c != Ordering::Equal {
                        return c;
                    }
                }
            }
            _ => {
                // Should be unreachable if ranks are correct
            }
        }
    }

    Ordering::Equal
}

pub fn get_rank(expr: &Expr) -> u8 {
    use Expr::*;
    match expr {
        Number(_) => 0,
        Constant(_) => 1,
        Variable(_) => 2,
        Function(_, _) => 3,
        Neg(_) => 4,
        Pow(_, _) => 5,
        Mul(_, _) => 6,
        Div(_, _) => 7,
        Add(_, _) => 8,
        Sub(_, _) => 9,
        Matrix { .. } => 10,
        SessionRef(_) => 11,
    }
}

fn compare_constant(c1: &Constant, c2: &Constant) -> Ordering {
    use Constant::*;
    // Arbitrary order for constants
    let r1 = match c1 {
        Pi => 0,
        E => 1,
        Infinity => 2,
        Undefined => 3,
        I => 4,
    };
    let r2 = match c2 {
        Pi => 0,
        E => 1,
        Infinity => 2,
        Undefined => 3,
        I => 4,
    };
    r1.cmp(&r2)
}
