use std::collections::HashSet;

use cas_ast::{Context, Expr, ExprId};

/// Count unique nodes and max depth (dedup by ExprId).
pub fn count_nodes_dedup(ctx: &Context, root: ExprId) -> (usize, usize) {
    let mut seen = HashSet::new();
    let mut max_depth = 0;
    let mut stack: Vec<(ExprId, usize)> = vec![(root, 0)];

    while let Some((id, depth)) = stack.pop() {
        if !seen.insert(id) {
            continue; // Already visited (DAG sharing)
        }
        max_depth = max_depth.max(depth);

        let child_depth = depth + 1;
        match ctx.get(id) {
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push((*a, child_depth));
                stack.push((*b, child_depth));
            }
            Expr::Neg(e) | Expr::Hold(e) => stack.push((*e, child_depth)),
            Expr::Function(_, args) => {
                for &arg in args {
                    stack.push((arg, child_depth));
                }
            }
            Expr::Matrix { data, .. } => {
                for &elem in data {
                    stack.push((elem, child_depth));
                }
            }
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    (seen.len(), max_depth)
}

/// Check if expression contains any `Div(_, 0)` nodes.
pub fn has_zero_denominator(ctx: &Context, id: ExprId) -> bool {
    let mut stack = vec![id];
    while let Some(node_id) = stack.pop() {
        match ctx.get(node_id) {
            Expr::Div(num, den) => {
                if matches!(ctx.get(*den), Expr::Number(n) if num_traits::Zero::is_zero(n)) {
                    return true;
                }
                stack.push(*num);
                stack.push(*den);
            }
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

/// Deep-copy an expression subtree from `src` context into `dst` context.
pub fn transplant_expr(src: &Context, id: ExprId, dst: &mut Context) -> ExprId {
    match src.get(id) {
        Expr::Number(n) => dst.add(Expr::Number(n.clone())),
        Expr::Constant(c) => dst.add(Expr::Constant(c.clone())),
        Expr::Variable(sym) => {
            // Preserve variable name
            let name = src.sym_name(*sym);
            dst.var(name)
        }
        Expr::SessionRef(r) => dst.add(Expr::SessionRef(*r)),
        Expr::Add(a, b) => {
            let ta = transplant_expr(src, *a, dst);
            let tb = transplant_expr(src, *b, dst);
            dst.add(Expr::Add(ta, tb))
        }
        Expr::Sub(a, b) => {
            let ta = transplant_expr(src, *a, dst);
            let tb = transplant_expr(src, *b, dst);
            dst.add(Expr::Sub(ta, tb))
        }
        Expr::Mul(a, b) => {
            let ta = transplant_expr(src, *a, dst);
            let tb = transplant_expr(src, *b, dst);
            dst.add(Expr::Mul(ta, tb))
        }
        Expr::Div(a, b) => {
            let ta = transplant_expr(src, *a, dst);
            let tb = transplant_expr(src, *b, dst);
            dst.add(Expr::Div(ta, tb))
        }
        Expr::Pow(a, b) => {
            let ta = transplant_expr(src, *a, dst);
            let tb = transplant_expr(src, *b, dst);
            dst.add(Expr::Pow(ta, tb))
        }
        Expr::Neg(inner) => {
            let ti = transplant_expr(src, *inner, dst);
            dst.add(Expr::Neg(ti))
        }
        Expr::Function(name, args) => {
            let targs: Vec<ExprId> = args
                .iter()
                .map(|&arg| transplant_expr(src, arg, dst))
                .collect();
            dst.add(Expr::Function(*name, targs))
        }
        Expr::Hold(inner) => {
            let ti = transplant_expr(src, *inner, dst);
            dst.add(Expr::Hold(ti))
        }
        Expr::Matrix { rows, cols, data } => {
            let tdata: Vec<ExprId> = data
                .iter()
                .map(|&elem| transplant_expr(src, elem, dst))
                .collect();
            dst.add(Expr::Matrix {
                rows: *rows,
                cols: *cols,
                data: tdata,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_count_counts_shared_nodes_once() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sum = ctx.add(Expr::Add(x, x)); // shared x
        let (count, _depth) = count_nodes_dedup(&ctx, sum);
        assert_eq!(count, 2);
    }

    #[test]
    fn detects_zero_denominator() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let zero = ctx.num(0);
        let div = ctx.add(Expr::Div(one, zero));
        assert!(has_zero_denominator(&ctx, div));
    }

    #[test]
    fn transplant_preserves_shape() {
        let mut src = Context::new();
        let a = src.var("a");
        let b = src.var("b");
        let expr = src.add(Expr::Mul(a, b));

        let mut dst = Context::new();
        let transplanted = transplant_expr(&src, expr, &mut dst);

        match dst.get(transplanted) {
            Expr::Mul(_, _) => {}
            other => panic!("expected Mul, got {:?}", other),
        }
    }
}
