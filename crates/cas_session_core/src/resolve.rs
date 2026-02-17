use cas_ast::{Context, Expr, ExprId};

use crate::types::EntryId;

/// Parse legacy session reference names like `#123`.
pub fn parse_legacy_session_ref(name: &str) -> Option<EntryId> {
    if !name.starts_with('#') || name.len() <= 1 {
        return None;
    }
    if !name[1..].chars().all(char::is_numeric) {
        return None;
    }
    name[1..].parse::<EntryId>().ok()
}

/// Rewrite session references (`Expr::SessionRef` and legacy `Variable("#N")`)
/// in an expression tree using the provided resolver callback.
///
/// The callback receives:
/// - the mutable context,
/// - the `ExprId` of the reference node in the current AST,
/// - the parsed entry id.
pub fn rewrite_session_refs<E, F>(
    ctx: &mut Context,
    expr: ExprId,
    resolver: &mut F,
) -> Result<ExprId, E>
where
    F: FnMut(&mut Context, ExprId, EntryId) -> Result<ExprId, E>,
{
    let node = ctx.get(expr).clone();

    match node {
        Expr::SessionRef(id) => resolver(ctx, expr, id),
        Expr::Variable(sym_id) => match parse_legacy_session_ref(ctx.sym_name(sym_id)) {
            Some(id) => resolver(ctx, expr, id),
            None => Ok(expr),
        },

        Expr::Add(l, r) => rewrite_binary(ctx, expr, l, r, Expr::Add, resolver),
        Expr::Sub(l, r) => rewrite_binary(ctx, expr, l, r, Expr::Sub, resolver),
        Expr::Mul(l, r) => rewrite_binary(ctx, expr, l, r, Expr::Mul, resolver),
        Expr::Div(l, r) => rewrite_binary(ctx, expr, l, r, Expr::Div, resolver),
        Expr::Pow(l, r) => rewrite_binary(ctx, expr, l, r, Expr::Pow, resolver),

        Expr::Neg(inner) => {
            let new_inner = rewrite_session_refs(ctx, inner, resolver)?;
            if new_inner == inner {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Neg(new_inner)))
            }
        }

        Expr::Function(name, args) => {
            let mut changed = false;
            let mut new_args = Vec::with_capacity(args.len());
            for arg in &args {
                let new_arg = rewrite_session_refs(ctx, *arg, resolver)?;
                if new_arg != *arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                Ok(ctx.add(Expr::Function(name, new_args)))
            } else {
                Ok(expr)
            }
        }

        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let mut new_data = Vec::with_capacity(data.len());
            for elem in &data {
                let new_elem = rewrite_session_refs(ctx, *elem, resolver)?;
                if new_elem != *elem {
                    changed = true;
                }
                new_data.push(new_elem);
            }
            if changed {
                Ok(ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                }))
            } else {
                Ok(expr)
            }
        }

        Expr::Hold(inner) => {
            let new_inner = rewrite_session_refs(ctx, inner, resolver)?;
            if new_inner == inner {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Hold(new_inner)))
            }
        }

        Expr::Number(_) | Expr::Constant(_) => Ok(expr),
    }
}

fn rewrite_binary<E, F, Ctor>(
    ctx: &mut Context,
    expr: ExprId,
    left: ExprId,
    right: ExprId,
    ctor: Ctor,
    resolver: &mut F,
) -> Result<ExprId, E>
where
    F: FnMut(&mut Context, ExprId, EntryId) -> Result<ExprId, E>,
    Ctor: Fn(ExprId, ExprId) -> Expr,
{
    let new_left = rewrite_session_refs(ctx, left, resolver)?;
    let new_right = rewrite_session_refs(ctx, right, resolver)?;
    if new_left == left && new_right == right {
        Ok(expr)
    } else {
        Ok(ctx.add(ctor(new_left, new_right)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_legacy_session_ref_valid() {
        assert_eq!(parse_legacy_session_ref("#1"), Some(1));
        assert_eq!(parse_legacy_session_ref("#42"), Some(42));
    }

    #[test]
    fn parse_legacy_session_ref_invalid() {
        assert_eq!(parse_legacy_session_ref("x"), None);
        assert_eq!(parse_legacy_session_ref("#"), None);
        assert_eq!(parse_legacy_session_ref("#abc"), None);
    }

    #[test]
    fn rewrite_session_refs_replaces_explicit_ref() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let ref1 = ctx.add(Expr::SessionRef(1));
        let input = ctx.add(Expr::Add(ref1, one));

        let out = rewrite_session_refs(&mut ctx, input, &mut |ctx,
                                                              _node,
                                                              id|
         -> Result<ExprId, ()> {
            Ok(ctx.num(id as i64))
        })
        .unwrap();
        let out_s = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &ctx,
                id: out
            }
        );
        assert!(out_s.contains('1'));
    }

    #[test]
    fn rewrite_session_refs_replaces_legacy_variable() {
        let mut ctx = Context::new();
        let legacy = ctx.var("#2");
        let input = ctx.add(Expr::Neg(legacy));

        let out = rewrite_session_refs(&mut ctx, input, &mut |ctx,
                                                              _node,
                                                              id|
         -> Result<ExprId, ()> {
            Ok(ctx.num(id as i64))
        })
        .unwrap();
        let out_s = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &ctx,
                id: out
            }
        );
        assert!(out_s.contains('2'));
        assert!(!out_s.contains("#2"));
    }
}
