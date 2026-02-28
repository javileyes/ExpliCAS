use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetaFunctionRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Evaluate meta helper functions that operate on expression arguments.
///
/// Supported calls:
/// - `simplify(expr)` -> `expr`
/// - `factor(expr)` -> factored `expr` (or unchanged if irreducible)
/// - `expand(expr)` -> expanded `expr`
pub fn try_rewrite_meta_function_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<MetaFunctionRewrite> {
    let (fn_id, args) = if let Expr::Function(fn_id, args) = ctx.get(expr) {
        (*fn_id, args.clone())
    } else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let arg = args[0];
    match ctx.sym_name(fn_id) {
        "simplify" => Some(MetaFunctionRewrite {
            rewritten: arg,
            desc: "simplify(x) = x (already processed)",
        }),
        "factor" => {
            let factored = crate::factor::factor(ctx, arg);
            let desc = if factored != arg {
                "factor(x) -> factored form"
            } else {
                "factor(x) = x (irreducible)"
            };
            Some(MetaFunctionRewrite {
                rewritten: factored,
                desc,
            })
        }
        "expand" => Some(MetaFunctionRewrite {
            rewritten: crate::expand_ops::expand(ctx, arg),
            desc: "expand(x) -> expanded form",
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn rewrites_simplify_transparently() {
        let mut ctx = Context::new();
        let expr = parse("simplify(x+1)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_meta_function_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "simplify(x) = x (already processed)");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(rendered, "x + 1");
    }

    #[test]
    fn rewrites_expand_call() {
        let mut ctx = Context::new();
        let expr = parse("expand((x+1)^2)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_meta_function_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "expand(x) -> expanded form");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(rendered.contains("x^2"));
    }
}
