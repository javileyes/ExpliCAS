use cas_ast::{Context, Expr, ExprId};
use cas_math::expand_call_support::expand_explicit_arg_with_post_compaction;

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
            let factored = cas_math::factor::factor(ctx, arg);
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
            rewritten: expand_explicit_arg_with_post_compaction(ctx, arg),
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

    #[test]
    fn rewrites_expand_call_with_compact_univariate_polynomial_terms() {
        let mut ctx = Context::new();
        let expr = parse("expand(3-(x^2+2*x+1)^2)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_meta_function_expr(&mut ctx, expr).expect("rewrite");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(rendered, "2 - x^4 - 4 * x^3 - 6 * x^2 - 4 * x");
    }

    #[test]
    fn rewrites_factor_call_with_multivar_common_monomial() {
        let mut ctx = Context::new();
        let expr = parse("factor(y^2*z^2 + 2*y^2*z + y^2)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_meta_function_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "factor(x) -> factored form");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(
            rendered.contains("y^2") && rendered.contains("(z + 1)^2"),
            "unexpected factor shape: {rendered}"
        );
    }

    #[test]
    fn rewrites_factor_call_with_multivar_common_numeric_content() {
        let mut ctx = Context::new();
        let expr = parse("factor(2*x + 4*y)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_meta_function_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "factor(x) -> factored form");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(
            rendered.contains("2 * (x + 2 * y)"),
            "unexpected factor shape: {rendered}"
        );
    }
}
