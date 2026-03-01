//! Parsing/render helpers for `integrate(...)` and `diff(...)` call forms.

use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedVarCall {
    pub target: ExprId,
    pub var_name: String,
}

/// Parse `integrate(target, var)` and `integrate(target)` (defaults to `x`).
pub fn try_extract_integrate_call(ctx: &Context, expr: ExprId) -> Option<NamedVarCall> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != "integrate" {
        return None;
    }

    if args.len() == 2 {
        let target = args[0];
        let var_expr = args[1];
        let Expr::Variable(var_sym) = ctx.get(var_expr) else {
            return None;
        };
        return Some(NamedVarCall {
            target,
            var_name: ctx.sym_name(*var_sym).to_string(),
        });
    }

    if args.len() == 1 {
        return Some(NamedVarCall {
            target: args[0],
            var_name: "x".to_string(),
        });
    }

    None
}

/// Parse `diff(target, var)` with explicit variable.
pub fn try_extract_diff_call(ctx: &Context, expr: ExprId) -> Option<NamedVarCall> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != "diff" || args.len() != 2 {
        return None;
    }

    let target = args[0];
    let var_expr = args[1];
    let Expr::Variable(var_sym) = ctx.get(var_expr) else {
        return None;
    };
    Some(NamedVarCall {
        target,
        var_name: ctx.sym_name(*var_sym).to_string(),
    })
}

/// Render `integrate(target, var)` description using a caller-provided expression renderer.
pub fn render_integrate_desc_with<F>(call: &NamedVarCall, mut render_expr: F) -> String
where
    F: FnMut(ExprId) -> String,
{
    format!("integrate({}, {})", render_expr(call.target), call.var_name)
}

/// Render `diff(target, var)` description using a caller-provided expression renderer.
pub fn render_diff_desc_with<F>(call: &NamedVarCall, mut render_expr: F) -> String
where
    F: FnMut(ExprId) -> String,
{
    format!("diff({}, {})", render_expr(call.target), call.var_name)
}

#[cfg(test)]
mod tests {
    use super::{
        render_diff_desc_with, render_integrate_desc_with, try_extract_diff_call,
        try_extract_integrate_call,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn extracts_integrate_with_default_var() {
        let mut ctx = Context::new();
        let expr = parse("integrate(x^2)", &mut ctx).expect("parse");
        let call = try_extract_integrate_call(&ctx, expr).expect("call");
        assert_eq!(call.var_name, "x");
    }

    #[test]
    fn extracts_integrate_with_explicit_var() {
        let mut ctx = Context::new();
        let expr = parse("integrate(y^2, y)", &mut ctx).expect("parse");
        let call = try_extract_integrate_call(&ctx, expr).expect("call");
        assert_eq!(call.var_name, "y");
    }

    #[test]
    fn extracts_diff_call() {
        let mut ctx = Context::new();
        let expr = parse("diff(sin(x), x)", &mut ctx).expect("parse");
        let call = try_extract_diff_call(&ctx, expr).expect("call");
        assert_eq!(call.var_name, "x");
    }

    #[test]
    fn renders_descriptions() {
        let mut ctx = Context::new();
        let expr = parse("diff(sin(x), x)", &mut ctx).expect("parse");
        let call = try_extract_diff_call(&ctx, expr).expect("call");
        let desc = render_diff_desc_with(&call, |id| {
            format!("{}", cas_formatter::DisplayExpr { context: &ctx, id })
        });
        assert!(desc.starts_with("diff("));

        let iexpr = parse("integrate(x^2)", &mut ctx).expect("parse");
        let icall = try_extract_integrate_call(&ctx, iexpr).expect("call");
        let idesc = render_integrate_desc_with(&icall, |id| {
            format!("{}", cas_formatter::DisplayExpr { context: &ctx, id })
        });
        assert!(idesc.starts_with("integrate("));
    }
}
