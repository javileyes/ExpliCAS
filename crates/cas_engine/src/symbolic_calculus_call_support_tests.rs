use super::{
    render_diff_desc_with, render_integrate_desc_with, try_desugar_higher_order_diff,
    try_extract_diff_call, try_extract_integrate_call,
};
use cas_ast::{Context, Expr, ExprId};
use cas_parser::parse;

/// The ordered differentiation variables encoded by a nested two-argument diff chain,
/// outermost (last-applied) first. `None` if the expression is not a nested diff chain.
fn diff_chain_vars(ctx: &Context, mut id: ExprId) -> Vec<String> {
    let mut vars = Vec::new();
    while let Expr::Function(fn_id, args) = ctx.get(id) {
        if ctx.sym_name(*fn_id) != "diff" || args.len() != 2 {
            break;
        }
        if let Expr::Variable(sym) = ctx.get(args[1]) {
            vars.push(ctx.sym_name(*sym).to_string());
        } else {
            break;
        }
        id = args[0];
    }
    vars
}

#[test]
fn extracts_integrate_with_default_var() {
    let mut ctx = Context::new();
    let expr = parse("integrate(x^2)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err:?}"));
    let call =
        try_extract_integrate_call(&ctx, expr).unwrap_or_else(|| panic!("missing integrate call"));
    assert_eq!(call.var_name, "x");
}

#[test]
fn extracts_integrate_with_explicit_var() {
    let mut ctx = Context::new();
    let expr = parse("integrate(y^2, y)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err:?}"));
    let call =
        try_extract_integrate_call(&ctx, expr).unwrap_or_else(|| panic!("missing integrate call"));
    assert_eq!(call.var_name, "y");
}

#[test]
fn extracts_diff_call() {
    let mut ctx = Context::new();
    let expr = parse("diff(sin(x), x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err:?}"));
    let call = try_extract_diff_call(&ctx, expr).unwrap_or_else(|| panic!("missing diff call"));
    assert_eq!(call.var_name, "x");
}

#[test]
fn renders_descriptions() {
    let mut ctx = Context::new();
    let expr = parse("diff(sin(x), x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err:?}"));
    let call = try_extract_diff_call(&ctx, expr).unwrap_or_else(|| panic!("missing diff call"));
    let desc = render_diff_desc_with(&call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: &ctx, id })
    });
    assert!(desc.starts_with("diff("));

    let iexpr = parse("integrate(x^2)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err:?}"));
    let icall =
        try_extract_integrate_call(&ctx, iexpr).unwrap_or_else(|| panic!("missing integrate call"));
    let idesc = render_integrate_desc_with(&icall, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: &ctx, id })
    });
    assert!(idesc.starts_with("integrate("));
}

#[test]
fn desugars_higher_order_diff_to_repeated_variable() {
    let mut ctx = Context::new();
    let expr = parse("diff(x^4, x, 3)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err:?}"));
    let nested =
        try_desugar_higher_order_diff(&mut ctx, expr).unwrap_or_else(|| panic!("missing desugar"));
    // Three applications, all w.r.t. x.
    assert_eq!(diff_chain_vars(&ctx, nested), vec!["x", "x", "x"]);
}

#[test]
fn desugars_mixed_partial_to_variable_sequence() {
    let mut ctx = Context::new();
    let expr =
        parse("diff(x^3*y^2, x, y)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err:?}"));
    let nested =
        try_desugar_higher_order_diff(&mut ctx, expr).unwrap_or_else(|| panic!("missing desugar"));
    // diff(diff(f, x), y): outermost (last-applied) is y, innermost is x.
    assert_eq!(diff_chain_vars(&ctx, nested), vec!["y", "x"]);
}

#[test]
fn desugars_mixed_counts_in_sympy_order() {
    let mut ctx = Context::new();
    let expr =
        parse("diff(x^3*y^2, x, 2, y, 2)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err:?}"));
    let nested =
        try_desugar_higher_order_diff(&mut ctx, expr).unwrap_or_else(|| panic!("missing desugar"));
    // x applied twice then y applied twice; chain read outermost-first is y,y,x,x.
    assert_eq!(diff_chain_vars(&ctx, nested), vec!["y", "y", "x", "x"]);
}

#[test]
fn desugars_order_one_to_single_diff() {
    let mut ctx = Context::new();
    let expr = parse("diff(x^4, x, 1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err:?}"));
    let nested =
        try_desugar_higher_order_diff(&mut ctx, expr).unwrap_or_else(|| panic!("missing desugar"));
    assert_eq!(diff_chain_vars(&ctx, nested), vec!["x"]);
}

#[test]
fn higher_order_desugar_rejects_two_argument_and_malformed_calls() {
    let cases = [
        "diff(x^4, x)",      // ordinary two-argument diff is left to DiffRule
        "diff(x^4, 2, x)",   // a count may not lead the variable list
        "diff(x^4, x, 0)",   // order zero is not a positive integer
        "diff(x^4, x, -1)",  // negative order
        "diff(x^2, x, 1/2)", // fractional (non-integer) order
        "sin(x)",            // not a diff call at all
        // AMBIGUOUS symbolic order: the trailing symbol is not free in the target,
        // so it is neither a genuine mixed partial nor a supported symbolic order —
        // decline (honest echo) instead of computing the wrong `∂/∂n ∂/∂x f = 0`.
        "diff(e^x, x, n)",
        "diff(sin(x), x, n)",
        "diff(x^2, x, y)",
    ];
    for src in cases {
        let mut ctx = Context::new();
        let expr = parse(src, &mut ctx).unwrap_or_else(|err| panic!("parse {src}: {err:?}"));
        assert!(
            try_desugar_higher_order_diff(&mut ctx, expr).is_none(),
            "expected no higher-order desugar for `{src}`"
        );
    }
}
