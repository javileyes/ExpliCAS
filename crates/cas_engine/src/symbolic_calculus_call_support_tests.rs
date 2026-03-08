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
