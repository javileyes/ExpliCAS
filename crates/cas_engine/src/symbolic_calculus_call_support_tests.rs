use super::{
    render_diff_desc_with, render_integrate_desc_with, try_extract_diff_call,
    try_extract_integrate_call,
};
use cas_ast::Context;
use cas_parser::parse;

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
