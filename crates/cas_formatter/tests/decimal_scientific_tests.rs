//! Scientific-notation rendering of `decimal(n)` payloads (the `approx()`
//! display wrapper): out-of-range magnitudes wear calculator form
//! (`1.26765060023*10^30` text, `1.26765060023 \times 10^{30}` LaTeX), and
//! the rendered string — a product/power under the hood — is parenthesized
//! wherever it stops being atomic (Pow bases, Div denominators).

use cas_ast::{Context, Expr, ExprId};
use cas_formatter::{DisplayExpr, LaTeXExpr};
use num_bigint::BigInt;
use num_rational::BigRational;

fn decimal_node(ctx: &mut Context, value: BigRational) -> ExprId {
    let n = ctx.add(Expr::Number(value));
    let sym = ctx.intern_symbol("decimal");
    ctx.add(Expr::Function(sym, vec![n]))
}

/// The rounded payload of `approx(2^100)`: 126765060023 · 10^19.
fn big_payload() -> BigRational {
    BigRational::from_integer(BigInt::from(126765060023u64) * BigInt::from(10u32).pow(19))
}

/// The rounded payload of `approx(1/2^100)`: 788860905221 / 10^42.
fn tiny_payload() -> BigRational {
    BigRational::new(788860905221u64.into(), BigInt::from(10u32).pow(42))
}

fn text(ctx: &Context, id: ExprId) -> String {
    DisplayExpr { context: ctx, id }.to_string()
}

fn latex(ctx: &Context, id: ExprId) -> String {
    LaTeXExpr { context: ctx, id }.to_latex()
}

#[test]
fn large_and_small_decimals_render_scientific() {
    let mut ctx = Context::new();
    let big = decimal_node(&mut ctx, big_payload());
    let tiny = decimal_node(&mut ctx, tiny_payload());

    assert_eq!(text(&ctx, big), "1.26765060023*10^30");
    assert_eq!(text(&ctx, tiny), "7.88860905221*10^(-31)");
    assert_eq!(latex(&ctx, big), "1.26765060023 \\times 10^{30}");
    assert_eq!(latex(&ctx, tiny), "7.88860905221 \\times 10^{-31}");
}

#[test]
fn in_range_decimals_keep_fixed_notation() {
    let mut ctx = Context::new();
    let half = decimal_node(&mut ctx, BigRational::new(1.into(), 2.into()));
    let edge = decimal_node(
        &mut ctx,
        BigRational::from_integer(BigInt::from(999_999_999_999u64)),
    );

    assert_eq!(text(&ctx, half), "0.5");
    assert_eq!(text(&ctx, edge), "999999999999");
    assert_eq!(latex(&ctx, edge), "999999999999");
}

#[test]
fn scientific_decimal_as_pow_base_is_parenthesized() {
    let mut ctx = Context::new();
    let big = decimal_node(&mut ctx, big_payload());
    let two = ctx.num(2);
    let squared = ctx.add(Expr::Pow(big, two));

    let rendered = text(&ctx, squared);
    assert_eq!(
        rendered, "(1.26765060023*10^30)^2",
        "sci base must be parenthesized, got: {rendered}"
    );
    let rendered = latex(&ctx, squared);
    assert!(
        rendered.contains("(1.26765060023 \\times 10^{30})"),
        "sci LaTeX base must be parenthesized, got: {rendered}"
    );

    // Fixed decimals stay atomic as bases (existing behavior).
    let half = decimal_node(&mut ctx, BigRational::new(1.into(), 2.into()));
    let squared_half = ctx.add(Expr::Pow(half, two));
    assert_eq!(text(&ctx, squared_half), "0.5^2");
}

#[test]
fn scientific_decimal_in_products_and_denominators() {
    let mut ctx = Context::new();
    let big = decimal_node(&mut ctx, big_payload());
    let x = ctx.var("x");

    // Product: value-correct without parens (`*` chains associate).
    let product = ctx.add(Expr::Mul(big, x));
    assert_eq!(text(&ctx, product), "1.26765060023*10^30 * x");

    // Denominator: must be parenthesized or the reading shifts.
    let quotient = ctx.add(Expr::Div(x, big));
    assert_eq!(text(&ctx, quotient), "x / (1.26765060023*10^30)");
}
