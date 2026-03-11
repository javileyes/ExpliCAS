mod common;

use std::hint::black_box;

use cas_ast::{Context, ExprId};
use cas_formatter::{clean_display_string, render_expr, DisplayExprStyled, StylePreferences};
use cas_parser::parse;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

struct RenderCase {
    name: &'static str,
    ctx: Context,
    expr: ExprId,
    styled_render: String,
}

fn parse_case(name: &'static str, input: &str, style: &StylePreferences) -> RenderCase {
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("parse failed");
    let styled_render = DisplayExprStyled::new(&ctx, expr, style).to_string();
    RenderCase {
        name,
        ctx,
        expr,
        styled_render,
    }
}

fn bench_frontend_render(c: &mut Criterion) {
    let style = StylePreferences::default();
    let cases = vec![
        parse_case("light/x_plus_1", "x + 1", &style),
        parse_case("light/numeric_add_chain", "2 * 3 + 4", &style),
        parse_case(
            "gcd/scalar_multiple_fraction",
            "(2*x + 2*y)/(4*x + 4*y)",
            &style,
        ),
        parse_case(
            "gcd/common_factor_fraction",
            "((x+y)*(a+b))/((x+y)*(c+d))",
            &style,
        ),
        parse_case("heavy/nested_root", "sqrt(12*x^3)", &style),
        parse_case("heavy/abs_square", "((5*x + 8)^2)^(1/2)", &style),
        parse_case("complex/gaussian_div", "(3 + 4*i)/(1 + 2*i)", &style),
        parse_case(
            "trig/pythagorean_chain",
            "sin(2*x + 1)^2 + cos(1 + 2*x)^2",
            &style,
        ),
    ];

    let mut group = c.benchmark_group("frontend_render");
    common::configure_standard_group(&mut group);

    group.bench_function("display_expr_batch/standard_8", |b| {
        b.iter(|| {
            for case in &cases {
                black_box(render_expr(&case.ctx, case.expr));
            }
        })
    });

    group.bench_function("styled_clean_batch/standard_8", |b| {
        b.iter(|| {
            for case in &cases {
                let rendered = DisplayExprStyled::new(&case.ctx, case.expr, &style).to_string();
                black_box(clean_display_string(&rendered));
            }
        })
    });

    group.bench_function("clean_only_batch/standard_8", |b| {
        b.iter(|| {
            for case in &cases {
                black_box(clean_display_string(&case.styled_render));
            }
        })
    });

    for case in &cases {
        group.bench_with_input(
            BenchmarkId::new("display_expr", case.name),
            case,
            |b, case| {
                b.iter(|| black_box(render_expr(&case.ctx, case.expr)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("styled_clean", case.name),
            case,
            |b, case| {
                b.iter(|| {
                    let rendered = DisplayExprStyled::new(&case.ctx, case.expr, &style).to_string();
                    black_box(clean_display_string(&rendered));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("clean_only", case.name),
            case,
            |b, case| {
                b.iter(|| black_box(clean_display_string(&case.styled_render)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_frontend_render);
criterion_main!(benches);
