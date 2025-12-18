//! mm_gcd benchmark: Famous polynomial GCD benchmark from benruijl.
//!
//! Benchmark: <https://gist.github.com/benruijl/3c53b1b0aea88b978ae609e73693fdbc>
//! Tests: (a·g, b·g) + gcd where a, b, g are 7-variable polynomials^7
//!
//! This measures raw MultiPoly multiplication and GCD performance
//! without going through the simplifier pipeline.

use cas_ast::Context;
use cas_engine::multipoly::{
    gcd_multivar_layer25, multipoly_from_expr, Layer25Budget, MultiPoly, PolyBudget,
};
use cas_parser::parse;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

/// Build the three polynomials a, b, g as MultiPoly.
/// Uses parser + multipoly_from_expr which handles Pow(linear, 7) via pow_poly.
fn build_polys() -> (MultiPoly, MultiPoly, MultiPoly) {
    let mut ctx = Context::new();
    let budget = PolyBudget {
        max_terms: 1_000_000,
        max_total_degree: 100,
    };

    // a = (1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^7 - 1
    let a_id = parse(
        "(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^7 - 1",
        &mut ctx,
    )
    .expect("parse a failed");

    // b = (1 - 3*x1 - 5*x2 - 7*x3 + 9*x4 - 11*x5 - 13*x6 + 15*x7)^7 + 1
    let b_id = parse(
        "(1 - 3*x1 - 5*x2 - 7*x3 + 9*x4 - 11*x5 - 13*x6 + 15*x7)^7 + 1",
        &mut ctx,
    )
    .expect("parse b failed");

    // g = (1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 - 15*x7)^7 + 3
    let g_id = parse(
        "(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 - 15*x7)^7 + 3",
        &mut ctx,
    )
    .expect("parse g failed");

    let a = multipoly_from_expr(&ctx, a_id, &budget).expect("convert a failed");
    let b = multipoly_from_expr(&ctx, b_id, &budget).expect("convert b failed");
    let g = multipoly_from_expr(&ctx, g_id, &budget).expect("convert g failed");

    (a, b, g)
}

/// Build the products a*g and b*g once for GCD-only benchmarks.
fn build_products(a: &MultiPoly, b: &MultiPoly, g: &MultiPoly) -> (MultiPoly, MultiPoly) {
    let budget = PolyBudget {
        max_terms: 10_000_000,
        max_total_degree: 200,
    };
    let ag = a.mul(g, &budget).expect("a*g failed");
    let bg = b.mul(g, &budget).expect("b*g failed");
    (ag, bg)
}

fn bench_mm_gcd(c: &mut Criterion) {
    // Build polys once (expensive but outside timing)
    let (a, b, g) = build_polys();

    println!("\n=== mm_gcd Benchmark Setup ===");
    println!(
        "  a: {} terms, total_degree {}",
        a.num_terms(),
        a.total_degree()
    );
    println!(
        "  b: {} terms, total_degree {}",
        b.num_terms(),
        b.total_degree()
    );
    println!(
        "  g: {} terms, total_degree {}",
        g.num_terms(),
        g.total_degree()
    );

    // Build products for GCD-only bench
    let (ag, bg) = build_products(&a, &b, &g);
    println!(
        "  a*g: {} terms, total_degree {}",
        ag.num_terms(),
        ag.total_degree()
    );
    println!(
        "  b*g: {} terms, total_degree {}",
        bg.num_terms(),
        bg.total_degree()
    );

    // Sanity check (outside timing): gcd(ag, bg) should be divisible by g
    let gcd_budget = Layer25Budget::default();
    {
        let d = gcd_multivar_layer25(&ag, &bg, &gcd_budget).expect("gcd failed");
        println!("  gcd(ag,bg): {} terms", d.num_terms());

        // Verify: g divides d (since gcd should be scalar * g)
        if let Some(q) = d.div_exact(&g) {
            println!(
                "  ✓ g divides gcd, quotient is_constant={}",
                q.is_constant()
            );
            if !q.is_constant() {
                println!(
                    "    WARNING: quotient has {} terms, expected constant",
                    q.num_terms()
                );
            }
        } else {
            println!("  ✗ WARNING: g does not divide gcd exactly");
        }
    }
    println!("=================================\n");

    // Configure for slow benchmarks
    let mut group = c.benchmark_group("mm_gcd");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));

    // Benchmark 1: Multiplication only (a*g + b*g)
    group.bench_function("mul_only", |bencher| {
        let budget = PolyBudget {
            max_terms: 10_000_000,
            max_total_degree: 200,
        };
        bencher.iter(|| {
            let ag = black_box(&a).mul(black_box(&g), &budget);
            let bg = black_box(&b).mul(black_box(&g), &budget);
            black_box((ag, bg))
        })
    });

    // Benchmark 2: GCD only (with precomputed ag, bg)
    group.bench_function("gcd_only", |bencher| {
        bencher.iter(|| {
            let d = gcd_multivar_layer25(black_box(&ag), black_box(&bg), &gcd_budget);
            black_box(d)
        })
    });

    // Benchmark 3: Full (mul + gcd) - comparable to Symbolica benchmark
    group.bench_function("full", |bencher| {
        let budget = PolyBudget {
            max_terms: 10_000_000,
            max_total_degree: 200,
        };
        bencher.iter(|| {
            let ag = black_box(&a).mul(black_box(&g), &budget).unwrap();
            let bg = black_box(&b).mul(black_box(&g), &budget).unwrap();
            let d = gcd_multivar_layer25(&ag, &bg, &gcd_budget);
            black_box(d)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_mm_gcd);
criterion_main!(benches);
