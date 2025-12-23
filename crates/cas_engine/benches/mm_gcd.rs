//! mm_gcd benchmark: Famous polynomial GCD benchmark from benruijl.
//!
//! Benchmark: <https://gist.github.com/benruijl/3c53b1b0aea88b978ae609e73693fdbc>
//! Tests: (a·g, b·g) + gcd where a, b, g are 7-variable polynomials^7
//!
//! This measures raw MultiPoly multiplication and GCD performance
//! without going through the simplifier pipeline.
//!
//! Note: Current Layer 2.5 tensor-grid GCD is limited to ~4 variables.
//! For 7-var mm_gcd, a Zippel modular GCD would be needed.

use cas_ast::Context;
use cas_engine::multipoly::{
    gcd_multivar_layer25, multipoly_from_expr, Layer25Budget, MultiPoly, PolyBudget,
};
use cas_parser::parse;
use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

/// Budget optimized for mm_gcd benchmark (larger than engine defaults)
fn gcd_budget_mm() -> Layer25Budget {
    Layer25Budget {
        max_vars: 8,       // Allow 7 variables
        max_samples: 1024, // More samples for interpolation
        max_param_deg: 10, // Higher degree params (7^7)
        max_gcd_deg: 20,   // Higher result degree
    }
}

/// Build the three polynomials a, b, g as MultiPoly.
/// Uses parser + multipoly_from_expr which handles Pow(linear, 7) via pow_poly.
fn build_polys() -> (MultiPoly, MultiPoly, MultiPoly) {
    let mut ctx = Context::new();
    let budget = PolyBudget {
        max_terms: 1_000_000,
        max_total_degree: 100,
        max_pow_exp: 10, // Allow large exponents for benchmarks
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
fn build_products(a: &MultiPoly, b: &MultiPoly, g: &MultiPoly) -> Option<(MultiPoly, MultiPoly)> {
    let budget = PolyBudget {
        max_terms: 10_000_000,
        max_total_degree: 200,
        max_pow_exp: 10,
    };
    let ag = a.mul_fast(g, &budget).ok()?;
    let bg = b.mul_fast(g, &budget).ok()?;
    Some((ag, bg))
}

fn bench_mm_gcd(c: &mut Criterion) {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║              mm_gcd Benchmark Diagnostic                  ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Build polys once (expensive but outside timing)
    println!("Step 1: Building base polynomials (a, b, g)...");
    let (a, b, g) = build_polys();

    println!(
        "  ✓ a: {} terms, total_degree {}",
        a.num_terms(),
        a.total_degree()
    );
    println!(
        "  ✓ b: {} terms, total_degree {}",
        b.num_terms(),
        b.total_degree()
    );
    println!(
        "  ✓ g: {} terms, total_degree {}",
        g.num_terms(),
        g.total_degree()
    );
    println!("  Variables: {:?}\n", a.vars);

    // Build products for GCD-only bench
    println!("Step 2: Computing products a*g and b*g...");
    let products = build_products(&a, &b, &g);

    let (ag, bg) = match products {
        Some((ag, bg)) => {
            println!(
                "  ✓ a*g: {} terms, total_degree {}",
                ag.num_terms(),
                ag.total_degree()
            );
            println!(
                "  ✓ b*g: {} terms, total_degree {}\n",
                bg.num_terms(),
                bg.total_degree()
            );
            (ag, bg)
        }
        None => {
            println!("  ✗ FAILED: Product computation exceeded budget");
            println!("\n╔════════════════════════════════════════════════════════════╗");
            println!("║  BENCHMARK SKIPPED: Cannot compute products                ║");
            println!("╚════════════════════════════════════════════════════════════╝\n");
            return;
        }
    };

    // Try GCD with expanded budget
    println!("Step 3: Testing GCD computation...");
    let gcd_budget = gcd_budget_mm();
    println!(
        "  Budget: max_vars={}, max_samples={}, max_param_deg={}, max_gcd_deg={}",
        gcd_budget.max_vars,
        gcd_budget.max_samples,
        gcd_budget.max_param_deg,
        gcd_budget.max_gcd_deg
    );

    let gcd_result = gcd_multivar_layer25(&ag, &bg, &gcd_budget);

    let gcd_works = match &gcd_result {
        Some(d) => {
            println!(
                "  ✓ gcd(ag,bg): {} terms, total_degree {}",
                d.num_terms(),
                d.total_degree()
            );

            // Verify: g divides d (since gcd should be scalar * g)
            if let Some(q) = d.div_exact(&g) {
                if q.is_constant() {
                    println!("  ✓ VERIFIED: g divides gcd, quotient is constant\n");
                    true
                } else {
                    println!(
                        "  ⚠ WARNING: g divides gcd but quotient has {} terms (expected 1)\n",
                        q.num_terms()
                    );
                    true
                }
            } else {
                println!("  ✗ WARNING: g does not divide gcd exactly\n");
                false
            }
        }
        None => {
            println!("  ✗ FAILED: GCD computation returned None");
            println!("\n  Likely causes:");
            println!("    - Tensor-grid interpolation explodes for 7 variables");
            println!("    - Degree bounds exceeded during reconstruction");
            println!("    - Insufficient samples for accurate interpolation");
            println!("\n  This is expected: mm_gcd requires Zippel modular GCD,");
            println!("  not tensor-grid interpolation. Layer 2.5 is designed for");
            println!("  typical simplification cases (2-4 variables).\n");
            false
        }
    };

    // Configure for slow benchmarks
    let mut group = c.benchmark_group("mm_gcd");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));

    // Benchmark 1: Multiplication only (always runs)
    println!("Running benchmark: mul_only (a*g + b*g)...");
    group.bench_function("mul_only", |bencher| {
        let budget = PolyBudget {
            max_terms: 10_000_000,
            max_total_degree: 200,
            max_pow_exp: 10,
        };
        bencher.iter(|| {
            let ag = std::hint::black_box(&a).mul_fast(std::hint::black_box(&g), &budget);
            let bg = std::hint::black_box(&b).mul_fast(std::hint::black_box(&g), &budget);
            std::hint::black_box((ag, bg))
        })
    });

    // Benchmark 2 & 3: Only if GCD works
    if gcd_works {
        println!("Running benchmark: gcd_only...");
        group.bench_function("gcd_only", |bencher| {
            bencher.iter(|| {
                let d = gcd_multivar_layer25(
                    std::hint::black_box(&ag),
                    std::hint::black_box(&bg),
                    &gcd_budget,
                );
                std::hint::black_box(d)
            })
        });

        println!("Running benchmark: full (mul + gcd)...");
        group.bench_function("full", |bencher| {
            let budget = PolyBudget {
                max_terms: 10_000_000,
                max_total_degree: 200,
                max_pow_exp: 10,
            };
            bencher.iter(|| {
                let ag = std::hint::black_box(&a)
                    .mul_fast(std::hint::black_box(&g), &budget)
                    .unwrap();
                let bg = std::hint::black_box(&b)
                    .mul_fast(std::hint::black_box(&g), &budget)
                    .unwrap();
                let d = gcd_multivar_layer25(&ag, &bg, &gcd_budget);
                std::hint::black_box(d)
            })
        });
    } else {
        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║  GCD benchmarks SKIPPED: Layer 2.5 cannot handle mm_gcd    ║");
        println!("║  Only mul_only benchmark will run                          ║");
        println!("╚════════════════════════════════════════════════════════════╝\n");
    }

    group.finish();

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    Benchmark Complete                      ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

criterion_group!(benches, bench_mm_gcd);
criterion_main!(benches);
