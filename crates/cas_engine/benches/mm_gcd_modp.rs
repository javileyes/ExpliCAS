//! mm_gcd benchmark using modular polynomial GCD (no BigRational).
//!
//! This benchmark uses the Zippel modular GCD algorithm which avoids
//! the BigRational overhead that makes the regular mm_gcd benchmark slow.
//!
//! Polynomials are constructed directly in Fp without going through CAS.

use cas_engine::gcd_zippel_modp::{budget_for_mm_gcd, gcd_zippel_modp_with_main};
use cas_engine::modp::neg_mod;
use cas_engine::multipoly_modp::{build_linear_pow_direct, MultiPolyModP};
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Prime for modular arithmetic (10^9 + 7)
const P: u64 = 1_000_000_007;

/// Number of variables in mm_gcd benchmark
const NUM_VARS: usize = 7;

/// Build polynomial: (c0 + c1*x1 + c2*x2 + ... + c7*x7)^7 + offset
/// Uses direct multinomial construction (O(3432) terms) instead of polynomial multiplication.
fn build_linear_pow7_fast(coeffs: &[i64; 8], offset: i64, p: u64) -> MultiPolyModP {
    // Normalize coefficients to mod p
    let coeffs_u64: Vec<u64> = coeffs.iter().map(|&c| normalize_mod(c, p)).collect();

    // Build directly using multinomial theorem
    let poly = build_linear_pow_direct(&coeffs_u64, 7, p, NUM_VARS);

    // Add offset
    let offset_u64 = normalize_mod(offset, p);
    poly.add_const(offset_u64)
}

/// Normalize signed integer to [0, p)
fn normalize_mod(x: i64, p: u64) -> u64 {
    if x >= 0 {
        (x as u64) % p
    } else {
        neg_mod(((-x) as u64) % p, p)
    }
}

/// Build the three polynomials a, b, g for mm_gcd using FAST direct construction.
fn build_polys_modp_fast() -> (MultiPolyModP, MultiPolyModP, MultiPolyModP) {
    // a = (1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^7 - 1
    let a = build_linear_pow7_fast(&[1, 3, 5, 7, 9, 11, 13, 15], -1, P);

    // b = (1 - 3*x1 - 5*x2 - 7*x3 + 9*x4 - 11*x5 - 13*x6 + 15*x7)^7 + 1
    let b = build_linear_pow7_fast(&[1, -3, -5, -7, 9, -11, -13, 15], 1, P);

    // g = (1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 - 15*x7)^7 + 3
    let g = build_linear_pow7_fast(&[1, 3, 5, 7, 9, 11, 13, -15], 3, P);

    (a, b, g)
}

fn bench_mm_gcd_modp(c: &mut Criterion) {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║        mm_gcd mod p Benchmark (Zippel GCD)                 ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Build polys once
    println!("Step 1: Building base polynomials (a, b, g) mod p...");
    let start_build = Instant::now();
    let (a, b, g) = build_polys_modp_fast();
    let build_time = start_build.elapsed();

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
        "  ✓ g: {} terms, total_degree {}\n",
        g.num_terms(),
        g.total_degree()
    );
    println!("  ✓ Build time: {:?}", build_time);

    // Compute products
    println!("Step 2: Computing products a*g and b*g...");
    let ag = a.mul(&g);
    let bg = b.mul(&g);

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

    // Test GCD
    println!("Step 3: Testing Zippel GCD...");
    let budget = budget_for_mm_gcd();
    println!(
        "  Budget: max_points={}, max_retries={}, verify_trials={}",
        budget.max_points_per_var, budget.max_retries, budget.verify_trials
    );

    // Force main_var to x7 (index 6) - this is the variable with -15 coefficient in g
    let main_var: usize = 6;
    println!("  Forced main_var: {} (x7)", main_var);

    let gcd_result = gcd_zippel_modp_with_main(&ag, &bg, main_var, &budget);

    let gcd_works = match &gcd_result {
        Some(gcd) => {
            println!(
                "  ✓ gcd computed: {} terms, total_degree {}",
                gcd.num_terms(),
                gcd.total_degree()
            );

            // Compare with g (both made monic)
            let mut g_monic = g.clone();
            g_monic.make_monic();

            let mut gcd_monic = gcd.clone();
            gcd_monic.make_monic();

            if gcd_monic.total_degree() == g_monic.total_degree()
                && gcd_monic.num_terms() == g_monic.num_terms()
            {
                println!("  ✓ GCD matches g (same degree and term count)\n");
                true
            } else {
                println!("  ⚠ GCD degree/terms differ from g\n");
                println!(
                    "    gcd: deg={}, terms={}",
                    gcd_monic.total_degree(),
                    gcd_monic.num_terms()
                );
                println!(
                    "    g:   deg={}, terms={}\n",
                    g_monic.total_degree(),
                    g_monic.num_terms()
                );
                true // Still run benchmark
            }
        }
        None => {
            println!("  ✗ GCD computation returned None\n");
            false
        }
    };

    // Configure benchmark
    let mut group = c.benchmark_group("mm_gcd_modp");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));

    // Benchmark 1: Multiplication only
    println!("Running benchmark: mul_only (a*g + b*g) mod p...");
    group.bench_function("mul_only", |bencher| {
        bencher.iter(|| {
            let ag = black_box(&a).mul(black_box(&g));
            let bg = black_box(&b).mul(black_box(&g));
            black_box((ag, bg))
        })
    });

    // Benchmark 2 & 3: Only if GCD works
    if gcd_works {
        println!("Running benchmark: gcd_only mod p...");
        group.bench_function("gcd_only", |bencher| {
            bencher.iter(|| {
                let gcd =
                    gcd_zippel_modp_with_main(black_box(&ag), black_box(&bg), main_var, &budget);
                black_box(gcd)
            })
        });

        println!("Running benchmark: full (mul + gcd) mod p...");
        group.bench_function("full", |bencher| {
            bencher.iter(|| {
                let ag = black_box(&a).mul(black_box(&g));
                let bg = black_box(&b).mul(black_box(&g));
                let gcd = gcd_zippel_modp_with_main(&ag, &bg, main_var, &budget);
                black_box(gcd)
            })
        });
    } else {
        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║  GCD benchmarks SKIPPED: Zippel GCD failed                 ║");
        println!("╚════════════════════════════════════════════════════════════╝\n");
    }

    group.finish();

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    Benchmark Complete                      ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

criterion_group!(benches, bench_mm_gcd_modp);
criterion_main!(benches);
