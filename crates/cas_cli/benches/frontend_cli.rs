mod common;

use std::hint::black_box;

use clap::Parser;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

#[path = "../src/commands/envelope.rs"]
pub mod envelope;

pub mod commands {
    pub use super::envelope;
}

#[path = "../src/cli_args.rs"]
mod cli_args;

use cli_args::Cli;

fn bench_frontend_cli(c: &mut Criterion) {
    let cases: [(&str, &[&str]); 5] = [
        ("eval/text/light", &["expli", "eval", "x + 1"]),
        (
            "eval/json/gcd",
            &[
                "expli",
                "eval",
                "(2*x + 2*y)/(4*x + 4*y)",
                "--format",
                "json",
                "--domain",
                "generic",
                "--steps",
                "off",
            ],
        ),
        (
            "limit/json",
            &[
                "expli",
                "limit",
                "(x^2+1)/(2*x^2-3)",
                "--var",
                "x",
                "--to",
                "infinity",
                "--presimplify",
                "off",
                "--format",
                "json",
            ],
        ),
        (
            "substitute/json",
            &[
                "expli",
                "substitute",
                "x^4 + x^2 + 1",
                "--target",
                "x^2",
                "--with",
                "y",
                "--format",
                "json",
            ],
        ),
        (
            "envelope/basic",
            &[
                "expli",
                "envelope",
                "x/x",
                "--domain",
                "generic",
                "--value-domain",
                "real",
            ],
        ),
    ];

    let mut group = c.benchmark_group("frontend_cli");
    common::configure_standard_group(&mut group);

    for (name, argv) in cases {
        group.bench_with_input(BenchmarkId::new("parse", name), &argv, |b, argv| {
            b.iter(|| black_box(Cli::try_parse_from(*argv).expect("cli parse failed")));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_frontend_cli);
criterion_main!(benches);
