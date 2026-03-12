mod common;

use std::hint::black_box;

use clap::Parser;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

#[path = "../src/commands/dispatch.rs"]
pub mod dispatch;
#[path = "../src/commands/envelope.rs"]
pub mod envelope;
#[path = "../src/commands/eval.rs"]
pub mod eval;
#[path = "../src/commands/eval_text.rs"]
pub mod eval_text;
#[path = "../src/commands/limit.rs"]
pub mod limit;
#[path = "../src/commands/output.rs"]
pub mod output;
#[path = "../src/commands/substitute.rs"]
pub mod substitute;

pub mod commands {
    pub use super::dispatch;
    pub use super::envelope;
    pub use super::eval;
    pub use super::eval_text;
    pub use super::limit;
    pub use super::output;
    pub use super::substitute;
}

#[path = "../src/cli_args.rs"]
mod cli_args;

pub use cli_args::*;

use cli_args::Cli as BenchCli;

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
            b.iter(|| black_box(BenchCli::try_parse_from(*argv).expect("cli parse failed")));
        });
        group.bench_with_input(BenchmarkId::new("parse_render", name), &argv, |b, argv| {
            b.iter(|| {
                let cli = BenchCli::try_parse_from(*argv).expect("cli parse failed");
                let command = cli.command.expect("command");
                black_box(
                    commands::dispatch::render_command(command)
                        .expect("render failed")
                        .map_or_else(String::new, |output| output.stdout),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_frontend_cli);
criterion_main!(benches);
