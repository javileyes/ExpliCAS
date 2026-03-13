mod common;

use std::hint::black_box;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

mod repl {
    pub use cas_api_models::wire::{WireKind, WireMsg, WireReply};

    #[allow(dead_code)]
    pub mod output {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/repl/output/messages.rs"
        ));
    }

    pub mod wire {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/repl/wire/convert.rs"
        ));
    }
}

use repl::output::{ReplMsg, ReplReply};
use repl::wire::wire_reply_from_repl;

fn sample_replies() -> [(&'static str, ReplReply); 4] {
    [
        (
            "light/output_only",
            vec![ReplMsg::Output("x + 1".to_string())],
        ),
        (
            "light/mixed_messages",
            vec![
                ReplMsg::Info("Using generic domain".to_string()),
                ReplMsg::Output("1/2".to_string()),
                ReplMsg::Steps("2 simplification step(s)".to_string()),
                ReplMsg::Warn("Requires: x != 0".to_string()),
            ],
        ),
        (
            "io/file_actions",
            vec![
                ReplMsg::WriteFile {
                    path: PathBuf::from("/tmp/expli/session/report.html"),
                    contents: "<html>ok</html>".to_string(),
                },
                ReplMsg::OpenFile {
                    path: PathBuf::from("/tmp/expli/session/report.html"),
                },
            ],
        ),
        (
            "io/full_reply",
            vec![
                ReplMsg::Info("Loaded session".to_string()),
                ReplMsg::Output("x = 5".to_string()),
                ReplMsg::Steps("1 step".to_string()),
                ReplMsg::WriteFile {
                    path: PathBuf::from("/tmp/expli/session/plot.svg"),
                    contents: "<svg />".to_string(),
                },
                ReplMsg::OpenFile {
                    path: PathBuf::from("/tmp/expli/session/plot.svg"),
                },
            ],
        ),
    ]
}

fn bench_frontend_repl_wire(c: &mut Criterion) {
    let cases = sample_replies();

    let mut group = c.benchmark_group("frontend_repl_wire");
    common::configure_standard_group(&mut group);

    for (name, reply) in &cases {
        group.bench_with_input(BenchmarkId::new("convert", name), reply, |b, reply| {
            b.iter(|| black_box(wire_reply_from_repl(black_box(reply))));
        });

        group.bench_with_input(
            BenchmarkId::new("convert_serialize", name),
            reply,
            |b, reply| {
                b.iter(|| {
                    let wire = wire_reply_from_repl(black_box(reply));
                    black_box(serde_json::to_string(&wire).expect("wire serialization failed"))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_frontend_repl_wire);
criterion_main!(benches);
