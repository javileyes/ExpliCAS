use std::fs;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::Path;

use cas_ast::Context;
use cas_parser::parse;
use cas_session_core::context_snapshot::ContextSnapshot;
use cas_session_core::snapshot_io::{
    encode_bincode, load_bincode, save_bincode_atomic, save_bincode_bytes_atomic,
};
use cas_session_core::store_snapshot::{
    CacheConfigSnapshot, EntryKindSnapshot, EntrySnapshot, SessionStoreSnapshot,
    SimplifiedCacheSnapshot,
};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use serde::{Deserialize, Serialize};
use tempfile::tempdir;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotFixture {
    context: ContextSnapshot,
    session: SessionStoreSnapshot<u64>,
}

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(25);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(2));
}

fn build_fixture(multiplier: usize) -> SnapshotFixture {
    let seeds = [
        "x + 1",
        "2 * 3 + 4",
        "sqrt(12*x^3)",
        "((5*x + 8)^2)^(1/2)",
        "(2*x + 2*y)/(4*x + 4*y)",
        "((x+y)*(a+b))/((x+y)*(c+d))",
        "sin(2*x + 1)^2 + cos(1 + 2*x)^2",
        "log(x^2, x^6)",
    ];

    let mut ctx = Context::new();
    let mut expr_ids = Vec::new();
    let mut texts = Vec::new();
    for n in 0..multiplier {
        for seed in seeds {
            let expr_id = parse(seed, &mut ctx).expect("parse fixture expr");
            expr_ids.push(expr_id);
            texts.push(format!("{seed} // {n}"));
        }
    }

    let entries = expr_ids
        .iter()
        .zip(texts)
        .enumerate()
        .map(|(i, (expr_id, raw_text))| EntrySnapshot {
            id: i as u64 + 1,
            raw_text,
            kind: EntryKindSnapshot::Expr(expr_id.index() as u32),
            simplified: (i % 2 == 0).then(|| SimplifiedCacheSnapshot {
                key: i as u64 + 1,
                expr: expr_id.index() as u32,
            }),
        })
        .collect::<Vec<_>>();

    let cache_order = (1..=entries.len() as u64).rev().collect::<Vec<_>>();

    SnapshotFixture {
        context: ContextSnapshot::from_context(&ctx),
        session: SessionStoreSnapshot {
            next_id: entries.len() as u64 + 1,
            entries,
            cache_order,
            cache_config: CacheConfigSnapshot {
                max_cached_entries: 256,
                max_cached_steps: 10_000,
                light_cache_threshold: Some(8),
            },
            cached_steps_count: multiplier * 64,
        },
    }
}

fn save_bytes_direct(bytes: &[u8], path: &Path) {
    let file = fs::File::create(path).expect("create direct snapshot");
    let mut writer = BufWriter::with_capacity(
        cas_session_core::snapshot_io::SNAPSHOT_IO_BUFFER_CAPACITY,
        file,
    );
    writer.write_all(bytes).expect("write direct snapshot");
    writer.flush().expect("flush direct snapshot");
}

fn save_bytes_direct_and_sync(bytes: &[u8], path: &Path) {
    let file = fs::File::create(path).expect("create synced snapshot");
    let mut writer = BufWriter::with_capacity(
        cas_session_core::snapshot_io::SNAPSHOT_IO_BUFFER_CAPACITY,
        file,
    );
    writer.write_all(bytes).expect("write synced snapshot");
    writer.flush().expect("flush synced snapshot");
    let file = writer.into_inner().expect("extract synced snapshot file");
    file.sync_all().expect("sync snapshot file");
}

fn save_bytes_to_tmp_only(bytes: &[u8], path: &Path) {
    let tmp = cas_session_core::snapshot_io::tmp_path(path);
    save_bytes_direct(bytes, &tmp);
}

fn save_bytes_atomic_with_synced_tmp(bytes: &[u8], path: &Path) {
    let tmp = cas_session_core::snapshot_io::tmp_path(path);
    save_bytes_direct_and_sync(bytes, &tmp);
    fs::rename(&tmp, path).expect("rename synced tmp snapshot");
}

fn bench_snapshot_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_io");
    configure_group(&mut group);

    for (name, multiplier) in [("medium", 8usize), ("large", 32usize)] {
        let fixture = build_fixture(multiplier);

        group.bench_with_input(
            BenchmarkId::new("serialize", name),
            &fixture,
            |b, fixture| b.iter(|| black_box(encode_bincode(fixture).unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("save_prebuilt", name),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || {
                        let tmp = tempdir().expect("tempdir failed");
                        let path = tmp.path().join("session.bin");
                        let bytes = encode_bincode(fixture).expect("encode snapshot");
                        (tmp, path, bytes)
                    },
                    |(_tmp, path, bytes)| {
                        save_bincode_bytes_atomic(&bytes, &path).unwrap();
                        black_box(())
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("save_prebuilt_tmp_file_synced", name),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || {
                        let tmp = tempdir().expect("tempdir failed");
                        let path = tmp.path().join("session.bin");
                        let bytes = encode_bincode(fixture).expect("encode snapshot");
                        (tmp, path, bytes)
                    },
                    |(_tmp, path, bytes)| {
                        save_bytes_atomic_with_synced_tmp(&bytes, &path);
                        black_box(())
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("save_direct", name),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || {
                        let tmp = tempdir().expect("tempdir failed");
                        let path = tmp.path().join("session.bin");
                        let bytes = encode_bincode(fixture).expect("encode snapshot");
                        (tmp, path, bytes)
                    },
                    |(_tmp, path, bytes)| {
                        save_bytes_direct(&bytes, &path);
                        black_box(())
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("save_direct_overwrite_existing", name),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || {
                        let tmp = tempdir().expect("tempdir failed");
                        let path = tmp.path().join("session.bin");
                        let bytes = encode_bincode(fixture).expect("encode snapshot");
                        save_bytes_direct(&bytes, &path);
                        (tmp, path, bytes)
                    },
                    |(_tmp, path, bytes)| {
                        save_bytes_direct(&bytes, &path);
                        black_box(())
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("save_tmp_only", name),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || {
                        let tmp = tempdir().expect("tempdir failed");
                        let path = tmp.path().join("session.bin");
                        let bytes = encode_bincode(fixture).expect("encode snapshot");
                        (tmp, path, bytes)
                    },
                    |(_tmp, path, bytes)| {
                        save_bytes_to_tmp_only(&bytes, &path);
                        black_box(())
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rename_only", name),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || {
                        let tmp = tempdir().expect("tempdir failed");
                        let path = tmp.path().join("session.bin");
                        let tmp_path = cas_session_core::snapshot_io::tmp_path(&path);
                        let bytes = encode_bincode(fixture).expect("encode snapshot");
                        save_bytes_direct(&bytes, &tmp_path);
                        (tmp, tmp_path, path)
                    },
                    |(_tmp, tmp_path, path)| {
                        fs::rename(&tmp_path, &path).expect("rename atomic snapshot");
                        black_box(())
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("save_prebuilt_overwrite_existing", name),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || {
                        let tmp = tempdir().expect("tempdir failed");
                        let path = tmp.path().join("session.bin");
                        let bytes = encode_bincode(fixture).expect("encode snapshot");
                        save_bytes_direct(&bytes, &path);
                        (tmp, path, bytes)
                    },
                    |(_tmp, path, bytes)| {
                        save_bincode_bytes_atomic(&bytes, &path).unwrap();
                        black_box(())
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("save_prebuilt_tmp_file_synced_overwrite_existing", name),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || {
                        let tmp = tempdir().expect("tempdir failed");
                        let path = tmp.path().join("session.bin");
                        let bytes = encode_bincode(fixture).expect("encode snapshot");
                        save_bytes_direct(&bytes, &path);
                        (tmp, path, bytes)
                    },
                    |(_tmp, path, bytes)| {
                        save_bytes_atomic_with_synced_tmp(&bytes, &path);
                        black_box(())
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rename_only_overwrite_existing", name),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || {
                        let tmp = tempdir().expect("tempdir failed");
                        let path = tmp.path().join("session.bin");
                        let tmp_path = cas_session_core::snapshot_io::tmp_path(&path);
                        let bytes = encode_bincode(fixture).expect("encode snapshot");
                        save_bytes_direct(&bytes, &path);
                        save_bytes_direct(&bytes, &tmp_path);
                        (tmp, tmp_path, path)
                    },
                    |(_tmp, tmp_path, path)| {
                        fs::rename(&tmp_path, &path).expect("rename over existing snapshot");
                        black_box(())
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("remove_existing_only", name),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || {
                        let tmp = tempdir().expect("tempdir failed");
                        let path = tmp.path().join("session.bin");
                        let bytes = encode_bincode(fixture).expect("encode snapshot");
                        save_bytes_direct(&bytes, &path);
                        (tmp, path)
                    },
                    |(_tmp, path)| {
                        fs::remove_file(&path).expect("remove existing snapshot");
                        black_box(())
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(BenchmarkId::new("save", name), &fixture, |b, fixture| {
            b.iter_batched(
                || {
                    let tmp = tempdir().expect("tempdir failed");
                    let path = tmp.path().join("session.bin");
                    (tmp, path, fixture.clone())
                },
                |(_tmp, path, fixture)| {
                    save_bincode_atomic(&fixture, &path).unwrap();
                    black_box(())
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("load", name), &fixture, |b, fixture| {
            b.iter_batched(
                || {
                    let tmp = tempdir().expect("tempdir failed");
                    let path = tmp.path().join("session.bin");
                    save_bincode_atomic(fixture, &path).expect("seed snapshot");
                    (tmp, path)
                },
                |(_tmp, path)| black_box(load_bincode::<SnapshotFixture>(&path).unwrap()),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_snapshot_io);
criterion_main!(benches);
