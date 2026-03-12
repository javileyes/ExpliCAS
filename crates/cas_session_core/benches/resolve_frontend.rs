use std::hint::black_box;
use std::time::Duration;

use cas_ast::{Context, Expr, ExprId};
use cas_session_core::env::Environment;
use cas_session_core::resolve::{
    resolve_all_with_mode_lookup_and_env, resolve_session_refs_with_lookup, rewrite_session_refs,
    ModeEntry, ModeResolveConfig,
};
use cas_session_core::types::{EntryId, EntryKind, RefMode};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(25);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
}

#[derive(Clone)]
struct ResolveLookupFixture {
    ctx: Context,
    expr: ExprId,
    entries: Vec<(EntryId, EntryKind)>,
}

#[derive(Clone)]
struct ResolveModeFixture {
    ctx: Context,
    expr: ExprId,
    env: Environment,
    entries: Vec<(EntryId, ModeEntry<u8, ()>)>,
}

fn make_explicit_add_fixture() -> ResolveLookupFixture {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let x_plus_one = ctx.add(Expr::Add(x, one));
    let ref1 = ctx.add(Expr::SessionRef(1));
    let expr = ctx.add(Expr::Add(ref1, one));
    ResolveLookupFixture {
        ctx,
        expr,
        entries: vec![(1, EntryKind::Expr(x_plus_one))],
    }
}

fn make_legacy_var_fixture() -> ResolveLookupFixture {
    let mut ctx = Context::new();
    let legacy = ctx.var("#2");
    let expr = ctx.add(Expr::Neg(legacy));
    let five = ctx.num(5);
    ResolveLookupFixture {
        ctx,
        expr,
        entries: vec![(2, EntryKind::Expr(five))],
    }
}

fn make_deep_chain_fixture() -> ResolveLookupFixture {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let terminal = ctx.add(Expr::Add(x, one));
    let ref2 = ctx.add(Expr::SessionRef(2));
    let ref3 = ctx.add(Expr::SessionRef(3));
    let expr = ctx.add(Expr::SessionRef(1));
    ResolveLookupFixture {
        ctx,
        expr,
        entries: vec![
            (1, EntryKind::Expr(ref2)),
            (2, EntryKind::Expr(ref3)),
            (3, EntryKind::Expr(terminal)),
        ],
    }
}

fn make_mode_env_fixture() -> ResolveModeFixture {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let five = ctx.num(5);
    let x_plus_one = ctx.add(Expr::Add(x, one));
    let ref1 = ctx.add(Expr::SessionRef(1));
    let expr = ctx.add(Expr::Mul(ref1, x));

    let mut env = Environment::new();
    env.set("x".to_string(), five);

    ResolveModeFixture {
        ctx,
        expr,
        env,
        entries: vec![(
            1,
            ModeEntry {
                kind: EntryKind::Expr(x_plus_one),
                requires: vec![],
                cache: None,
            },
        )],
    }
}

fn bench_resolve_frontend(c: &mut Criterion) {
    let mut group = c.benchmark_group("resolve_frontend");
    configure_group(&mut group);

    let explicit_add = make_explicit_add_fixture();
    group.bench_function("rewrite/explicit_add", |b| {
        b.iter_batched(
            || explicit_add.clone(),
            |fixture| {
                let ResolveLookupFixture {
                    mut ctx,
                    expr,
                    entries,
                } = fixture;
                black_box(
                    rewrite_session_refs(&mut ctx, expr, &mut |_ctx, _node, id| {
                        entries
                            .iter()
                            .find(|(entry_id, _)| *entry_id == id)
                            .map(|(_, kind)| kind.clone())
                            .and_then(|kind| match kind {
                                EntryKind::Expr(expr) => Some(expr),
                                EntryKind::Eq { .. } => None,
                            })
                            .ok_or(())
                    })
                    .unwrap(),
                )
            },
            BatchSize::SmallInput,
        )
    });

    for (name, fixture) in [
        ("explicit_add", make_explicit_add_fixture()),
        ("legacy_var", make_legacy_var_fixture()),
        ("deep_chain", make_deep_chain_fixture()),
    ] {
        group.bench_with_input(
            BenchmarkId::new("resolve_lookup", name),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || fixture.clone(),
                    |fixture| {
                        let ResolveLookupFixture {
                            mut ctx,
                            expr,
                            entries,
                        } = fixture;
                        let mut lookup = |id| {
                            entries
                                .iter()
                                .find(|(entry_id, _)| *entry_id == id)
                                .map(|(_, kind)| kind.clone())
                        };
                        black_box(
                            resolve_session_refs_with_lookup(&mut ctx, expr, &mut lookup).unwrap(),
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    let mode_fixture = make_mode_env_fixture();
    group.bench_with_input(
        BenchmarkId::new("resolve_mode", "raw_env"),
        &mode_fixture,
        |b, fixture| {
            b.iter_batched(
                || fixture.clone(),
                |fixture| {
                    let ResolveModeFixture {
                        mut ctx,
                        expr,
                        env,
                        entries,
                    } = fixture;
                    let mut lookup = |id| {
                        entries
                            .iter()
                            .find(|(entry_id, _)| *entry_id == id)
                            .map(|(_, entry)| entry.clone())
                    };
                    let mut same_requirement = |_lhs: &(), _rhs: &()| true;
                    let mut mark_session_propagated = |_item: &mut ()| {};
                    black_box(
                        resolve_all_with_mode_lookup_and_env(
                            &mut ctx,
                            expr,
                            ModeResolveConfig {
                                mode: RefMode::Raw,
                                cache_key: &0u8,
                                env: &env,
                            },
                            &mut lookup,
                            &mut same_requirement,
                            &mut mark_session_propagated,
                        )
                        .unwrap(),
                    )
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.finish();
}

criterion_group!(benches, bench_resolve_frontend);
criterion_main!(benches);
