use cas_api_models::{
    BudgetWireInfo, DomainWire, EngineWireWarning, ErrorWireOutput, EvalOutputBuild,
    EvalWireOutput, ExprStatsWire, OptionsWire, RequiredConditionWire, SemanticsWire, StepWire,
    TimingsWire, WarningWire,
};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use serde_json::json;
use std::hint::black_box;

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(25);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(2));
}

fn build_eval_wire(light: bool) -> EvalWireOutput {
    let steps = if light {
        Vec::new()
    } else {
        vec![
            StepWire {
                index: 1,
                rule: "Simplify Nested Fraction".into(),
                rule_latex: "\\text{Simplify Nested Fraction}".into(),
                before: "(2*x + 2*y)/(4*x + 4*y)".into(),
                after: "(x + y)/(2*(x + y))".into(),
                before_latex: "\\frac{2x+2y}{4x+4y}".into(),
                after_latex: "\\frac{x+y}{2(x+y)}".into(),
                substeps: Vec::new(),
            },
            StepWire {
                index: 2,
                rule: "Simplify Nested Fraction".into(),
                rule_latex: "\\text{Simplify Nested Fraction}".into(),
                before: "(x + y)/(2*(x + y))".into(),
                after: "1/2".into(),
                before_latex: "\\frac{x+y}{2(x+y)}".into(),
                after_latex: "\\frac{1}{2}".into(),
                substeps: Vec::new(),
            },
        ]
    };

    let warnings = if light {
        Vec::new()
    } else {
        vec![WarningWire {
            rule: "Log Power Base".into(),
            assumption: "x > 0".into(),
        }]
    };

    let required_conditions = if light {
        Vec::new()
    } else {
        vec![RequiredConditionWire {
            kind: "NonZero".into(),
            expr_display: "4*x + 4*y != 0".into(),
            expr_canonical: "4*x + 4*y != 0".into(),
        }]
    };

    EvalWireOutput::from_build(EvalOutputBuild {
        input: if light {
            "x + 1"
        } else {
            "(2*x + 2*y)/(4*x + 4*y)"
        },
        input_latex: None,
        result: if light { "x + 1".into() } else { "1/2".into() },
        result_truncated: false,
        result_chars: if light { 5 } else { 3 },
        result_latex: None,
        steps_mode: if light { "off" } else { "on" },
        steps_count: steps.len(),
        steps,
        solve_steps: Vec::new(),
        warnings,
        required_conditions,
        required_display: if light {
            Vec::new()
        } else {
            vec!["4*x + 4*y != 0".into()]
        },
        budget_preset: "standard",
        strict: false,
        domain: "generic",
        stats: ExprStatsWire {
            node_count: if light { 3 } else { 11 },
            depth: if light { 2 } else { 5 },
            term_count: None,
        },
        hash: Some(if light {
            "light-hash".into()
        } else {
            "gcd-hash".into()
        }),
        timings_us: TimingsWire {
            parse_us: 4,
            simplify_us: if light { 8 } else { 22 },
            total_us: if light { 12 } else { 26 },
        },
        context_mode: "standard",
        branch_mode: "strict",
        expand_policy: "off",
        complex_mode: "auto",
        const_fold: "off",
        value_domain: "real",
        complex_branch: "principal",
        inv_trig: "strict",
        assume_scope: "real",
        wire: if light {
            None
        } else {
            Some(json!({ "kind": "eval", "steps_compact": true }))
        },
    })
}

fn build_error_wire(parse_error: bool) -> ErrorWireOutput {
    if parse_error {
        ErrorWireOutput::from_eval_error_message("Parse error: unexpected token", "x+")
    } else {
        ErrorWireOutput::from_eval_error_message("Evaluation failed", "(x+y)/(x+y)")
    }
}

fn bench_frontend_wire(c: &mut Criterion) {
    let mut group = c.benchmark_group("frontend_wire");
    configure_group(&mut group);

    for (name, light) in [("light_success", true), ("heavy_success", false)] {
        group.bench_with_input(BenchmarkId::new("build", name), &light, |b, light| {
            b.iter(|| black_box(build_eval_wire(*light)))
        });

        let output = build_eval_wire(light);
        group.bench_with_input(
            BenchmarkId::new("serialize_json", name),
            &output,
            |b, output| b.iter(|| black_box(output.to_json())),
        );
        group.bench_with_input(
            BenchmarkId::new("serialize_pretty", name),
            &output,
            |b, output| b.iter(|| black_box(output.to_json_pretty())),
        );
    }

    for (name, parse_error) in [("parse_error", true), ("runtime_error", false)] {
        group.bench_with_input(
            BenchmarkId::new("error_build", name),
            &parse_error,
            |b, parse_error| b.iter(|| black_box(build_error_wire(*parse_error))),
        );

        group.bench_with_input(
            BenchmarkId::new("error_pretty", name),
            &parse_error,
            |b, parse_error| {
                b.iter_batched(
                    || build_error_wire(*parse_error),
                    |error| black_box(error.to_json_pretty()),
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.bench_function("metadata_bundle", |b| {
        b.iter(|| {
            black_box((
                BudgetWireInfo::new("standard", false),
                DomainWire::from_mode("generic"),
                OptionsWire::from_eval_axes(
                    "standard", "strict", "off", "auto", "off", "generic", "off",
                ),
                SemanticsWire::from_eval_axes("generic", "real", "principal", "strict", "real"),
                EngineWireWarning {
                    kind: "DomainWarning".into(),
                    message: "x > 0".into(),
                },
            ))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_frontend_wire);
criterion_main!(benches);
