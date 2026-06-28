use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalStepsMode,
    EvalValueDomain,
};
use cas_engine::{
    clear_orchestrator_shortcut_profile, orchestrator_shortcut_profile_report,
    orchestrator_shortcut_profiling_enabled,
};
use cas_session::eval::{evaluate_eval_command_with_session, EvalCommandConfig};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

const ENGINE_HEAVY_RERUN_LIMIT: usize = 5;
const ENGINE_HEAVY_RERUN_COUNT: usize = 3;

#[derive(Debug, Clone)]
struct CorpusCase {
    case_number: usize,
    expression: String,
    expected_result: String,
    composition: String,
    source_expr_1: String,
    source_expr_2: String,
}

#[derive(Debug, Clone)]
struct FailureRecord {
    case_number: usize,
    window_label: String,
    expression: String,
    expected_result: String,
    actual_result: String,
    ok: bool,
    composition: String,
    source_expr_1: String,
    source_expr_2: String,
    error_kind: String,
    error_message: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct CompositionSummary {
    total: usize,
    passed: usize,
    failed: usize,
    wall_elapsed_seconds: f64,
    wire_elapsed_seconds: f64,
    parse_elapsed_seconds: f64,
    simplify_elapsed_seconds: f64,
}

#[derive(Debug, Clone)]
struct WindowSpec {
    composition_filter: Option<String>,
    offset: usize,
    limit: usize,
}

impl WindowSpec {
    fn label(&self) -> String {
        let composition = self.composition_filter.as_deref().unwrap_or("all");
        format!("{composition}@{}+{}", self.offset, self.limit)
    }
}

#[derive(Debug, Clone)]
struct SelectedCase {
    case: CorpusCase,
    window_label: String,
}

#[derive(Debug, Clone)]
struct SlowCaseRecord {
    case_number: usize,
    window_label: String,
    composition: String,
    wall_elapsed_seconds: f64,
    wire_elapsed_seconds: f64,
    parse_elapsed_seconds: f64,
    simplify_elapsed_seconds: f64,
    expression: String,
}

#[derive(Debug, Clone)]
struct SteadyStateEngineHeavyRecord {
    case_number: usize,
    window_label: String,
    composition: String,
    runs: usize,
    median_wall_elapsed_seconds: f64,
    median_wire_elapsed_seconds: f64,
    median_parse_elapsed_seconds: f64,
    median_simplify_elapsed_seconds: f64,
    expression: String,
}

#[derive(Debug, Clone, Copy, Default)]
struct CaseTimingRecord {
    wire_elapsed_seconds: f64,
    parse_elapsed_seconds: f64,
    simplify_elapsed_seconds: f64,
}

#[derive(Debug)]
struct CaseEvaluation {
    failure: Option<FailureRecord>,
    timings: CaseTimingRecord,
}

#[derive(Debug)]
struct RunnerConfig {
    csv_path: PathBuf,
    failures_path: PathBuf,
    offset: usize,
    limit: Option<usize>,
    composition_filter: Option<String>,
    windows: Vec<WindowSpec>,
    case_numbers: Vec<usize>,
    trace_from: Option<usize>,
    warmup: bool,
}

fn main() {
    let config = parse_args();
    // Keep pressure corpora on the same canonical eval path as `cas_cli eval`
    // and the embedded corpus runner. This benchmark is intended to measure
    // engine robustness, not differences between frontend entrypoints.
    let panic_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let exit_code = if orchestrator_shortcut_profiling_enabled() {
        std::thread::Builder::new()
            .name("mixed-zero-orchestrator-profile".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(move || run(config))
            .expect("failed to spawn mixed zero profiling thread")
            .join()
            .expect("mixed zero profiling thread panicked")
    } else {
        run(config)
    };
    panic::set_hook(panic_hook);
    std::process::exit(exit_code);
}

fn run(config: RunnerConfig) -> i32 {
    let all_cases = load_cases(&config.csv_path);
    let selected_cases = select_cases(&all_cases, &config);

    if selected_cases.is_empty() {
        eprintln!("No corpus cases matched the requested filters.");
        return 0;
    }

    // Discard one untimed warmup evaluation so per-case rankings reflect
    // steady-state engine/runtime cost rather than one-time session startup.
    if config.warmup {
        warm_up_eval_path(&selected_cases[0].case.expression);
    }
    if orchestrator_shortcut_profiling_enabled() {
        clear_orchestrator_shortcut_profile();
    }

    let start = Instant::now();
    let mut failures = Vec::new();
    let mut by_composition = BTreeMap::<String, CompositionSummary>::new();
    let mut by_window = BTreeMap::<String, CompositionSummary>::new();
    let mut slow_cases = Vec::<SlowCaseRecord>::new();

    for (index, selected_case) in selected_cases.iter().enumerate() {
        let case = &selected_case.case;
        if config
            .trace_from
            .is_some_and(|trace_from| index + 1 >= trace_from)
        {
            eprintln!(
                "TRACE case {} (source #{} {}) : {}",
                index + 1,
                case.case_number,
                selected_case.window_label,
                case.expression
            );
        }
        let case_start = Instant::now();
        let evaluation = match panic::catch_unwind(AssertUnwindSafe(|| evaluate_case(case))) {
            Ok(evaluation) => evaluation,
            Err(payload) => CaseEvaluation {
                failure: Some(FailureRecord {
                    case_number: case.case_number,
                    window_label: selected_case.window_label.clone(),
                    expression: case.expression.clone(),
                    expected_result: case.expected_result.clone(),
                    actual_result: String::new(),
                    ok: false,
                    composition: case.composition.clone(),
                    source_expr_1: case.source_expr_1.clone(),
                    source_expr_2: case.source_expr_2.clone(),
                    error_kind: "panic".to_string(),
                    error_message: panic_payload_to_string(payload),
                }),
                timings: CaseTimingRecord::default(),
            },
        };
        let case_elapsed_seconds = case_start.elapsed().as_secs_f64();
        slow_cases.push(SlowCaseRecord {
            case_number: case.case_number,
            window_label: selected_case.window_label.clone(),
            composition: case.composition.clone(),
            wall_elapsed_seconds: case_elapsed_seconds,
            wire_elapsed_seconds: evaluation.timings.wire_elapsed_seconds,
            parse_elapsed_seconds: evaluation.timings.parse_elapsed_seconds,
            simplify_elapsed_seconds: evaluation.timings.simplify_elapsed_seconds,
            expression: case.expression.clone(),
        });
        let entry = by_composition.entry(case.composition.clone()).or_default();
        entry.total += 1;
        entry.wall_elapsed_seconds += case_elapsed_seconds;
        entry.wire_elapsed_seconds += evaluation.timings.wire_elapsed_seconds;
        entry.parse_elapsed_seconds += evaluation.timings.parse_elapsed_seconds;
        entry.simplify_elapsed_seconds += evaluation.timings.simplify_elapsed_seconds;
        if evaluation.failure.is_some() {
            entry.failed += 1;
        } else {
            entry.passed += 1;
        }
        if !selected_case.window_label.is_empty() {
            let window_entry = by_window
                .entry(selected_case.window_label.clone())
                .or_default();
            window_entry.total += 1;
            window_entry.wall_elapsed_seconds += case_elapsed_seconds;
            window_entry.wire_elapsed_seconds += evaluation.timings.wire_elapsed_seconds;
            window_entry.parse_elapsed_seconds += evaluation.timings.parse_elapsed_seconds;
            window_entry.simplify_elapsed_seconds += evaluation.timings.simplify_elapsed_seconds;
            if evaluation.failure.is_some() {
                window_entry.failed += 1;
            } else {
                window_entry.passed += 1;
            }
        }

        if let Some(mut failure) = evaluation.failure {
            failure.window_label = selected_case.window_label.clone();
            failures.push(failure);
        }

        if (index + 1) % 250 == 0 || index + 1 == selected_cases.len() {
            eprintln!("Processed {}/{} cases...", index + 1, selected_cases.len());
        }
    }

    write_failures_csv(&config.failures_path, &failures);

    let passed = selected_cases.len().saturating_sub(failures.len());
    let elapsed = start.elapsed();
    println!("Corpus file: {}", config.csv_path.display());
    println!("Failures file: {}", config.failures_path.display());
    println!("Total cases: {}", selected_cases.len());
    println!("Passed: {}", passed);
    println!("Failed: {}", failures.len());
    println!("Elapsed: {:.2?}", elapsed);
    println!();
    println!("By composition:");
    for (composition, summary) in &by_composition {
        let avg_case_ms = if summary.total > 0 {
            summary.wall_elapsed_seconds * 1_000.0 / summary.total as f64
        } else {
            0.0
        };
        let avg_wire_ms = if summary.total > 0 {
            summary.wire_elapsed_seconds * 1_000.0 / summary.total as f64
        } else {
            0.0
        };
        let avg_parse_ms = if summary.total > 0 {
            summary.parse_elapsed_seconds * 1_000.0 / summary.total as f64
        } else {
            0.0
        };
        let avg_simplify_ms = if summary.total > 0 {
            summary.simplify_elapsed_seconds * 1_000.0 / summary.total as f64
        } else {
            0.0
        };
        println!(
            "  {}: total={} passed={} failed={} elapsed={} avg_case_ms={:.2} wire_elapsed={} avg_wire_ms={:.2} parse_elapsed={} avg_parse_ms={:.2} simplify_elapsed={} avg_simplify_ms={:.2}",
            composition,
            summary.total,
            summary.passed,
            summary.failed,
            format_duration(summary.wall_elapsed_seconds),
            avg_case_ms,
            format_duration(summary.wire_elapsed_seconds),
            avg_wire_ms,
            format_duration(summary.parse_elapsed_seconds),
            avg_parse_ms,
            format_duration(summary.simplify_elapsed_seconds),
            avg_simplify_ms
        );
    }
    if !by_window.is_empty() {
        println!();
        println!("By window:");
        for (window_label, summary) in &by_window {
            let avg_case_ms = if summary.total > 0 {
                summary.wall_elapsed_seconds * 1_000.0 / summary.total as f64
            } else {
                0.0
            };
            let avg_wire_ms = if summary.total > 0 {
                summary.wire_elapsed_seconds * 1_000.0 / summary.total as f64
            } else {
                0.0
            };
            let avg_parse_ms = if summary.total > 0 {
                summary.parse_elapsed_seconds * 1_000.0 / summary.total as f64
            } else {
                0.0
            };
            let avg_simplify_ms = if summary.total > 0 {
                summary.simplify_elapsed_seconds * 1_000.0 / summary.total as f64
            } else {
                0.0
            };
            println!(
                "  {}: total={} passed={} failed={} elapsed={} avg_case_ms={:.2} wire_elapsed={} avg_wire_ms={:.2} parse_elapsed={} avg_parse_ms={:.2} simplify_elapsed={} avg_simplify_ms={:.2}",
                window_label,
                summary.total,
                summary.passed,
                summary.failed,
                format_duration(summary.wall_elapsed_seconds),
                avg_case_ms,
                format_duration(summary.wire_elapsed_seconds),
                avg_wire_ms,
                format_duration(summary.parse_elapsed_seconds),
                avg_parse_ms,
                format_duration(summary.simplify_elapsed_seconds),
                avg_simplify_ms
            );
        }
    }

    slow_cases.sort_by(|lhs, rhs| {
        rhs.wall_elapsed_seconds
            .total_cmp(&lhs.wall_elapsed_seconds)
    });
    println!();
    println!("Top slow cases:");
    for slow_case in slow_cases.iter().take(10) {
        let window_prefix = if slow_case.window_label.is_empty() {
            String::new()
        } else {
            format!("{} ", slow_case.window_label)
        };
        println!(
            "  [{}#{} {}] elapsed={} wire={} parse={} simplify={} expr={}",
            window_prefix,
            slow_case.case_number,
            slow_case.composition,
            format_duration(slow_case.wall_elapsed_seconds),
            format_duration(slow_case.wire_elapsed_seconds),
            format_duration(slow_case.parse_elapsed_seconds),
            format_duration(slow_case.simplify_elapsed_seconds),
            slow_case.expression
        );
    }

    let mut engine_heavy_cases = slow_cases.clone();
    engine_heavy_cases.sort_by(|lhs, rhs| {
        rhs.simplify_elapsed_seconds
            .total_cmp(&lhs.simplify_elapsed_seconds)
    });
    println!();
    println!("Top engine-heavy cases:");
    for slow_case in engine_heavy_cases.iter().take(10) {
        let window_prefix = if slow_case.window_label.is_empty() {
            String::new()
        } else {
            format!("{} ", slow_case.window_label)
        };
        println!(
            "  [{}#{} {}] simplify={} wire={} elapsed={} expr={}",
            window_prefix,
            slow_case.case_number,
            slow_case.composition,
            format_duration(slow_case.simplify_elapsed_seconds),
            format_duration(slow_case.wire_elapsed_seconds),
            format_duration(slow_case.wall_elapsed_seconds),
            slow_case.expression
        );
    }

    let steady_state_engine_heavy = rerun_engine_heavy_cases(&selected_cases, &engine_heavy_cases);
    if !steady_state_engine_heavy.is_empty() {
        println!();
        println!("Steady-state engine-heavy reruns:");
        for rerun_case in steady_state_engine_heavy {
            let window_prefix = if rerun_case.window_label.is_empty() {
                String::new()
            } else {
                format!("{} ", rerun_case.window_label)
            };
            println!(
                "  [{}#{} {}] runs={} median_simplify={} median_wire={} median_parse={} median_elapsed={} expr={}",
                window_prefix,
                rerun_case.case_number,
                rerun_case.composition,
                rerun_case.runs,
                format_duration(rerun_case.median_simplify_elapsed_seconds),
                format_duration(rerun_case.median_wire_elapsed_seconds),
                format_duration(rerun_case.median_parse_elapsed_seconds),
                format_duration(rerun_case.median_wall_elapsed_seconds),
                rerun_case.expression
            );
        }
    }

    let status_code = if failures.is_empty() {
        0
    } else {
        println!();
        println!("Sample failures:");
        for failure in failures.iter().take(10) {
            let window_prefix = if failure.window_label.is_empty() {
                String::new()
            } else {
                format!("{} ", failure.window_label)
            };
            println!(
                "  [{}#{} {}] expected={} actual={} expr={}",
                window_prefix,
                failure.case_number,
                failure.composition,
                failure.expected_result,
                failure.actual_result,
                failure.expression
            );
            if !failure.error_message.is_empty() {
                println!("    error: {}", failure.error_message);
            }
        }
        1
    };

    if orchestrator_shortcut_profiling_enabled() {
        eprintln!();
        eprintln!("{}", orchestrator_shortcut_profile_report());
    }

    status_code
}

fn rerun_engine_heavy_cases(
    selected_cases: &[SelectedCase],
    engine_heavy_cases: &[SlowCaseRecord],
) -> Vec<SteadyStateEngineHeavyRecord> {
    let mut reruns = Vec::new();

    for slow_case in engine_heavy_cases.iter().take(ENGINE_HEAVY_RERUN_LIMIT) {
        let Some(selected_case) = selected_cases.iter().find(|selected_case| {
            selected_case.case.case_number == slow_case.case_number
                && selected_case.window_label == slow_case.window_label
        }) else {
            continue;
        };

        let mut wall_samples = Vec::with_capacity(ENGINE_HEAVY_RERUN_COUNT);
        let mut wire_samples = Vec::with_capacity(ENGINE_HEAVY_RERUN_COUNT);
        let mut parse_samples = Vec::with_capacity(ENGINE_HEAVY_RERUN_COUNT);
        let mut simplify_samples = Vec::with_capacity(ENGINE_HEAVY_RERUN_COUNT);

        for _ in 0..ENGINE_HEAVY_RERUN_COUNT {
            let case_start = Instant::now();
            let evaluation = match panic::catch_unwind(AssertUnwindSafe(|| {
                evaluate_case(&selected_case.case)
            })) {
                Ok(evaluation) if evaluation.failure.is_none() => evaluation,
                _ => continue,
            };
            wall_samples.push(case_start.elapsed().as_secs_f64());
            wire_samples.push(evaluation.timings.wire_elapsed_seconds);
            parse_samples.push(evaluation.timings.parse_elapsed_seconds);
            simplify_samples.push(evaluation.timings.simplify_elapsed_seconds);
        }

        if wall_samples.is_empty() {
            continue;
        }

        reruns.push(SteadyStateEngineHeavyRecord {
            case_number: slow_case.case_number,
            window_label: slow_case.window_label.clone(),
            composition: slow_case.composition.clone(),
            runs: wall_samples.len(),
            median_wall_elapsed_seconds: median_seconds(&mut wall_samples),
            median_wire_elapsed_seconds: median_seconds(&mut wire_samples),
            median_parse_elapsed_seconds: median_seconds(&mut parse_samples),
            median_simplify_elapsed_seconds: median_seconds(&mut simplify_samples),
            expression: slow_case.expression.clone(),
        });
    }

    reruns
}

fn median_seconds(samples: &mut [f64]) -> f64 {
    samples.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}

fn select_cases(all_cases: &[CorpusCase], config: &RunnerConfig) -> Vec<SelectedCase> {
    if !config.case_numbers.is_empty() {
        return config
            .case_numbers
            .iter()
            .map(|case_number| {
                let case = all_cases
                    .iter()
                    .find(|case| case.case_number == *case_number)
                    .unwrap_or_else(|| panic!("unknown --case-number: {case_number}"));
                SelectedCase {
                    case: case.clone(),
                    window_label: format!("case#{case_number}"),
                }
            })
            .collect();
    }

    if !config.windows.is_empty() {
        let mut selected_cases = Vec::new();
        for window in &config.windows {
            selected_cases.extend(select_window_cases(all_cases, window));
        }
        return selected_cases;
    }

    let mut cases = all_cases.to_vec();
    if let Some(filter) = &config.composition_filter {
        cases.retain(|case| &case.composition == filter);
    }
    if config.offset > 0 {
        cases = cases.into_iter().skip(config.offset).collect();
    }
    if let Some(limit) = config.limit {
        cases.truncate(limit);
    }

    cases
        .into_iter()
        .map(|case| SelectedCase {
            case,
            window_label: String::new(),
        })
        .collect()
}

fn select_window_cases(all_cases: &[CorpusCase], window: &WindowSpec) -> Vec<SelectedCase> {
    let mut cases: Vec<CorpusCase> = all_cases.to_vec();
    if let Some(filter) = &window.composition_filter {
        cases.retain(|case| &case.composition == filter);
    }
    if window.offset > 0 {
        cases = cases.into_iter().skip(window.offset).collect();
    }
    cases.truncate(window.limit);

    let window_label = window.label();
    cases
        .into_iter()
        .map(|case| SelectedCase {
            case,
            window_label: window_label.clone(),
        })
        .collect()
}

fn parse_args() -> RunnerConfig {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let default_csv = repo_root
        .join("docs")
        .join("simplify_zero_mixed_corpus.csv");
    let default_failures = repo_root
        .join("docs")
        .join("generated")
        .join("simplify_zero_mixed_corpus_failures.csv");

    let mut csv_path = default_csv;
    let mut failures_path = default_failures;
    let mut offset = 0usize;
    let mut limit = None;
    let mut composition_filter = None;
    let mut windows = Vec::new();
    let mut case_numbers = Vec::new();
    let mut trace_from = None;
    let mut warmup = true;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--csv" => {
                let value = args.next().expect("--csv requires a path");
                csv_path = PathBuf::from(value);
            }
            "--failures" => {
                let value = args.next().expect("--failures requires a path");
                failures_path = PathBuf::from(value);
            }
            "--offset" => {
                let value = args.next().expect("--offset requires a number");
                offset = value
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("invalid --offset value: {value}"));
            }
            "--limit" => {
                let value = args.next().expect("--limit requires a number");
                limit = Some(
                    value
                        .parse::<usize>()
                        .unwrap_or_else(|_| panic!("invalid --limit value: {value}")),
                );
            }
            "--composition" => {
                let value = args.next().expect("--composition requires a value");
                composition_filter = Some(value);
            }
            "--window" => {
                let value = args
                    .next()
                    .expect("--window requires composition:offset:limit");
                windows.push(parse_window_spec(&value));
            }
            "--case-number" => {
                let value = args.next().expect("--case-number requires a number");
                case_numbers.push(
                    value
                        .parse::<usize>()
                        .unwrap_or_else(|_| panic!("invalid --case-number value: {value}")),
                );
            }
            "--trace-from" => {
                let value = args.next().expect("--trace-from requires a number");
                trace_from = Some(
                    value
                        .parse::<usize>()
                        .unwrap_or_else(|_| panic!("invalid --trace-from value: {value}")),
                );
            }
            "--no-warmup" => warmup = false,
            other => panic!("unknown argument: {other}"),
        }
    }

    RunnerConfig {
        csv_path,
        failures_path,
        offset,
        limit,
        composition_filter,
        windows,
        case_numbers,
        trace_from,
        warmup,
    }
}

fn parse_window_spec(raw: &str) -> WindowSpec {
    let mut parts = raw.split(':');
    let composition = parts
        .next()
        .unwrap_or_else(|| panic!("invalid --window value: {raw}"));
    let offset = parts
        .next()
        .unwrap_or_else(|| panic!("invalid --window value: {raw}"))
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid --window offset: {raw}"));
    let limit = parts
        .next()
        .unwrap_or_else(|| panic!("invalid --window value: {raw}"))
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid --window limit: {raw}"));
    assert!(parts.next().is_none(), "invalid --window value: {raw}");

    WindowSpec {
        composition_filter: match composition {
            "all" | "*" => None,
            _ => Some(composition.to_string()),
        },
        offset,
        limit,
    }
}

fn load_cases(path: &Path) -> Vec<CorpusCase> {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    contents
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(index, line)| {
            let parts = split_csv_line(line);
            assert_eq!(
                parts.len(),
                5,
                "unexpected mixed corpus csv columns: {line}"
            );
            CorpusCase {
                case_number: index + 1,
                expression: parts[0].trim().to_string(),
                expected_result: parts[1].trim().to_string(),
                composition: parts[2].trim().to_string(),
                source_expr_1: parts[3].trim().to_string(),
                source_expr_2: parts[4].trim().to_string(),
            }
        })
        .collect()
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                parts.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    parts.push(current.trim().to_string());
    parts
}

fn format_duration(seconds: f64) -> String {
    if seconds >= 1.0 {
        format!("{seconds:.2}s")
    } else {
        format!("{:.2}ms", seconds * 1_000.0)
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn evaluate_case(case: &CorpusCase) -> CaseEvaluation {
    let config = mixed_zero_eval_command_config(&case.expression);
    let (output, _, _) = evaluate_eval_command_with_session(
        None,
        config,
        cas_solver_core::eval_option_axes::Language::Es,
        |_, _, _, _| Vec::new(),
    );

    let (actual_result, ok, error_kind, error_message, timings) = match output {
        Ok(output) => {
            let actual_result = output.result;
            let ok = actual_result == case.expected_result;
            let error_kind = if ok {
                String::new()
            } else {
                "result_mismatch".to_string()
            };
            (
                actual_result,
                ok,
                error_kind,
                String::new(),
                CaseTimingRecord {
                    wire_elapsed_seconds: output.timings_us.total_us as f64 / 1_000_000.0,
                    parse_elapsed_seconds: output.timings_us.parse_us as f64 / 1_000_000.0,
                    simplify_elapsed_seconds: output.timings_us.simplify_us as f64 / 1_000_000.0,
                },
            )
        }
        Err(error_message) => (
            String::new(),
            false,
            "session_eval_error".to_string(),
            error_message,
            CaseTimingRecord::default(),
        ),
    };

    if ok {
        return CaseEvaluation {
            failure: None,
            timings,
        };
    }

    CaseEvaluation {
        failure: Some(FailureRecord {
            case_number: case.case_number,
            window_label: String::new(),
            expression: case.expression.clone(),
            expected_result: case.expected_result.clone(),
            actual_result,
            ok,
            composition: case.composition.clone(),
            source_expr_1: case.source_expr_1.clone(),
            source_expr_2: case.source_expr_2.clone(),
            error_kind,
            error_message,
        }),
        timings,
    }
}

fn warm_up_eval_path(expression: &str) {
    let config = mixed_zero_eval_command_config(expression);
    let _ = evaluate_eval_command_with_session(
        None,
        config,
        cas_solver_core::eval_option_axes::Language::Es,
        |_, _, _, _| Vec::new(),
    );
}

fn mixed_zero_eval_command_config<'a>(expression: &'a str) -> EvalCommandConfig<'a> {
    EvalCommandConfig {
        expr: expression,
        auto_store: false,
        max_chars: usize::MAX,
        time_budget_ms: None,
        steps_mode: EvalStepsMode::Off,
        budget_preset: EvalBudgetPreset::Standard,
        strict: false,
        domain: EvalDomainMode::Generic,
        context_mode: EvalContextMode::Auto,
        branch_mode: EvalBranchMode::Strict,
        expand_policy: EvalExpandPolicy::Off,
        complex_mode: EvalComplexMode::Auto,
        const_fold: EvalConstFoldMode::Off,
        value_domain: EvalValueDomain::Real,
        complex_branch: EvalBranchMode::Principal,
        inv_trig: EvalInvTrigPolicy::Strict,
        assume_scope: EvalAssumeScope::Real,
    }
}

fn write_failures_csv(path: &Path, failures: &[FailureRecord]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("failed to create {}: {err}", parent.display()));
    }

    let mut output = String::from(
        "case_number,window,expression,expected_result,actual_result,ok,composition,source_expr_1,source_expr_2,error_kind,error_message\n",
    );
    for failure in failures {
        output.push_str(&failure.case_number.to_string());
        output.push(',');
        output.push_str(&csv_escape(&failure.window_label));
        output.push(',');
        output.push_str(&csv_escape(&failure.expression));
        output.push(',');
        output.push_str(&csv_escape(&failure.expected_result));
        output.push(',');
        output.push_str(&csv_escape(&failure.actual_result));
        output.push(',');
        output.push_str(if failure.ok { "true" } else { "false" });
        output.push(',');
        output.push_str(&csv_escape(&failure.composition));
        output.push(',');
        output.push_str(&csv_escape(&failure.source_expr_1));
        output.push(',');
        output.push_str(&csv_escape(&failure.source_expr_2));
        output.push(',');
        output.push_str(&csv_escape(&failure.error_kind));
        output.push(',');
        output.push_str(&csv_escape(&failure.error_message));
        output.push('\n');
    }

    fs::write(path, output)
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", path.display()));
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}
