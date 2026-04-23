use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalStepsMode,
    EvalValueDomain,
};
use cas_session::eval::{evaluate_eval_command_with_session, EvalCommandConfig};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
struct CorpusCase {
    expression: String,
    expected_result: String,
    composition: String,
    source_expr_1: String,
    source_expr_2: String,
}

#[derive(Debug, Clone)]
struct FailureRecord {
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
    elapsed_seconds: f64,
}

#[derive(Debug)]
struct RunnerConfig {
    csv_path: PathBuf,
    failures_path: PathBuf,
    limit: Option<usize>,
    composition_filter: Option<String>,
    trace_from: Option<usize>,
}

fn main() {
    let config = parse_args();
    // Keep pressure corpora on the same canonical eval path as `cas_cli eval`
    // and the embedded corpus runner. This benchmark is intended to measure
    // engine robustness, not differences between frontend entrypoints.
    std::process::exit(run(config));
}

fn run(config: RunnerConfig) -> i32 {
    let mut cases = load_cases(&config.csv_path);
    if let Some(filter) = &config.composition_filter {
        cases.retain(|case| &case.composition == filter);
    }
    if let Some(limit) = config.limit {
        cases.truncate(limit);
    }

    if cases.is_empty() {
        eprintln!("No corpus cases matched the requested filters.");
        return 0;
    }

    let start = Instant::now();
    let mut failures = Vec::new();
    let mut by_composition = BTreeMap::<String, CompositionSummary>::new();

    for (index, case) in cases.iter().enumerate() {
        if config
            .trace_from
            .is_some_and(|trace_from| index + 1 >= trace_from)
        {
            eprintln!("TRACE case {}: {}", index + 1, case.expression);
        }
        let case_start = Instant::now();
        let failure = evaluate_case(case);
        let case_elapsed_seconds = case_start.elapsed().as_secs_f64();
        let entry = by_composition.entry(case.composition.clone()).or_default();
        entry.total += 1;
        entry.elapsed_seconds += case_elapsed_seconds;
        if failure.is_some() {
            entry.failed += 1;
        } else {
            entry.passed += 1;
        }

        if let Some(failure) = failure {
            failures.push(failure);
        }

        if (index + 1) % 250 == 0 || index + 1 == cases.len() {
            eprintln!("Processed {}/{} cases...", index + 1, cases.len());
        }
    }

    write_failures_csv(&config.failures_path, &failures);

    let passed = cases.len().saturating_sub(failures.len());
    let elapsed = start.elapsed();
    println!("Corpus file: {}", config.csv_path.display());
    println!("Failures file: {}", config.failures_path.display());
    println!("Total cases: {}", cases.len());
    println!("Passed: {}", passed);
    println!("Failed: {}", failures.len());
    println!("Elapsed: {:.2?}", elapsed);
    println!();
    println!("By composition:");
    for (composition, summary) in &by_composition {
        let avg_case_ms = if summary.total > 0 {
            summary.elapsed_seconds * 1_000.0 / summary.total as f64
        } else {
            0.0
        };
        println!(
            "  {}: total={} passed={} failed={} elapsed={} avg_case_ms={:.2}",
            composition,
            summary.total,
            summary.passed,
            summary.failed,
            format_duration(summary.elapsed_seconds),
            avg_case_ms
        );
    }

    if failures.is_empty() {
        0
    } else {
        println!();
        println!("Sample failures:");
        for failure in failures.iter().take(10) {
            println!(
                "  [{}] expected={} actual={} expr={}",
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
    }
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
    let mut limit = None;
    let mut composition_filter = None;
    let mut trace_from = None;

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
            "--trace-from" => {
                let value = args.next().expect("--trace-from requires a number");
                trace_from = Some(
                    value
                        .parse::<usize>()
                        .unwrap_or_else(|_| panic!("invalid --trace-from value: {value}")),
                );
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    RunnerConfig {
        csv_path,
        failures_path,
        limit,
        composition_filter,
        trace_from,
    }
}

fn load_cases(path: &Path) -> Vec<CorpusCase> {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    contents
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let parts = split_csv_line(line);
            assert_eq!(
                parts.len(),
                5,
                "unexpected mixed corpus csv columns: {line}"
            );
            CorpusCase {
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

fn evaluate_case(case: &CorpusCase) -> Option<FailureRecord> {
    let config = EvalCommandConfig {
        expr: &case.expression,
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
    };
    let (output, _, _) = evaluate_eval_command_with_session(None, config, |_, _, _, _| Vec::new());

    let (actual_result, ok, error_kind, error_message) = match output {
        Ok(output) => {
            let actual_result = output.result;
            let ok = actual_result == case.expected_result;
            let error_kind = if ok {
                String::new()
            } else {
                "result_mismatch".to_string()
            };
            (actual_result, ok, error_kind, String::new())
        }
        Err(error_message) => (
            String::new(),
            false,
            "session_eval_error".to_string(),
            error_message,
        ),
    };

    if ok {
        return None;
    }

    Some(FailureRecord {
        expression: case.expression.clone(),
        expected_result: case.expected_result.clone(),
        actual_result,
        ok,
        composition: case.composition.clone(),
        source_expr_1: case.source_expr_1.clone(),
        source_expr_2: case.source_expr_2.clone(),
        error_kind,
        error_message,
    })
}

fn write_failures_csv(path: &Path, failures: &[FailureRecord]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("failed to create {}: {err}", parent.display()));
    }

    let mut output = String::from(
        "expression,expected_result,actual_result,ok,composition,source_expr_1,source_expr_2,error_kind,error_message\n",
    );
    for failure in failures {
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
