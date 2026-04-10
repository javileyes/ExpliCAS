use cas_solver::wire::eval_str_to_wire;
use serde_json::Value;
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
}

#[derive(Debug)]
struct RunnerConfig {
    csv_path: PathBuf,
    failures_path: PathBuf,
    limit: Option<usize>,
    composition_filter: Option<String>,
}

fn main() {
    let config = parse_args();
    let mut cases = load_cases(&config.csv_path);
    if let Some(filter) = &config.composition_filter {
        cases.retain(|case| &case.composition == filter);
    }
    if let Some(limit) = config.limit {
        cases.truncate(limit);
    }

    if cases.is_empty() {
        eprintln!("No corpus cases matched the requested filters.");
        return;
    }

    let start = Instant::now();
    let mut failures = Vec::new();
    let mut by_composition = BTreeMap::<String, CompositionSummary>::new();

    for (index, case) in cases.iter().enumerate() {
        let failure = evaluate_case(case);
        let entry = by_composition.entry(case.composition.clone()).or_default();
        entry.total += 1;
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
        println!(
            "  {}: total={} passed={} failed={}",
            composition, summary.total, summary.passed, summary.failed
        );
    }

    if !failures.is_empty() {
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
        std::process::exit(1);
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
            other => panic!("unknown argument: {other}"),
        }
    }

    RunnerConfig {
        csv_path,
        failures_path,
        limit,
        composition_filter,
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

fn evaluate_case(case: &CorpusCase) -> Option<FailureRecord> {
    let payload = eval_str_to_wire(&case.expression, "{}");
    let wire = serde_json::from_str::<Value>(&payload).unwrap_or_else(|err| {
        panic!(
            "failed to parse wire for '{}': {err}\n{payload}",
            case.expression
        )
    });
    let ok = wire.get("ok").and_then(Value::as_bool).unwrap_or(false);
    let actual_result = wire
        .get("result")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    if ok && actual_result == case.expected_result {
        return None;
    }

    let error_kind = wire
        .get("error")
        .and_then(|error| error.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let error_message = wire
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

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
