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
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
struct CorpusCase {
    expression: String,
    expected_result: String,
    wrapper: String,
    pair_id: String,
    family: String,
    source: String,
    target: String,
    expected_strategy: String,
}

#[derive(Debug, Clone)]
struct FailureRecord {
    expression: String,
    expected_result: String,
    actual_result: String,
    ok: bool,
    wrapper: String,
    pair_id: String,
    family: String,
    source: String,
    target: String,
    expected_strategy: String,
    error_kind: String,
    error_message: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct Summary {
    total: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug)]
struct RunnerConfig {
    csv_path: PathBuf,
    failures_path: PathBuf,
    limit: Option<usize>,
    wrapper_filter: Option<String>,
    family_filter: Option<String>,
}

fn main() {
    let config = parse_args();
    // Match the direct wire path used by `cas_cli eval`. The fixed-stack helper
    // thread has proven less robust than the main-thread path on deep recursive
    // shifted-quotient cases, even when those cases succeed through the normal
    // CLI entrypoint.
    std::process::exit(run(config));
}

fn run(config: RunnerConfig) -> i32 {
    if orchestrator_shortcut_profiling_enabled() {
        clear_orchestrator_shortcut_profile();
    }

    let mut cases = load_cases(&config.csv_path);
    if let Some(wrapper) = &config.wrapper_filter {
        cases.retain(|case| &case.wrapper == wrapper);
    }
    if let Some(family) = &config.family_filter {
        cases.retain(|case| &case.family == family);
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
    let mut by_wrapper = BTreeMap::<String, Summary>::new();
    let mut by_family = BTreeMap::<String, Summary>::new();

    for (index, case) in cases.iter().enumerate() {
        let failure = evaluate_case(case);

        let wrapper_entry = by_wrapper.entry(case.wrapper.clone()).or_default();
        wrapper_entry.total += 1;

        let family_entry = by_family.entry(case.family.clone()).or_default();
        family_entry.total += 1;

        if let Some(failure) = failure {
            wrapper_entry.failed += 1;
            family_entry.failed += 1;
            failures.push(failure);
        } else {
            wrapper_entry.passed += 1;
            family_entry.passed += 1;
        }

        if (index + 1) % 250 == 0 || index + 1 == cases.len() {
            eprintln!("Processed {}/{} cases...", index + 1, cases.len());
        }
    }

    write_failures_csv(&config.failures_path, &failures);

    let passed = cases.len().saturating_sub(failures.len());
    let elapsed = start.elapsed();
    let wrapper_names: Vec<_> = by_wrapper.keys().cloned().collect();
    let largest_wrapper_cases = by_wrapper.values().map(|summary| summary.total).max().unwrap_or(0);
    let largest_wrapper_share = if cases.is_empty() {
        0.0
    } else {
        largest_wrapper_cases as f64 / cases.len() as f64
    };
    println!("Corpus file: {}", config.csv_path.display());
    println!("Failures file: {}", config.failures_path.display());
    println!("Total cases: {}", cases.len());
    println!("Passed: {}", passed);
    println!("Failed: {}", failures.len());
    println!("Elapsed: {:.2?}", elapsed);
    println!("Distinct wrappers: {}", by_wrapper.len());
    println!("Distinct families: {}", by_family.len());
    println!("Largest wrapper share: {:.1}%", largest_wrapper_share * 100.0);
    println!("Wrappers: {}", wrapper_names.join(", "));
    println!();
    println!("By wrapper:");
    for (wrapper, summary) in &by_wrapper {
        println!(
            "  {}: total={} passed={} failed={}",
            wrapper, summary.total, summary.passed, summary.failed
        );
    }
    println!();
    println!("Top families by failures:");
    let mut family_rows: Vec<_> = by_family.iter().collect();
    family_rows.sort_by_key(|(_, summary)| std::cmp::Reverse(summary.failed));
    for (family, summary) in family_rows.into_iter().take(20) {
        println!(
            "  {}: total={} passed={} failed={}",
            family, summary.total, summary.passed, summary.failed
        );
    }

    let status_code = if failures.is_empty() {
        0
    } else {
        println!();
        println!("Sample failures:");
        for failure in failures.iter().take(10) {
            println!(
                "  [{}|{}] expected={} actual={} expr={}",
                failure.wrapper,
                failure.family,
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

fn parse_args() -> RunnerConfig {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let default_csv = repo_root
        .join("docs")
        .join("embedded_equivalence_context_corpus.csv");
    let default_failures = repo_root
        .join("docs")
        .join("generated")
        .join("embedded_equivalence_context_corpus_failures.csv");

    let mut csv_path = default_csv;
    let mut failures_path = default_failures;
    let mut limit = None;
    let mut wrapper_filter = None;
    let mut family_filter = None;

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
            "--wrapper" => {
                wrapper_filter = Some(args.next().expect("--wrapper requires a value"));
            }
            "--family" => {
                family_filter = Some(args.next().expect("--family requires a value"));
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    RunnerConfig {
        csv_path,
        failures_path,
        limit,
        wrapper_filter,
        family_filter,
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
                8,
                "unexpected embedded context csv columns: {line}"
            );
            CorpusCase {
                expression: parts[0].clone(),
                expected_result: parts[1].clone(),
                wrapper: parts[2].clone(),
                pair_id: parts[3].clone(),
                family: parts[4].clone(),
                source: parts[5].clone(),
                target: parts[6].clone(),
                expected_strategy: parts[7].clone(),
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
    let config = EvalCommandConfig {
        expr: &case.expression,
        auto_store: false,
        max_chars: usize::MAX,
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

    (!ok).then(|| FailureRecord {
        expression: case.expression.clone(),
        expected_result: case.expected_result.clone(),
        actual_result,
        ok,
        wrapper: case.wrapper.clone(),
        pair_id: case.pair_id.clone(),
        family: case.family.clone(),
        source: case.source.clone(),
        target: case.target.clone(),
        expected_strategy: case.expected_strategy.clone(),
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
        "expression,expected_result,actual_result,ok,wrapper,pair_id,family,source,target,expected_strategy,error_kind,error_message\n",
    );
    for failure in failures {
        let fields = vec![
            failure.expression.clone(),
            failure.expected_result.clone(),
            failure.actual_result.clone(),
            if failure.ok {
                "true".to_string()
            } else {
                "false".to_string()
            },
            failure.wrapper.clone(),
            failure.pair_id.clone(),
            failure.family.clone(),
            failure.source.clone(),
            failure.target.clone(),
            failure.expected_strategy.clone(),
            failure.error_kind.clone(),
            failure.error_message.clone(),
        ];
        let row = fields
            .iter()
            .map(|s| csv_escape(s))
            .collect::<Vec<_>>()
            .join(",");
        output.push_str(&row);
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
