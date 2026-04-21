use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalStepsMode,
    EvalValueDomain,
};
use cas_ast::{count_nodes_and_max_depth, Context};
use cas_engine::{
    clear_orchestrator_shortcut_profile, orchestrator_shortcut_profile_report,
    orchestrator_shortcut_profiling_enabled,
};
use cas_parser::parse;
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
    complexity: ComplexityProfile,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ComplexityLevel {
    L0RootPair,
    L1SingleWrapper,
    L2WrapperPlusNoise,
    L3NestedOrComposed,
}

impl ComplexityLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::L0RootPair => "l0_root_pair",
            Self::L1SingleWrapper => "l1_single_wrapper",
            Self::L2WrapperPlusNoise => "l2_wrapper_plus_noise",
            Self::L3NestedOrComposed => "l3_nested_or_composed",
        }
    }

    fn parse_arg(value: &str) -> Option<Self> {
        match value {
            "l0" | "l0_root_pair" | "root_pair" => Some(Self::L0RootPair),
            "l1" | "l1_single_wrapper" | "single_wrapper" => Some(Self::L1SingleWrapper),
            "l2" | "l2_wrapper_plus_noise" | "wrapper_plus_noise" => Some(Self::L2WrapperPlusNoise),
            "l3" | "l3_nested_or_composed" | "nested_or_composed" => Some(Self::L3NestedOrComposed),
            _ => None,
        }
    }
}

impl std::fmt::Display for ComplexityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy)]
struct ComplexityProfile {
    expression_depth: usize,
    wrapper_overhead_nodes: usize,
    shell_depth: usize,
    level: ComplexityLevel,
}

#[derive(Debug, Default, Clone, Copy)]
struct ComplexitySummary {
    total: usize,
    passed: usize,
    failed: usize,
    total_wrapper_overhead_nodes: usize,
    total_shell_depth: usize,
    max_shell_depth: usize,
}

impl ComplexitySummary {
    fn record(&mut self, complexity: ComplexityProfile, failed: bool) {
        self.total += 1;
        if failed {
            self.failed += 1;
        } else {
            self.passed += 1;
        }
        self.total_wrapper_overhead_nodes += complexity.wrapper_overhead_nodes;
        self.total_shell_depth += complexity.shell_depth;
        self.max_shell_depth = self.max_shell_depth.max(complexity.shell_depth);
    }

    fn avg_wrapper_overhead_nodes(self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.total_wrapper_overhead_nodes as f64 / self.total as f64
        }
    }

    fn avg_shell_depth(self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.total_shell_depth as f64 / self.total as f64
        }
    }
}

#[derive(Debug)]
struct RunnerConfig {
    csv_path: PathBuf,
    failures_path: PathBuf,
    limit: Option<usize>,
    wrapper_filter: Option<String>,
    family_filter: Option<String>,
    complexity_level_filter: Option<ComplexityLevel>,
}

fn main() {
    let config = parse_args();
    // Match the direct wire path used by `cas_cli eval` in the default case.
    // When orchestrator profiling is enabled, the extra wrapper frames can push
    // deep embedded cases past the default thread stack budget, so run the
    // corpus inside a larger-stack worker thread.
    let status = if orchestrator_shortcut_profiling_enabled() {
        std::thread::Builder::new()
            .name("embedded-orchestrator-profile".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(move || run(config))
            .expect("failed to spawn embedded profiling thread")
            .join()
            .expect("embedded profiling thread panicked")
    } else {
        run(config)
    };
    std::process::exit(status);
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
    if let Some(level) = config.complexity_level_filter {
        cases.retain(|case| case.complexity.level == level);
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
    let mut by_shell_depth = BTreeMap::<usize, Summary>::new();
    let mut by_complexity = BTreeMap::<ComplexityLevel, ComplexitySummary>::new();
    let mut by_wrapper_complexity = BTreeMap::<(String, ComplexityLevel), ComplexitySummary>::new();

    for (index, case) in cases.iter().enumerate() {
        let failure = evaluate_case(case);
        let failed = failure.is_some();

        let wrapper_entry = by_wrapper.entry(case.wrapper.clone()).or_default();
        wrapper_entry.total += 1;

        let family_entry = by_family.entry(case.family.clone()).or_default();
        family_entry.total += 1;

        let shell_depth_entry = by_shell_depth
            .entry(case.complexity.shell_depth)
            .or_default();
        shell_depth_entry.total += 1;

        by_complexity
            .entry(case.complexity.level)
            .or_default()
            .record(case.complexity, failed);
        by_wrapper_complexity
            .entry((case.wrapper.clone(), case.complexity.level))
            .or_default()
            .record(case.complexity, failed);

        if let Some(failure) = failure {
            wrapper_entry.failed += 1;
            family_entry.failed += 1;
            shell_depth_entry.failed += 1;
            failures.push(failure);
        } else {
            wrapper_entry.passed += 1;
            family_entry.passed += 1;
            shell_depth_entry.passed += 1;
        }

        if (index + 1) % 250 == 0 || index + 1 == cases.len() {
            eprintln!("Processed {}/{} cases...", index + 1, cases.len());
        }
    }

    write_failures_csv(&config.failures_path, &failures);

    let passed = cases.len().saturating_sub(failures.len());
    let elapsed = start.elapsed();
    let wrapper_names: Vec<_> = by_wrapper.keys().cloned().collect();
    let largest_wrapper_cases = by_wrapper
        .values()
        .map(|summary| summary.total)
        .max()
        .unwrap_or(0);
    let largest_wrapper_share = if cases.is_empty() {
        0.0
    } else {
        largest_wrapper_cases as f64 / cases.len() as f64
    };
    let largest_wrapper_complexity_cases = by_wrapper_complexity
        .values()
        .map(|summary| summary.total)
        .max()
        .unwrap_or(0);
    let largest_wrapper_complexity_share = if cases.is_empty() {
        0.0
    } else {
        largest_wrapper_complexity_cases as f64 / cases.len() as f64
    };
    let max_shell_depth = by_shell_depth.keys().copied().max().unwrap_or(0);
    let max_expression_depth = cases
        .iter()
        .map(|case| case.complexity.expression_depth)
        .max()
        .unwrap_or(0);
    let avg_wrapper_overhead_nodes = if cases.is_empty() {
        0.0
    } else {
        cases
            .iter()
            .map(|case| case.complexity.wrapper_overhead_nodes)
            .sum::<usize>() as f64
            / cases.len() as f64
    };
    println!("Corpus file: {}", config.csv_path.display());
    println!("Failures file: {}", config.failures_path.display());
    println!("Total cases: {}", cases.len());
    println!("Passed: {}", passed);
    println!("Failed: {}", failures.len());
    println!("Elapsed: {:.2?}", elapsed);
    println!("Distinct wrappers: {}", by_wrapper.len());
    println!("Distinct families: {}", by_family.len());
    println!("Distinct complexity levels: {}", by_complexity.len());
    println!("Distinct shell depths: {}", by_shell_depth.len());
    println!(
        "Largest wrapper share: {:.1}%",
        largest_wrapper_share * 100.0
    );
    println!(
        "Largest wrapper x complexity share: {:.1}%",
        largest_wrapper_complexity_share * 100.0
    );
    println!("Max shell depth: {}", max_shell_depth);
    println!("Max expression depth: {}", max_expression_depth);
    println!(
        "Average wrapper overhead nodes: {:.2}",
        avg_wrapper_overhead_nodes
    );
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
    println!("By shell depth:");
    for (shell_depth, summary) in &by_shell_depth {
        println!(
            "  depth {}: total={} passed={} failed={}",
            shell_depth, summary.total, summary.passed, summary.failed
        );
    }
    println!();
    println!("By complexity level:");
    for (level, summary) in &by_complexity {
        println!(
            "  {}: total={} passed={} failed={} avg_wrapper_overhead_nodes={:.2} avg_shell_depth={:.2} max_shell_depth={}",
            level,
            summary.total,
            summary.passed,
            summary.failed,
            summary.avg_wrapper_overhead_nodes(),
            summary.avg_shell_depth(),
            summary.max_shell_depth,
        );
    }
    println!();
    println!("Top wrapper x complexity buckets:");
    let mut wrapper_complexity_rows: Vec<_> = by_wrapper_complexity.iter().collect();
    wrapper_complexity_rows.sort_by_key(|((_, _), summary)| std::cmp::Reverse(summary.total));
    for ((wrapper, level), summary) in wrapper_complexity_rows.into_iter().take(20) {
        println!(
            "  {} x {}: total={} passed={} failed={} avg_wrapper_overhead_nodes={:.2} avg_shell_depth={:.2} max_shell_depth={}",
            wrapper,
            level,
            summary.total,
            summary.passed,
            summary.failed,
            summary.avg_wrapper_overhead_nodes(),
            summary.avg_shell_depth(),
            summary.max_shell_depth,
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
    let mut complexity_level_filter = None;

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
            "--complexity-level" => {
                let value = args.next().expect("--complexity-level requires a value");
                complexity_level_filter = Some(ComplexityLevel::parse_arg(&value).unwrap_or_else(
                    || {
                        panic!(
                            "invalid --complexity-level value: {value} (expected one of: l0_root_pair, l1_single_wrapper, l2_wrapper_plus_noise, l3_nested_or_composed)"
                        )
                    },
                ));
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
        complexity_level_filter,
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
                complexity: analyze_case_complexity(&parts[0], &parts[5], &parts[6]),
            }
        })
        .collect()
}

fn analyze_case_complexity(expression: &str, source: &str, target: &str) -> ComplexityProfile {
    let mut ctx = Context::default();
    let expression_id = parse(expression, &mut ctx)
        .unwrap_or_else(|err| panic!("failed to parse embedded expression `{expression}`: {err}"));
    let source_id = parse(source, &mut ctx)
        .unwrap_or_else(|err| panic!("failed to parse source `{source}`: {err}"));
    let target_id = parse(target, &mut ctx)
        .unwrap_or_else(|err| panic!("failed to parse target `{target}`: {err}"));

    let (expression_nodes, expression_depth) = count_nodes_and_max_depth(&ctx, expression_id);
    let (source_nodes, source_depth) = count_nodes_and_max_depth(&ctx, source_id);
    let (target_nodes, target_depth) = count_nodes_and_max_depth(&ctx, target_id);

    // Use the larger side of the naked pair as the core baseline. This keeps the
    // contextual delta focused on wrapper/noise overhead instead of the intrinsic
    // asymmetry of a specific identity orientation.
    let core_nodes = source_nodes.max(target_nodes);
    let core_depth = source_depth.max(target_depth);
    let wrapper_overhead_nodes = expression_nodes.saturating_sub(core_nodes);
    let shell_depth = expression_depth.saturating_sub(core_depth);
    let level = classify_complexity_level(shell_depth, wrapper_overhead_nodes);

    ComplexityProfile {
        expression_depth,
        wrapper_overhead_nodes,
        shell_depth,
        level,
    }
}

fn classify_complexity_level(shell_depth: usize, wrapper_overhead_nodes: usize) -> ComplexityLevel {
    if shell_depth == 0 && wrapper_overhead_nodes <= 2 {
        ComplexityLevel::L0RootPair
    } else if shell_depth <= 1 && wrapper_overhead_nodes <= 6 {
        ComplexityLevel::L1SingleWrapper
    } else if shell_depth <= 2 && wrapper_overhead_nodes <= 14 {
        ComplexityLevel::L2WrapperPlusNoise
    } else {
        ComplexityLevel::L3NestedOrComposed
    }
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
