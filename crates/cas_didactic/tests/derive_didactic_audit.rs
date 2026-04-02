use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalStepsMode,
    EvalValueDomain,
};
use cas_didactic::collect_step_payloads_with_events;
use cas_session::eval::{evaluate_eval_command_pretty_with_session, EvalCommandConfig};
use cas_solver::runtime::{Simplifier, SimplifyOptions};
use cas_solver::session_api::analysis::{
    evaluate_derive_command_lines_with_resolver, FullSimplifyDisplayMode,
};
use serde_json::Value;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct DeriveCase {
    id: String,
    family: String,
    source: String,
    target: String,
    expected_status: String,
}

#[derive(Debug)]
struct AuditArtifact {
    result: String,
    step_count: usize,
    web_substep_count: usize,
    cli_lines: Vec<String>,
    json_steps: Vec<Value>,
    flags: Vec<String>,
}

fn load_derive_cases() -> Vec<DeriveCase> {
    include_str!("../../cas_solver/tests/derive_pairs.csv")
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let parts = split_csv_line(line);
            assert_eq!(parts.len(), 6, "unexpected derive csv columns: {line}");
            DeriveCase {
                id: parts[0].trim().to_string(),
                family: parts[1].trim().to_string(),
                source: parts[2].trim().to_string(),
                target: parts[3].trim().to_string(),
                expected_status: parts[4].trim().to_string(),
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

fn derive_eval_config(expr: &str) -> EvalCommandConfig<'_> {
    EvalCommandConfig {
        expr,
        auto_store: false,
        max_chars: 8000,
        steps_mode: EvalStepsMode::On,
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

fn run_cli_lines(case: &DeriveCase) -> Vec<String> {
    let input = format!("derive {}, {}", case.source, case.target);
    let mut simplifier = Simplifier::with_default_rules();
    evaluate_derive_command_lines_with_resolver(
        &mut simplifier,
        &input,
        FullSimplifyDisplayMode::Normal,
        SimplifyOptions::default(),
        |_ctx, expr| Ok(expr),
    )
    .expect("derive should evaluate")
}

fn normalize_latex_snapshot(input: &str) -> String {
    let mut normalized = input.replace("\\,", "");
    for needle in ["{\\color{red}{", "{\\color{green}{", "{\\color{blue}{"] {
        normalized = normalized.replace(needle, "");
    }
    normalized.retain(|ch| !ch.is_whitespace());
    while normalized.starts_with('{') && normalized.ends_with('}') && normalized.len() >= 2 {
        normalized = normalized[1..normalized.len() - 1].to_string();
    }
    normalized
}

fn has_redundant_single_substep(step: &Value) -> bool {
    let Some(substeps) = step.get("substeps").and_then(Value::as_array) else {
        return false;
    };
    if substeps.len() != 1 {
        return false;
    }
    let substep = &substeps[0];
    let parent_before = step
        .get("before_latex")
        .and_then(Value::as_str)
        .map(normalize_latex_snapshot);
    let parent_after = step
        .get("after_latex")
        .and_then(Value::as_str)
        .map(normalize_latex_snapshot);
    let sub_before = substep
        .get("before_latex")
        .and_then(Value::as_str)
        .map(normalize_latex_snapshot);
    let sub_after = substep
        .get("after_latex")
        .and_then(Value::as_str)
        .map(normalize_latex_snapshot);

    matches!(
        (parent_before, parent_after, sub_before, sub_after),
        (Some(pb), Some(pa), Some(sb), Some(sa)) if pb == sb && pa == sa
    )
}

fn audit_case(case: &DeriveCase) -> AuditArtifact {
    let expr = format!("derive({}, {})", case.source, case.target);
    let json = evaluate_eval_command_pretty_with_session(
        None,
        derive_eval_config(&expr),
        |steps, events, context, steps_mode| {
            collect_step_payloads_with_events(steps, events, context, steps_mode)
        },
    );
    let payload: Value = serde_json::from_str(&json).expect("derive pretty json");
    assert_eq!(
        payload.get("ok").and_then(Value::as_bool),
        Some(true),
        "derive audit case {} should evaluate successfully",
        case.id
    );

    let result = payload
        .get("result")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let json_steps = payload
        .get("steps")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let step_count = json_steps.len();
    let web_substep_count = json_steps
        .iter()
        .map(|step| {
            step.get("substeps")
                .and_then(Value::as_array)
                .map_or(0, |substeps| substeps.len())
        })
        .sum();
    let cli_lines = run_cli_lines(case);

    let mut flags = Vec::new();
    if step_count == 0 {
        flags.push("no web steps emitted".to_string());
    }
    if case.expected_status == "derived" && web_substep_count == 0 {
        flags.push("no web substeps emitted".to_string());
    }
    if json_steps.iter().any(has_redundant_single_substep) {
        flags.push("redundant single substep duplicates parent step".to_string());
    }

    AuditArtifact {
        result,
        step_count,
        web_substep_count,
        cli_lines,
        json_steps,
        flags,
    }
}

fn derive_case_by_id(id: &str) -> DeriveCase {
    load_derive_cases()
        .into_iter()
        .find(|case| case.id == id)
        .unwrap_or_else(|| panic!("missing derive audit case {id}"))
}

fn report_output_path() -> PathBuf {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .expect("crates/")
        .parent()
        .expect("repo root")
        .join("docs/generated/DERIVE_DIDACTIC_AUDIT.md")
}

fn build_report(cases: &[DeriveCase], artifacts: &[AuditArtifact]) -> String {
    let mut out = String::new();
    let derived_cases = cases
        .iter()
        .filter(|case| case.expected_status == "derived")
        .count();
    let total_steps = artifacts
        .iter()
        .map(|artifact| artifact.step_count)
        .sum::<usize>();
    let total_substeps = artifacts
        .iter()
        .map(|artifact| artifact.web_substep_count)
        .sum::<usize>();
    let mean_steps = if derived_cases == 0 {
        0.0
    } else {
        total_steps as f64 / derived_cases as f64
    };

    writeln!(out, "# Derive Didactic Audit").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Generated from [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Command: `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`"
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(out, "## Summary").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "- Derived cases audited: `{derived_cases}`").unwrap();
    writeln!(out, "- Mean top-level step count: `{mean_steps:.2}`").unwrap();
    writeln!(out, "- Total web substeps: `{total_substeps}`").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| id | family | web steps | web substeps | flags |").unwrap();
    writeln!(out, "| --- | --- | ---: | ---: | --- |").unwrap();

    for (case, artifact) in cases.iter().zip(artifacts.iter()) {
        let flags = if artifact.flags.is_empty() {
            "none".to_string()
        } else {
            artifact.flags.join("; ")
        };
        writeln!(
            out,
            "| `{}` | `{}` | {} | {} | {} |",
            case.id, case.family, artifact.step_count, artifact.web_substep_count, flags
        )
        .unwrap();
    }

    for (case, artifact) in cases.iter().zip(artifacts.iter()) {
        writeln!(out).unwrap();
        writeln!(out, "## {} ({})", case.id, case.family).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "- Source: `{}`", case.source).unwrap();
        writeln!(out, "- Target: `{}`", case.target).unwrap();
        writeln!(out, "- Result: `{}`", artifact.result).unwrap();
        writeln!(out, "- Web step count: `{}`", artifact.step_count).unwrap();
        writeln!(out, "- Web substep count: `{}`", artifact.web_substep_count).unwrap();
        if artifact.flags.is_empty() {
            writeln!(out, "- Flags: none").unwrap();
        } else {
            writeln!(out, "- Flags: {}", artifact.flags.join("; ")).unwrap();
        }

        writeln!(out).unwrap();
        writeln!(out, "### CLI").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "```text").unwrap();
        for line in &artifact.cli_lines {
            writeln!(out, "{line}").unwrap();
        }
        writeln!(out, "```").unwrap();

        writeln!(out).unwrap();
        writeln!(out, "### Web / JSON Steps").unwrap();
        writeln!(out).unwrap();

        for (index, step) in artifact.json_steps.iter().enumerate() {
            let rule = step.get("rule").and_then(Value::as_str).unwrap_or("");
            let before = step.get("before").and_then(Value::as_str).unwrap_or("");
            let after = step.get("after").and_then(Value::as_str).unwrap_or("");
            writeln!(out, "{}. `{}`", index + 1, rule).unwrap();
            writeln!(out, "   - before: `{}`", before).unwrap();
            writeln!(out, "   - after: `{}`", after).unwrap();

            if let Some(substeps) = step.get("substeps").and_then(Value::as_array) {
                if substeps.is_empty() {
                    writeln!(out, "   - substeps: none").unwrap();
                } else {
                    writeln!(out, "   - substeps:").unwrap();
                    for (sub_index, substep) in substeps.iter().enumerate() {
                        let title = substep.get("title").and_then(Value::as_str).unwrap_or("");
                        writeln!(out, "     {}. `{}`", sub_index + 1, title).unwrap();
                    }
                }
            } else {
                writeln!(out, "   - substeps: none").unwrap();
            }
        }
    }

    out
}

#[test]
fn derive_didactic_audit_corpus_is_unique_and_nonempty() {
    let cases = load_derive_cases();
    assert!(
        !cases.is_empty(),
        "derive didactic audit corpus must not be empty"
    );

    let mut ids = HashSet::new();
    for case in cases
        .into_iter()
        .filter(|case| case.expected_status == "derived")
    {
        assert!(
            ids.insert(case.id.clone()),
            "duplicate derive didactic audit case id: {}",
            case.id
        );
    }
}

#[test]
fn derive_didactic_cases_render_steps_without_redundant_single_substeps() {
    let cases: Vec<_> = load_derive_cases()
        .into_iter()
        .filter(|case| case.expected_status == "derived")
        .collect();

    for case in &cases {
        let artifact = audit_case(case);
        assert!(
            artifact.step_count > 0,
            "derive audit case {} should emit web steps",
            case.id
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "redundant single substep duplicates parent step"),
            "derive audit case {} has redundant single-substep duplication; cli=\n{}\nflags={:?}",
            case.id,
            artifact.cli_lines.join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "no web substeps emitted"),
            "derive audit case {} emitted no web substeps; cli=\n{}\nflags={:?}",
            case.id,
            artifact.cli_lines.join("\n"),
            artifact.flags
        );
    }
}

#[test]
fn derive_didactic_factorial_ratio_explains_expand_then_cancel() {
    let artifact = audit_case(&derive_case_by_id("consecutive_factorial_ratio"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Cancelar factoriales consecutivos")
        })
        .expect("expected factorial ratio derive step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected factorial derive substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Escribir el factorial superior como el siguiente número por el factorial anterior",
            "Cancelar el factorial común"
        ]
    );
}

#[test]
fn derive_didactic_perfect_square_factorization_explains_pattern() {
    let artifact = audit_case(&derive_case_by_id("factor_perfect_square_trinomial"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected perfect-square factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Usar a^2 + 2ab + b^2 = (a + b)^2", "Aquí a = x y b = 1"]
    );
}

#[test]
fn derive_didactic_symbolic_perfect_square_factorization_explains_pattern() {
    let artifact = audit_case(&derive_case_by_id(
        "factor_perfect_square_trinomial_symbolic",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected symbolic perfect-square factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected symbolic perfect-square factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Usar a^2 + 2ab + b^2 = (a + b)^2", "Aquí a = a y b = b"]
    );
}

#[test]
fn derive_didactic_symbolic_binomial_cube_factorization_explains_pattern() {
    let artifact = audit_case(&derive_case_by_id("factor_symbolic_binomial_cube"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected symbolic cubic factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected symbolic cubic factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Usar a^3 + 3a^2b + 3ab^2 + b^3 = (a + b)^3",
            "Aquí a = a y b = b"
        ]
    );
}

#[test]
fn derive_didactic_negative_perfect_square_factorization_explains_pattern() {
    let artifact = audit_case(&derive_case_by_id("factor_perfect_square_trinomial_minus"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected negative perfect-square factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected negative perfect-square factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Usar a^2 - 2ab + b^2 = (a - b)^2", "Aquí a = a y b = b"]
    );
}

#[test]
fn derive_didactic_negative_symbolic_binomial_cube_factorization_explains_pattern() {
    let artifact = audit_case(&derive_case_by_id("factor_symbolic_binomial_cube_minus"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected negative symbolic cubic factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected negative symbolic cubic factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Usar a^3 - 3a^2b + 3ab^2 - b^3 = (a - b)^3",
            "Aquí a = a y b = b"
        ]
    );
}

#[test]
fn derive_didactic_geometric_difference_factorization_explains_series_identity() {
    let artifact = audit_case(&derive_case_by_id("factor_geometric_difference_power_6"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected geometric difference factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected geometric difference factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Usar a^n - 1 = (a - 1) · (a^(n-1) + a^(n-2) + ... + a + 1)",
            "Aquí a = x y n = 6"
        ]
    );
}

#[test]
fn derive_didactic_sophie_germain_factorization_explains_identity() {
    let artifact = audit_case(&derive_case_by_id("factor_sophie_germain"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected Sophie Germain factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Usar a^4 + 4b^4 = (a^2 - 2ab + 2b^2) · (a^2 + 2ab + 2b^2)",
            "Aquí a = x y b = y"
        ]
    );
}

#[test]
fn derive_didactic_sophie_germain_expansion_explains_identity() {
    let artifact = audit_case(&derive_case_by_id("expand_sophie_germain"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir la expresión")
        })
        .expect("expected Sophie Germain expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected Sophie Germain expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Usar (a^2 - 2ab + 2b^2) · (a^2 + 2ab + 2b^2) = a^4 + 4b^4",
            "Sustituir a = x y b = y"
        ]
    );
}

#[test]
fn derive_didactic_difference_of_cubes_expansion_explains_identity() {
    let artifact = audit_case(&derive_case_by_id("expand_difference_cubes"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir la expresión")
        })
        .expect("expected difference-of-cubes expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected difference-of-cubes expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Reconocer el patrón (a - b)(a^2 + ab + b^2)",
            "Aplicar (a - b)(a^2 + ab + b^2) = a^3 - b^3"
        ]
    );
}

#[test]
fn derive_didactic_sum_of_cubes_expansion_explains_identity() {
    let artifact = audit_case(&derive_case_by_id("expand_sum_cubes"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir la expresión")
        })
        .expect("expected sum-of-cubes expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sum-of-cubes expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Reconocer el patrón (a + b)(a^2 - ab + b^2)",
            "Aplicar (a + b)(a^2 - ab + b^2) = a^3 + b^3"
        ]
    );
}

#[test]
fn derive_didactic_negative_binomial_expansion_explains_the_minus_identity() {
    let artifact = audit_case(&derive_case_by_id("expand_symbolic_binomial_minus"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir binomio")
        })
        .expect("expected negative binomial expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected negative binomial expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Usar (a - b)^2 = a^2 - 2ab + b^2",
            "Sustituir a = a y b = b"
        ]
    );
}

#[test]
fn derive_didactic_symbolic_binomial_cube_expansion_explains_the_identity() {
    let artifact = audit_case(&derive_case_by_id("expand_symbolic_binomial_cube"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir binomio")
        })
        .expect("expected symbolic cubic expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected symbolic cubic expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Usar (a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3",
            "Sustituir a = a y b = b"
        ]
    );
}

#[test]
fn derive_didactic_negative_symbolic_binomial_cube_expansion_explains_the_identity() {
    let artifact = audit_case(&derive_case_by_id("expand_symbolic_binomial_cube_minus"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir binomio")
        })
        .expect("expected negative symbolic cubic expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected negative symbolic cubic expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Usar (a - b)^3 = a^3 - 3a^2b + 3ab^2 - b^3",
            "Sustituir a = a y b = b"
        ]
    );
}

#[test]
fn derive_didactic_alternating_cubic_vandermonde_factorization_explains_zero_factors_then_linear_part(
) {
    let artifact = audit_case(&derive_case_by_id("factor_alternating_cubic_vandermonde"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected alternating cubic Vandermonde factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected alternating cubic Vandermonde factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Si dos variables coinciden, la expresión vale 0",
            "El factor restante es lineal y simétrico"
        ]
    );
}

#[test]
fn derive_didactic_difference_of_cubes_factorization_explains_identity() {
    let artifact = audit_case(&derive_case_by_id("factor_difference_cubes"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected difference-of-cubes factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected difference-of-cubes factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Reconocer la forma a^3 - b^3",
            "Aplicar a^3 - b^3 = (a - b)(a^2 + ab + b^2)"
        ]
    );
}

#[test]
fn derive_didactic_sum_of_cubes_factorization_explains_identity() {
    let artifact = audit_case(&derive_case_by_id("factor_sum_cubes"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected sum-of-cubes factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sum-of-cubes factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Reconocer la forma a^3 + b^3",
            "Aplicar a^3 + b^3 = (a + b)(a^2 - ab + b^2)"
        ]
    );
}

#[test]
fn derive_didactic_pythagorean_identity_uses_human_step_title() {
    let artifact = audit_case(&derive_case_by_id("pythagorean_identity"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar la identidad pitagórica")
        })
        .expect("expected pythagorean identity step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected pythagorean identity substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Usar sin²(u) + cos²(u) = 1",
            "Aquí seno y coseno tienen el mismo ángulo"
        ]
    );
}

#[test]
fn derive_didactic_pythagorean_factor_form_to_cos_sq_uses_human_identity_step() {
    let artifact = audit_case(&derive_case_by_id("pythagorean_factor_form_to_cos_sq"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad pitagórica")
        })
        .expect("expected pythagorean factor form step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected pythagorean factor form substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar 1 - sin²(u) = cos²(u)"]);
}

#[test]
fn derive_didactic_reverse_pythagorean_factor_form_uses_human_identity_step() {
    let artifact = audit_case(&derive_case_by_id("pythagorean_factor_form_from_sin_sq"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad pitagórica")
        })
        .expect("expected reverse pythagorean factor form step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected reverse pythagorean factor form substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar 1 - cos²(u) = sin²(u)"]);
}

#[test]
fn derive_didactic_sec_squared_contraction_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_sec_squared"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Reconocer secante cuadrada")
        })
        .expect("expected sec squared contraction step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sec squared contraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar 1 + tan²(u) = sec²(u)"]);
}

#[test]
fn derive_didactic_sec_squared_expansion_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("expand_trig_sec_squared"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir secante cuadrada")
        })
        .expect("expected sec squared expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sec squared expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar sec²(u) = 1 + tan²(u)"]);
}

#[test]
fn derive_didactic_secant_reciprocal_expansion_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("expand_trig_sec_reciprocal"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad trigonométrica recíproca")
        })
        .expect("expected reciprocal trig step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected reciprocal trig substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar sec(u) = 1 / cos(u)"]);
}

#[test]
fn derive_didactic_reciprocal_cosine_contraction_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_sec_reciprocal"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad trigonométrica recíproca")
        })
        .expect("expected reciprocal trig step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected reciprocal trig substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar 1 / cos(u) = sec(u)"]);
}

#[test]
fn derive_didactic_cotangent_quotient_contraction_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_cot_quotient"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad trigonométrica recíproca")
        })
        .expect("expected reciprocal trig step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected reciprocal trig substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar cos(u) / sin(u) = cot(u)"]);
}

#[test]
fn derive_didactic_reciprocal_trig_product_to_one_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("reciprocal_trig_product_to_one"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Cancelar funciones trigonométricas recíprocas")
        })
        .expect("expected reciprocal product step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected reciprocal product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar tan(u) · cot(u) = 1"]);
}

#[test]
fn derive_didactic_sec_tan_pythagorean_to_one_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("sec_tan_pythagorean_to_one"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad pitagórica recíproca")
        })
        .expect("expected reciprocal pythagorean step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected reciprocal pythagorean substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar sec²(u) - tan²(u) = 1"]);
}

#[test]
fn derive_didactic_csc_cot_pythagorean_to_one_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("csc_cot_pythagorean_to_one"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad pitagórica recíproca")
        })
        .expect("expected reciprocal pythagorean step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected reciprocal pythagorean substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar csc²(u) - cot²(u) = 1"]);
}

#[test]
fn derive_didactic_half_angle_sin_square_expansion_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("expand_trig_half_angle_sin_squared"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad de ángulo mitad")
        })
        .expect("expected half-angle sine-square expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected half-angle sine-square expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar sin²(u) = (1 - cos(2u)) / 2"]);
}

#[test]
fn derive_didactic_half_angle_cos_square_contraction_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_half_angle_cos_squared"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad de ángulo mitad")
        })
        .expect("expected half-angle cosine-square contraction step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected half-angle cosine-square contraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar (1 + cos(2u)) / 2 = cos²(u)"]);
}

#[test]
fn derive_didactic_inverse_tan_identity_uses_exact_angle_language() {
    let artifact = audit_case(&derive_case_by_id("inverse_tan_identity"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad de arctangentes")
        })
        .expect("expected inverse tangent identity step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected inverse tangent identity substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Usar arctan(u) + arctan(1/u) = pi/2",
            "Esa pareja vale pi/2"
        ]
    );
}

#[test]
fn derive_didactic_trig_quotient_explains_identities_then_tangent() {
    let artifact = audit_case(&derive_case_by_id(
        "contract_trig_cos_diff_sin_diff_quotient",
    ));

    let trig_steps: Vec<_> = artifact
        .json_steps
        .iter()
        .filter(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Convertir un cociente trigonométrico en tangente")
        })
        .collect();

    assert_eq!(
        trig_steps.len(),
        3,
        "expected the derive case to keep the three trig quotient stages"
    );

    let first_titles: Vec<&str> = trig_steps[0]
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected numerator rewrite substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        first_titles,
        vec!["Usar cos(A) - cos(B) = 2 · sin((A+B)/2) · sin((B-A)/2)"]
    );

    let second_titles: Vec<&str> = trig_steps[1]
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected denominator rewrite substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        second_titles,
        vec!["Usar sin(B) - sin(A) = 2 · cos((A+B)/2) · sin((B-A)/2)"]
    );

    let third_titles: Vec<&str> = trig_steps[2]
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected tangent contraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        third_titles,
        vec![
            "Cancelar el factor común del numerador y del denominador",
            "Reconocer el patrón sin(u) / cos(u) = tan(u)"
        ]
    );
}

#[test]
fn derive_didactic_special_sine_difference_uses_sum_to_product_directly() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_sin_diff_special"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar suma a producto")
        })
        .expect("expected sum-to-product step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sum-to-product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Usar sin(A) - sin(B) = 2 · cos((A+B)/2) · sin((A-B)/2)"]
    );
}

#[test]
fn derive_didactic_special_sine_sum_uses_sum_to_product_directly() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_sin_sum_special"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar suma a producto")
        })
        .expect("expected sum-to-product step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sum-to-product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar sin(A) + sin(B) = 2 · sin((A+B)/2) · cos((A-B)/2)"]
    );
}

#[test]
fn derive_didactic_special_cosine_sum_uses_sum_to_product_directly() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_cos_sum_special"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar suma a producto")
        })
        .expect("expected sum-to-product step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sum-to-product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar cos(A) + cos(B) = 2 · cos((A+B)/2) · cos((A-B)/2)"]
    );
}

#[test]
fn derive_didactic_general_cosine_sum_uses_sum_to_product_directly() {
    let artifact = audit_case(&derive_case_by_id(
        "expand_trig_sum_to_product_cos_sum_general",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar suma a producto")
        })
        .expect("expected sum-to-product step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sum-to-product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar cos(A) + cos(B) = 2 · cos((A+B)/2) · cos((A-B)/2)"]
    );
}

#[test]
fn derive_didactic_general_cosine_difference_uses_sum_to_product_directly() {
    let artifact = audit_case(&derive_case_by_id(
        "expand_trig_sum_to_product_cos_diff_general",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar suma a producto")
        })
        .expect("expected sum-to-product step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sum-to-product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar cos(A) - cos(B) = -2 · sin((A+B)/2) · sin((A-B)/2)"]
    );
}

#[test]
fn derive_didactic_difference_of_squares_fraction_cancel_recaps_factor_then_cancel() {
    let artifact = audit_case(&derive_case_by_id("cancel_fraction_difference_squares"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar una diferencia de cuadrados y cancelar")
        })
        .expect("expected difference-of-squares fraction cancel step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected difference-of-squares cancellation substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Reescribir el numerador como diferencia de cuadrados",
            "Ahora se cancela el factor a - b"
        ]
    );
}

#[test]
fn derive_didactic_difference_of_squares_fraction_cancel_mirror_recaps_factor_then_cancel() {
    let artifact = audit_case(&derive_case_by_id(
        "cancel_fraction_difference_squares_mirror",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar una diferencia de cuadrados y cancelar")
        })
        .expect("expected mirrored difference-of-squares fraction cancel step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected mirrored difference-of-squares cancellation substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Reescribir el numerador como diferencia de cuadrados",
            "Ahora se cancela el factor a + b"
        ]
    );
}

#[test]
fn derive_didactic_difference_of_cubes_fraction_cancel_recaps_factor_then_cancel() {
    let artifact = audit_case(&derive_case_by_id("cancel_fraction_difference_cubes"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar cubos y cancelar")
        })
        .expect("expected difference-of-cubes fraction cancel step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected difference-of-cubes cancellation substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Factorizar el numerador como suma o diferencia de cubos",
            "Ahora se cancela el factor (a - b)"
        ]
    );
}

#[test]
fn derive_didactic_sum_of_cubes_fraction_cancel_recaps_factor_then_cancel() {
    let artifact = audit_case(&derive_case_by_id("cancel_fraction_sum_cubes"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar cubos y cancelar")
        })
        .expect("expected sum-of-cubes fraction cancel step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sum-of-cubes cancellation substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Factorizar el numerador como suma o diferencia de cubos",
            "Ahora se cancela el factor (a + b)"
        ]
    );
}

#[test]
fn derive_didactic_perfect_square_fraction_cancel_plus_uses_repeated_factor_rule() {
    let artifact = audit_case(&derive_case_by_id("cancel_fraction_perfect_square_plus"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Cancelar factores en una fracción")
        })
        .expect("expected perfect-square fraction cancel step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected perfect-square cancellation substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Si x + 1 aparece dos veces arriba y una abajo, queda una sola copia"]
    );
}

#[test]
fn derive_didactic_perfect_square_fraction_cancel_minus_numeric_uses_repeated_factor_rule() {
    let artifact = audit_case(&derive_case_by_id(
        "cancel_fraction_perfect_square_minus_numeric",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Pre-order Perfect Square Minus Cancel")
        })
        .expect("expected negative perfect-square fraction cancel step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected negative perfect-square cancellation substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Reconocer que el numerador es un cuadrado perfecto",
            "Si (x - 1)^2 está dividido entre x - 1, queda una sola copia",
        ]
    );
}

#[test]
fn derive_didactic_perfect_square_fraction_cancel_minus_symbolic_uses_repeated_factor_rule() {
    let artifact = audit_case(&derive_case_by_id(
        "cancel_fraction_perfect_square_minus_symbolic",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Pre-order Perfect Square Minus Cancel")
        })
        .expect("expected symbolic negative perfect-square fraction cancel step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected symbolic negative perfect-square cancellation substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec![
            "Reconocer que el numerador es un cuadrado perfecto",
            "Si (a - b)^2 está dividido entre a - b, queda una sola copia",
        ]
    );
}

#[test]
fn derive_didactic_log_quotient_expansion_uses_quotient_language() {
    let artifact = audit_case(&derive_case_by_id("expand_log_quotient"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir logaritmos")
        })
        .expect("expected log expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected log quotient expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Usar que el logaritmo de un cociente se separa en una resta"]
    );
}

#[test]
fn derive_didactic_even_power_log_expansion_uses_absolute_value_language() {
    let artifact = audit_case(&derive_case_by_id("expand_log_even_power_abs"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Sacar un exponente fuera del logaritmo")
        })
        .expect("expected even-power log expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected even-power log expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Sacar un exponente par fuera del logaritmo"]);
}

#[test]
fn derive_didactic_general_base_log_power_expansion_pulls_exponent_out() {
    let artifact = audit_case(&derive_case_by_id("expand_log_general_base_power"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Sacar un exponente fuera del logaritmo")
        })
        .expect("expected general-base log power expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected general-base log power expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Sacar el exponente fuera del logaritmo"]);
}

#[test]
fn derive_didactic_even_power_log_contraction_pushes_coefficient_inside() {
    let artifact = audit_case(&derive_case_by_id("contract_log_even_power_abs"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Meter el coeficiente dentro del logaritmo")
        })
        .expect("expected even-power log contraction step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected even-power log contraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar n · ln(|u|) = ln(u^n) cuando n es par"]);
}

#[test]
fn derive_didactic_general_base_log_power_contraction_pushes_coefficient_inside() {
    let artifact = audit_case(&derive_case_by_id("contract_log_general_base_power"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Meter el coeficiente dentro del logaritmo")
        })
        .expect("expected general-base log power contraction step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected general-base log power contraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar n · log_b(u) = log_b(u^n)"]);
}

#[test]
fn derive_didactic_general_base_log_quotient_contraction_uses_quotient_language() {
    let artifact = audit_case(&derive_case_by_id("contract_log_general_base_difference"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Contraer logaritmos")
        })
        .expect("expected general-base log contraction step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected general-base log quotient contraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Usar que una resta de logaritmos se puede reunir en un cociente"]
    );
}

#[test]
fn derive_didactic_scaled_log_sum_contraction_uses_power_product_language() {
    let artifact = audit_case(&derive_case_by_id("contract_log_sum_with_scaled_powers"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Contraer logaritmos")
        })
        .expect("expected scaled log contraction step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected scaled log contraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Meter los coeficientes dentro de los logaritmos como exponentes"]
    );
}

#[test]
fn derive_didactic_scaled_general_base_log_difference_contraction_uses_power_quotient_language() {
    let artifact = audit_case(&derive_case_by_id(
        "contract_log_general_base_difference_with_scaled_powers",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Contraer logaritmos")
        })
        .expect("expected scaled general-base log contraction step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected scaled general-base log contraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Meter los coeficientes dentro de los logaritmos y reunir la resta en un cociente"]
    );
}

#[test]
fn derive_didactic_log_change_of_base_chain_contraction_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("contract_log_change_of_base_chain"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Contraer cadena de logaritmos")
        })
        .expect("expected log change-of-base contraction step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected log change-of-base contraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar log_b(a) · log_a(c) = log_b(c)"]);
}

#[test]
fn derive_didactic_log_change_of_base_chain_expansion_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("expand_log_change_of_base_chain"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir cambio de base")
        })
        .expect("expected log change-of-base expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected log change-of-base expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar log_b(c) = log_a(c) · log_b(a)"]);
}

#[test]
fn derive_didactic_three_link_log_change_of_base_chain_contraction_uses_chain_language() {
    let artifact = audit_case(&derive_case_by_id(
        "contract_log_change_of_base_chain_three",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Contraer cadena de logaritmos")
        })
        .expect("expected three-link log change-of-base contraction step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected three-link log change-of-base contraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Encadenar los cambios de base intermedios"]);
}

#[test]
fn derive_didactic_three_link_log_change_of_base_chain_expansion_uses_chain_language() {
    let artifact = audit_case(&derive_case_by_id("expand_log_change_of_base_chain_three"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir cambio de base")
        })
        .expect("expected three-link log change-of-base expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected three-link log change-of-base expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Desplegar un logaritmo en una cadena de cambios de base"]
    );
}

#[test]
fn derive_didactic_cos_double_angle_expansion_keeps_an_explicit_identity_step() {
    let artifact = audit_case(&derive_case_by_id(
        "expand_trig_double_cos_as_one_minus_sin_sq",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir ángulo doble")
                && step
                    .get("after")
                    .and_then(Value::as_str)
                    .is_some_and(|after| after.contains("1 - 2"))
        })
        .expect("expected double-angle cosine expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar la identidad de ángulo doble"]);
}

#[test]
fn derive_didactic_cos_double_angle_contraction_explains_the_additive_pattern() {
    let artifact = audit_case(&derive_case_by_id(
        "contract_trig_double_cos_from_one_minus_sin_sq",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| {
                    rule == "Contraer ángulo doble" || rule == "Expandir ángulo doble"
                })
        })
        .expect("expected additive cosine contraction step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected contraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Reconocer el patrón 1 - 2 · sin(u)^2 = cos(2u)"]
    );
}

#[test]
fn derive_didactic_half_angle_tangent_uses_the_direct_identity() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_half_angle_tangent"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad de tangente de ángulo mitad")
        })
        .expect("expected half-angle tangent step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected half-angle tangent substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar (1 - cos(2u)) / sin(2u) = tan(u)"]);
}

#[test]
fn derive_didactic_half_angle_tangent_alt_uses_the_direct_identity() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_half_angle_tangent_alt"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad de tangente de ángulo mitad")
        })
        .expect("expected half-angle tangent alternative step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected half-angle tangent alternative substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar sin(2u) / (1 + cos(2u)) = tan(u)"]);
}

#[test]
fn derive_didactic_half_angle_tangent_expansion_uses_the_direct_identity() {
    let artifact = audit_case(&derive_case_by_id("expand_trig_half_angle_tangent"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad de tangente de ángulo mitad")
        })
        .expect("expected half-angle tangent expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected half-angle tangent expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar (1 - cos(2u)) / sin(2u) = tan(u)"]);
}

#[test]
fn derive_didactic_half_angle_tangent_alt_expansion_uses_the_direct_identity() {
    let artifact = audit_case(&derive_case_by_id("expand_trig_half_angle_tangent_alt"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad de tangente de ángulo mitad")
        })
        .expect("expected half-angle tangent alternative expansion step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected half-angle tangent alternative expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar sin(2u) / (1 + cos(2u)) = tan(u)"]);
}

#[test]
fn derive_didactic_same_base_fractional_power_merge_explains_exponent_sum() {
    let artifact = audit_case(&derive_case_by_id("merge_same_base_fractional_powers"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Sumar exponentes de la misma base")
        })
        .expect("expected same-base power merge step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected power merge substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(titles, vec!["Usar x^a · x^b = x^(a+b)"]);
}

#[test]
fn derive_didactic_mixed_root_and_power_merge_explains_root_then_exponent_sum() {
    let artifact = audit_case(&derive_case_by_id("merge_mixed_root_and_power"));

    let root_step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Reescribir la raíz como potencia fraccionaria")
        })
        .expect("expected root canonicalization step");
    let root_titles: Vec<&str> = root_step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected root canonicalization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(root_titles, vec!["Usar sqrt(u) = u^(1/2)"]);

    let merge_step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Sumar exponentes de la misma base")
        })
        .expect("expected power merge step");
    let merge_titles: Vec<&str> = merge_step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected power merge substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(merge_titles, vec!["Usar x^a · x^b = x^(a+b)"]);
}

#[test]
fn derive_didactic_radical_notable_quotient_recognizes_pattern_then_cleans_power() {
    let artifact = audit_case(&derive_case_by_id("radical_notable_quotient"));

    let notable_step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Reconocer un cociente notable")
        })
        .expect("expected notable quotient step");
    let notable_titles: Vec<&str> = notable_step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected notable quotient substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        notable_titles,
        vec![
            "Llamar t = sqrt(x) para reconocer la forma",
            "Ese cociente notable se convierte en t^2 + t + 1",
            "Volver a poner t = sqrt(x)"
        ]
    );

    let cleanup_step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Deshacer raíz y potencia")
        })
        .expect("expected reciprocal-exponent cleanup step");
    let cleanup_titles: Vec<&str> = cleanup_step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected cleanup substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        cleanup_titles,
        vec![
            "El cuadrado deshace la raíz",
            "Reemplazar ese bloque en la expresión"
        ]
    );
}

#[test]
fn derive_didactic_perfect_square_root_explains_square_then_absolute_value() {
    let artifact = audit_case(&derive_case_by_id("perfect_square_root_to_abs"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Reconocer un cuadrado perfecto bajo la raíz")
        })
        .expect("expected perfect-square root derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected perfect-square root substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Reescribir el radicando como un cuadrado perfecto",
            "La raíz de un cuadrado da un valor absoluto"
        ]
    );
}

#[test]
fn derive_didactic_rationalized_difference_zero_keeps_conjugate_then_direct_cancel() {
    let artifact = audit_case(&derive_case_by_id("rationalize_then_cancel_to_zero"));

    let rationalize_step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Racionalizar el denominador")
        })
        .expect("expected rationalization derive step");
    let rationalize_titles: Vec<&str> = rationalize_step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected rationalization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        rationalize_titles,
        vec![
            "Cambiar el signo para formar el conjugado",
            "Multiplicar numerador y denominador por ese conjugado",
            "En el denominador aparece una diferencia de cuadrados"
        ]
    );

    let cancel_step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Restar dos expresiones iguales")
        })
        .expect("expected self-cancel derive step");
    let cancel_substeps = cancel_step
        .get("substeps")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    assert!(
        cancel_substeps.is_empty(),
        "self-cancel derive step should stay direct without tautological substeps"
    );
}

#[test]
fn derive_didactic_polynomial_cancel_case_expands_then_cancels_pairs() {
    let artifact = audit_case(&derive_case_by_id("expand_then_cancel_to_square"));

    let expand_step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir binomio")
        })
        .expect("expected binomial expansion step");
    let expand_titles: Vec<&str> = expand_step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected binomial expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        expand_titles,
        vec![
            "Usar (a + b)^2 = a^2 + 2ab + b^2",
            "Sustituir a = a y b = b"
        ]
    );
}

#[test]
fn derive_didactic_nested_fraction_sum_explains_common_denominator_then_inversion() {
    let artifact = audit_case(&derive_case_by_id("nested_fraction_one_over_sum"));

    let add_step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Sumar fracciones")
        })
        .expect("expected add-fractions derive step");
    let add_titles: Vec<&str> = add_step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected add-fractions substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        add_titles,
        vec![
            "Llevar ambas fracciones al mismo denominador",
            "Juntar todo en una sola fracción"
        ]
    );

    let nested_step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Simplificar fracción anidada")
        })
        .expect("expected nested-fraction simplify step");
    let nested_titles: Vec<&str> = nested_step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected nested-fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        nested_titles,
        vec!["Dividir entre una fracción equivale a invertirla"]
    );
}

#[test]
fn derive_didactic_same_denominator_focus_only_combines_fraction_part() {
    let artifact = audit_case(&derive_case_by_id(
        "combine_fraction_part_with_same_denominator",
    ));

    let combine_step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Sumar fracciones con mismo denominador")
        })
        .expect("expected same-denominator step");
    let titles: Vec<&str> = combine_step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected same-denominator substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Como el denominador ya es el mismo, se mantiene igual",
            "Basta sumar los numeradores"
        ]
    );
}

#[test]
fn derive_didactic_same_denominator_focus_only_combines_three_fraction_part() {
    let artifact = audit_case(&derive_case_by_id(
        "combine_fraction_part_with_same_denominator_three_terms",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Sumar fracciones con mismo denominador")
        })
        .expect("expected same-denominator combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected same-denominator substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Como el denominador ya es el mismo, se mantiene igual",
            "Basta sumar los numeradores"
        ]
    );
}

#[test]
fn derive_didactic_consecutive_telescoping_fraction_uses_direct_partial_fraction_identity() {
    let artifact = audit_case(&derive_case_by_id("split_telescoping_fraction_consecutive"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)", "Aquí u = n"]
    );
}

#[test]
fn derive_didactic_gap_two_telescoping_fraction_uses_gap_identity_language() {
    let artifact = audit_case(&derive_case_by_id("split_telescoping_fraction_gap_two"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected gap-two telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected gap-two telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
            "Aquí u = n y k = 2"
        ]
    );
}

#[test]
fn derive_didactic_gap_three_telescoping_fraction_uses_gap_identity_language() {
    let artifact = audit_case(&derive_case_by_id("split_telescoping_fraction_gap_three"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected gap-three telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected gap-three telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
            "Aquí u = n y k = 3"
        ]
    );
}

#[test]
fn derive_didactic_negative_consecutive_telescoping_fraction_uses_shifted_base_language() {
    let artifact = audit_case(&derive_case_by_id(
        "split_telescoping_fraction_negative_consecutive",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected negative consecutive telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected negative consecutive telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)",
            "Aquí u = n - 1"
        ]
    );
}

#[test]
fn derive_didactic_negative_gap_two_telescoping_fraction_uses_shifted_base_and_gap_language() {
    let artifact = audit_case(&derive_case_by_id(
        "split_telescoping_fraction_negative_gap_two",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected negative gap-two telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected negative gap-two telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
            "Aquí u = n - 2 y k = 2"
        ]
    );
}

#[test]
fn derive_didactic_affine_gap_two_telescoping_fraction_tracks_affine_base_and_gap_language() {
    let artifact = audit_case(&derive_case_by_id(
        "split_telescoping_fraction_affine_gap_two",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected affine gap-two telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine gap-two telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
            "Aquí u = 2 · n + 1 y k = 2"
        ]
    );
}

#[test]
fn derive_didactic_affine_shifted_gap_two_telescoping_fraction_tracks_affine_shifted_base_and_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "split_telescoping_fraction_affine_shifted_gap_two",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected affine shifted gap-two telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine shifted gap-two telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
            "Aquí u = 2 · n - 1 y k = 2"
        ]
    );
}

#[test]
fn derive_didactic_affine_coeff_three_gap_three_telescoping_fraction_tracks_affine_base_and_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "split_telescoping_fraction_affine_coeff_three_gap_three",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected affine coeff-three gap-three telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine coeff-three gap-three telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
            "Aquí u = 3 · n + 2 y k = 3"
        ]
    );
}

#[test]
fn derive_didactic_affine_coeff_three_shifted_gap_three_telescoping_fraction_tracks_affine_shifted_base_and_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "split_telescoping_fraction_affine_coeff_three_shifted_gap_three",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected affine coeff-three shifted gap-three telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine coeff-three shifted gap-three telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
            "Aquí u = 3 · n - 1 y k = 3"
        ]
    );
}

#[test]
fn derive_didactic_affine_symbolic_coeff_gap_three_telescoping_fraction_tracks_affine_base_and_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "split_telescoping_fraction_affine_symbolic_coeff_gap_three",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected affine symbolic-coeff gap-three telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine symbolic-coeff gap-three telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
            "Aquí u = a · n + 2 y k = 3"
        ]
    );
}

#[test]
fn derive_didactic_affine_symbolic_coeff_shifted_gap_three_telescoping_fraction_tracks_affine_shifted_base_and_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "split_telescoping_fraction_affine_symbolic_coeff_shifted_gap_three",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected affine symbolic-coeff shifted gap-three telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine symbolic-coeff shifted gap-three telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
            "Aquí u = a · n - 1 y k = 3"
        ]
    );
}

#[test]
fn derive_didactic_symbolic_shift_gap_telescoping_fraction_tracks_symbolic_gap_language() {
    let artifact = audit_case(&derive_case_by_id(
        "split_telescoping_fraction_symbolic_shift_gap",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected symbolic shift-gap telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected symbolic shift-gap telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
            "Aquí u = a + n y k = b - a"
        ]
    );
}

#[test]
fn derive_didactic_affine_symbolic_shift_gap_telescoping_fraction_tracks_affine_symbolic_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "split_telescoping_fraction_affine_symbolic_shift_gap",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Descomponer en fracciones telescópicas")
        })
        .expect("expected affine symbolic shift-gap telescoping fraction step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine symbolic shift-gap telescoping fraction substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
            "Aquí u = a · n + b y k = c - b"
        ]
    );
}

#[test]
fn derive_didactic_consecutive_telescoping_fraction_combine_uses_inverse_identity_language() {
    let artifact = audit_case(&derive_case_by_id(
        "combine_telescoping_fraction_consecutive",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected telescoping fraction combine substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar 1 / u - 1 / (u + 1) = 1 / (u · (u + 1))", "Aquí u = n"]
    );
}

#[test]
fn derive_didactic_gap_two_telescoping_fraction_combine_uses_inverse_gap_identity_language() {
    let artifact = audit_case(&derive_case_by_id("combine_telescoping_fraction_gap_two"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected gap-two telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected gap-two telescoping fraction combine substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
            "Aquí u = n y k = 2"
        ]
    );
}

#[test]
fn derive_didactic_gap_three_telescoping_fraction_combine_uses_inverse_gap_identity_language() {
    let artifact = audit_case(&derive_case_by_id("combine_telescoping_fraction_gap_three"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected gap-three telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected gap-three telescoping fraction combine substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
            "Aquí u = n y k = 3"
        ]
    );
}

#[test]
fn derive_didactic_negative_consecutive_telescoping_fraction_combine_uses_shifted_inverse_identity_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "combine_telescoping_fraction_negative_consecutive",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected negative consecutive telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected negative consecutive telescoping fraction combine substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / u - 1 / (u + 1) = 1 / (u · (u + 1))",
            "Aquí u = n - 1"
        ]
    );
}

#[test]
fn derive_didactic_negative_gap_two_telescoping_fraction_combine_uses_shifted_inverse_gap_identity_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "combine_telescoping_fraction_negative_gap_two",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected negative gap-two telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected negative gap-two telescoping fraction combine substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
            "Aquí u = n - 2 y k = 2"
        ]
    );
}

#[test]
fn derive_didactic_affine_gap_two_telescoping_fraction_combine_tracks_affine_base_and_gap_language()
{
    let artifact = audit_case(&derive_case_by_id(
        "combine_telescoping_fraction_affine_gap_two",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected affine gap-two telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine gap-two telescoping fraction combine substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
            "Aquí u = 2 · n + 1 y k = 2"
        ]
    );
}

#[test]
fn derive_didactic_affine_shifted_gap_two_telescoping_fraction_combine_tracks_affine_shifted_base_and_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "combine_telescoping_fraction_affine_shifted_gap_two",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected affine shifted gap-two telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine shifted gap-two telescoping fraction combine substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
            "Aquí u = 2 · n - 1 y k = 2"
        ]
    );
}

#[test]
fn derive_didactic_affine_coeff_three_gap_three_telescoping_fraction_combine_tracks_affine_base_and_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "combine_telescoping_fraction_affine_coeff_three_gap_three",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected affine coeff-three gap-three telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine coeff-three gap-three telescoping fraction combine substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
            "Aquí u = 3 · n + 2 y k = 3"
        ]
    );
}

#[test]
fn derive_didactic_affine_coeff_three_shifted_gap_three_telescoping_fraction_combine_tracks_affine_shifted_base_and_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "combine_telescoping_fraction_affine_coeff_three_shifted_gap_three",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected affine coeff-three shifted gap-three telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect(
            "expected affine coeff-three shifted gap-three telescoping fraction combine substeps",
        )
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
            "Aquí u = 3 · n - 1 y k = 3"
        ]
    );
}

#[test]
fn derive_didactic_affine_symbolic_coeff_gap_three_telescoping_fraction_combine_tracks_affine_base_and_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "combine_telescoping_fraction_affine_symbolic_coeff_gap_three",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected affine symbolic-coeff gap-three telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine symbolic-coeff gap-three telescoping fraction combine substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
            "Aquí u = a · n + 2 y k = 3"
        ]
    );
}

#[test]
fn derive_didactic_affine_symbolic_coeff_shifted_gap_three_telescoping_fraction_combine_tracks_affine_shifted_base_and_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "combine_telescoping_fraction_affine_symbolic_coeff_shifted_gap_three",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect(
            "expected affine symbolic-coeff shifted gap-three telescoping fraction combine step",
        );
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect(
            "expected affine symbolic-coeff shifted gap-three telescoping fraction combine substeps",
        )
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
            "Aquí u = a · n - 1 y k = 3"
        ]
    );
}

#[test]
fn derive_didactic_symbolic_shift_gap_telescoping_fraction_combine_tracks_symbolic_gap_language() {
    let artifact = audit_case(&derive_case_by_id(
        "combine_telescoping_fraction_symbolic_shift_gap",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected symbolic shift-gap telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected symbolic shift-gap telescoping fraction combine substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
            "Aquí u = a + n y k = b - a"
        ]
    );
}

#[test]
fn derive_didactic_affine_symbolic_shift_gap_telescoping_fraction_combine_tracks_affine_symbolic_gap_language(
) {
    let artifact = audit_case(&derive_case_by_id(
        "combine_telescoping_fraction_affine_symbolic_shift_gap",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Recomponer fracción telescópica")
        })
        .expect("expected affine symbolic shift-gap telescoping fraction combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected affine symbolic shift-gap telescoping fraction combine substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
            "Aquí u = a · n + b y k = c - b"
        ]
    );
}

#[test]
fn derive_didactic_morrie_telescoping_uses_cosine_telescoping_language() {
    let artifact = audit_case(&derive_case_by_id("integrate_prep_morrie_basic"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar telescopado de cosenos")
        })
        .expect("expected Morrie telescoping step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected Morrie telescoping substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(titles, vec!["Usar el telescopado de cosenos", "Aquí u = x"]);
}

#[test]
fn derive_didactic_scaled_morrie_telescoping_tracks_scaled_base_argument() {
    let artifact = audit_case(&derive_case_by_id("integrate_prep_morrie_scaled"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar telescopado de cosenos")
        })
        .expect("expected scaled Morrie telescoping step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected scaled Morrie telescoping substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar el telescopado de cosenos", "Aquí u = 3 · x"]
    );
}

#[test]
fn derive_didactic_dirichlet_kernel_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("integrate_prep_dirichlet_basic"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad del núcleo de Dirichlet")
        })
        .expect("expected Dirichlet kernel step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected Dirichlet kernel substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar el núcleo de Dirichlet", "Aquí n = 2 y u = x"]
    );
}

#[test]
fn derive_didactic_dirichlet_kernel_longer_chain_tracks_n_value() {
    let artifact = audit_case(&derive_case_by_id("integrate_prep_dirichlet_longer"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad del núcleo de Dirichlet")
        })
        .expect("expected longer Dirichlet kernel step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected longer Dirichlet kernel substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar el núcleo de Dirichlet", "Aquí n = 3 y u = x"]
    );
}

#[test]
fn derive_didactic_scaled_dirichlet_kernel_tracks_scaled_argument() {
    let artifact = audit_case(&derive_case_by_id("integrate_prep_dirichlet_scaled"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad del núcleo de Dirichlet")
        })
        .expect("expected scaled Dirichlet kernel step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected scaled Dirichlet kernel substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar el núcleo de Dirichlet", "Aquí n = 2 y u = 3 · x"]
    );
}

#[test]
fn derive_didactic_scaled_dirichlet_kernel_longer_chain_tracks_scaled_argument_and_n() {
    let artifact = audit_case(&derive_case_by_id("integrate_prep_dirichlet_scaled_longer"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad del núcleo de Dirichlet")
        })
        .expect("expected longer scaled Dirichlet kernel step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected longer scaled Dirichlet kernel substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar el núcleo de Dirichlet", "Aquí n = 3 y u = 2 · x"]
    );
}

#[test]
fn derive_didactic_finite_telescoping_product_expands_then_cancels() {
    let artifact = audit_case(&derive_case_by_id("finite_telescoping_product_basic"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Evaluar producto telescópico finito")
        })
        .expect("expected finite telescoping product step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected finite telescoping product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Escribir los primeros y últimos factores del producto",
            "Los factores intermedios se cancelan por parejas",
            "Solo quedan el último numerador y el primer denominador",
        ]
    );
}

#[test]
fn derive_didactic_shifted_finite_telescoping_product_stops_at_the_closed_form_fraction() {
    let artifact = audit_case(&derive_case_by_id("finite_telescoping_product_shifted"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Evaluar producto telescópico finito")
        })
        .expect("expected shifted finite telescoping product step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected shifted finite telescoping product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Escribir los primeros y últimos factores del producto",
            "Los factores intermedios se cancelan por parejas",
            "Solo quedan el último numerador y el primer denominador",
        ]
    );
}

#[test]
fn derive_didactic_finite_telescoping_sum_uses_partial_fraction_then_cancellation_language() {
    let artifact = audit_case(&derive_case_by_id("finite_telescoping_sum_basic"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Evaluar suma telescópica finita")
        })
        .expect("expected finite telescoping sum step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected finite telescoping sum substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)",
            "Aquí u = k",
            "La suma telescópica cancela los términos intermedios",
        ]
    );
}

#[test]
fn derive_didactic_shifted_finite_telescoping_sum_tracks_the_shifted_base() {
    let artifact = audit_case(&derive_case_by_id("finite_telescoping_sum_shifted"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Evaluar suma telescópica finita")
        })
        .expect("expected shifted finite telescoping sum step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected shifted finite telescoping sum substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)",
            "Aquí u = k + 2",
            "La suma telescópica cancela los términos intermedios",
        ]
    );
}

#[test]
fn derive_didactic_three_same_denominator_fractions_keep_denominator_and_sum_numerators() {
    let artifact = audit_case(&derive_case_by_id(
        "combine_three_same_denominator_fractions",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Sumar fracciones con mismo denominador")
        })
        .expect("expected same-denominator combine step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected same-denominator substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Como el denominador ya es el mismo, se mantiene igual",
            "Basta sumar los numeradores"
        ]
    );
}

#[test]
fn derive_didactic_three_term_fraction_distribution_mentions_each_term_in_the_numerator() {
    let artifact = audit_case(&derive_case_by_id(
        "expand_fraction_same_denominator_three_terms",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Repartir el denominador común")
        })
        .expect("expected distribute-division derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected distribute-division substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Repartir el mismo denominador sobre cada término del numerador"]
    );
}

#[test]
fn derive_didactic_same_denominator_focus_only_expands_fraction_part() {
    let artifact = audit_case(&derive_case_by_id(
        "expand_fraction_part_with_same_denominator_three_terms",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Repartir el denominador común")
        })
        .expect("expected distribute-division step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected distribute-division substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Repartir el mismo denominador sobre cada término del numerador"]
    );
}

#[test]
fn derive_didactic_combine_like_terms_with_zero_sums_coefficients_without_noise() {
    let artifact = audit_case(&derive_case_by_id("combine_like_terms_with_zero"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Agrupar términos semejantes")
        })
        .expect("expected combine-like-terms step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected combine-like-terms substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(titles, vec!["Sumar los coeficientes que acompañan a x"]);
}

#[test]
fn derive_didactic_common_factor_factorization_sum_explains_the_shared_factor() {
    let artifact = audit_case(&derive_case_by_id("factor_common_factor_sum"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected factorization derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar el factor común", "Aquí el factor común es a"]
    );
}

#[test]
fn derive_didactic_common_factor_expansion_sum_explains_distributive_law() {
    let artifact = audit_case(&derive_case_by_id("expand_common_factor_sum"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir la expresión")
        })
        .expect("expected expansion derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar la distributiva",
            "Aquí se distribuye a sobre cada término del paréntesis"
        ]
    );
}

#[test]
fn derive_didactic_three_term_common_factor_factorization_explains_shared_factor() {
    let artifact = audit_case(&derive_case_by_id("factor_common_factor_sum_three_terms"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected factorization derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar el factor común", "Aquí el factor común es x"]
    );
}

#[test]
fn derive_didactic_three_term_common_factor_expansion_explains_distributive_law() {
    let artifact = audit_case(&derive_case_by_id(
        "expand_common_factor_difference_three_terms",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir la expresión")
        })
        .expect("expected expansion derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected expansion substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Usar la distributiva",
            "Aquí se distribuye x sobre cada término del paréntesis"
        ]
    );
}

#[test]
fn derive_didactic_numeric_common_factor_fraction_cancels_then_simplifies_remaining_fraction() {
    let artifact = audit_case(&derive_case_by_id("cancel_fraction_common_factor_numeric"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Cancelar factores en una fracción")
        })
        .expect("expected fraction-cancel derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected fraction-cancel substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Cancelar el factor común 2",
            "Simplificar la fracción restante"
        ]
    );
}

#[test]
fn derive_didactic_monomial_common_factor_fraction_cancels_symbol_then_simplifies_coefficients() {
    let artifact = audit_case(&derive_case_by_id("cancel_fraction_monomial_common_factor"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Cancelar factores en una fracción")
        })
        .expect("expected fraction-cancel derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected fraction-cancel substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Cancelar el factor común x",
            "Simplificar la fracción restante"
        ]
    );
}

#[test]
#[ignore]
fn derive_didactic_audit_generates_markdown_report() {
    let cases: Vec<_> = load_derive_cases()
        .into_iter()
        .filter(|case| case.expected_status == "derived")
        .collect();
    let artifacts = cases.iter().map(audit_case).collect::<Vec<_>>();
    let report = build_report(&cases, &artifacts);
    let path = report_output_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create report dir");
    }
    fs::write(&path, report).expect("write derive didactic audit report");
    eprintln!("wrote {}", path.display());
}
