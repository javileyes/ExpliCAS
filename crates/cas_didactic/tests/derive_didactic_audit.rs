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
use std::collections::BTreeMap;
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
    let primary = include_str!("../../cas_solver/tests/derive_pairs.csv");
    let supplemental = include_str!("derive_didactic_cases.csv");
    let generated_audit = include_str!("../../../docs/generated/DERIVE_DIDACTIC_AUDIT.md");
    let mut by_id = BTreeMap::new();

    for csv in [primary, supplemental] {
        for line in csv.lines().skip(1).filter(|line| !line.trim().is_empty()) {
            let parts = split_csv_line(line);
            assert_eq!(parts.len(), 6, "unexpected derive csv columns: {line}");
            by_id
                .entry(parts[0].trim().to_string())
                .or_insert_with(|| DeriveCase {
                    id: parts[0].trim().to_string(),
                    family: parts[1].trim().to_string(),
                    source: parts[2].trim().to_string(),
                    target: parts[3].trim().to_string(),
                    expected_status: parts[4].trim().to_string(),
                });
        }
    }

    let mut current_id = None::<String>;
    let mut current_family = None::<String>;
    let mut current_source = None::<String>;
    let mut current_target = None::<String>;

    let flush_generated_case =
        |by_id: &mut BTreeMap<String, DeriveCase>,
         current_id: &mut Option<String>,
         current_family: &mut Option<String>,
         current_source: &mut Option<String>,
         current_target: &mut Option<String>| {
            let (Some(id), Some(family), Some(source), Some(target)) = (
                current_id.take(),
                current_family.take(),
                current_source.take(),
                current_target.take(),
            ) else {
                return;
            };

            by_id.entry(id.clone()).or_insert(DeriveCase {
                id,
                family,
                source,
                target,
                expected_status: "derived".to_string(),
            });
        };

    for line in generated_audit.lines() {
        if let Some(rest) = line.strip_prefix("## ") {
            flush_generated_case(
                &mut by_id,
                &mut current_id,
                &mut current_family,
                &mut current_source,
                &mut current_target,
            );

            if let Some((id, family)) = rest.rsplit_once(" (") {
                current_id = Some(id.trim().to_string());
                current_family = Some(family.trim_end_matches(')').trim().to_string());
            } else {
                current_id = None;
                current_family = None;
            }
            continue;
        }

        if let Some(source) = line.strip_prefix("- Source: `") {
            current_source = Some(source.trim_end_matches('`').to_string());
            continue;
        }

        if let Some(target) = line.strip_prefix("- Target: `") {
            current_target = Some(target.trim_end_matches('`').to_string());
        }
    }

    flush_generated_case(
        &mut by_id,
        &mut current_id,
        &mut current_family,
        &mut current_source,
        &mut current_target,
    );

    by_id.into_values().collect()
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
    substep_duplicates_parent_snapshot(step, substep) && !is_contextual_substep(substep)
}

fn has_repeated_substep_snapshot(step: &Value) -> bool {
    let Some(substeps) = step.get("substeps").and_then(Value::as_array) else {
        return false;
    };

    let mut seen = std::collections::HashSet::new();
    for substep in substeps {
        let before = substep
            .get("before_latex")
            .and_then(Value::as_str)
            .map(normalize_latex_snapshot)
            .unwrap_or_default();
        let after = substep
            .get("after_latex")
            .and_then(Value::as_str)
            .map(normalize_latex_snapshot)
            .unwrap_or_default();
        if before.is_empty() && after.is_empty() {
            continue;
        }
        if !seen.insert((before, after)) {
            return true;
        }
    }
    false
}

fn normalize_human_label(input: &str) -> String {
    input
        .to_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch.is_whitespace() {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_contextual_substep(substep: &Value) -> bool {
    let title = substep
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let normalized = normalize_human_label(title);
    normalized.starts_with("aquí ") || normalized.starts_with("aqui ")
}

fn substep_duplicates_parent_snapshot(step: &Value, substep: &Value) -> bool {
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

fn has_non_contextual_parent_snapshot_substep(step: &Value) -> bool {
    let Some(substeps) = step.get("substeps").and_then(Value::as_array) else {
        return false;
    };

    substeps.iter().any(|substep| {
        substep_duplicates_parent_snapshot(step, substep) && !is_contextual_substep(substep)
    })
}

fn has_weak_generic_fraction_cleanup_title(step: &Value) -> bool {
    let Some(substeps) = step.get("substeps").and_then(Value::as_array) else {
        return false;
    };

    substeps.iter().any(|substep| {
        matches!(
            substep.get("title").and_then(Value::as_str),
            Some("Simplificar la fracción restante")
                | Some("Simplificar cada fracción resultante por separado")
        )
    })
}

fn has_weak_editorial_substep_title(step: &Value) -> bool {
    let Some(substeps) = step.get("substeps").and_then(Value::as_array) else {
        return false;
    };

    substeps.iter().any(|substep| {
        matches!(
            substep.get("title").and_then(Value::as_str),
            Some("Reescribir el producto ya agrupado")
                | Some("Usar una identidad de tangente de ángulo mitad")
                | Some("Ver que el denominador coincide con un factor del numerador")
                | Some("Queda el cociente exacto del cubo")
        )
    })
}

fn has_noisy_template_substep(step: &Value) -> bool {
    let Some(substeps) = step.get("substeps").and_then(Value::as_array) else {
        return false;
    };

    substeps.iter().any(|substep| {
        matches!(
            substep.get("title").and_then(Value::as_str),
            Some("Dividir entre una fracción equivale a invertirla")
                | Some("Usar 1 / (p / q) = q / p")
                | Some("Usar n / (p / q) = n · q / p")
                | Some("Usar n / (1 / d) = n · d")
                | Some("Combinar términos del numerador (denominador común)")
                | Some("Aquí a = a y b = b")
        ) || matches!(
            (
                substep.get("before_expr").and_then(Value::as_str),
                substep.get("after_expr").and_then(Value::as_str)
            ),
            (Some("1 / (p / q)"), Some("q / p"))
                | (Some("n / (p / q)"), Some("n · q / p"))
                | (Some("n / (1 / d)"), Some("n · d"))
                | (_, Some("(numerador combinado) / B"))
        )
    })
}

fn is_self_explanatory_fraction_rule(rule: &str) -> bool {
    matches!(
        rule,
        "Sumar fracciones"
            | "Restar fracciones"
            | "Sumar fracciones con mismo denominador"
            | "Restar fracciones con mismo denominador"
            | "Llevar a denominador común"
    )
}

fn has_redundant_fraction_rule_substeps(step: &Value) -> bool {
    let Some(rule) = step.get("rule").and_then(Value::as_str) else {
        return false;
    };
    if !is_self_explanatory_fraction_rule(rule) {
        return false;
    }

    let Some(substeps) = step.get("substeps").and_then(Value::as_array) else {
        return false;
    };
    if substeps.is_empty() {
        return false;
    }

    let titles: Vec<&str> = substeps
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    !matches!(
        (rule, titles.as_slice()),
        ("Sumar fracciones", ["Llevar a denominador común"])
            | (
                "Sumar fracciones",
                [
                    "Llevar a denominador común",
                    "Simplificar el numerador y el denominador"
                ]
            )
            | ("Restar fracciones", ["Llevar a denominador común"])
            | (
                "Restar fracciones",
                [
                    "Llevar a denominador común",
                    "Simplificar el numerador y el denominador"
                ]
            )
    )
}

fn is_self_explanatory_identity_rule(rule: &str) -> bool {
    matches!(
        rule,
        "Aplicar identidad pitagórica"
            | "Expandir secante cuadrada"
            | "Expandir cosecante cuadrada"
            | "Reconocer secante cuadrada"
            | "Reconocer cosecante cuadrada"
            | "Aplicar identidad trigonométrica recíproca"
            | "Cancelar funciones trigonométricas recíprocas"
            | "Aplicar identidad pitagórica recíproca"
            | "Aplicar identidad de ángulo mitad"
            | "Aplicar identidad de tangente de ángulo mitad"
            | "Convertir un cociente trigonométrico en tangente"
            | "Expandir logaritmos"
            | "Contraer logaritmos"
            | "Expandir cambio de base"
            | "Contraer cadena de logaritmos"
            | "Expandir ángulo doble"
            | "Contraer ángulo doble"
            | "Reescribir la raíz como potencia fraccionaria"
            | "Sumar exponentes de la misma base"
    )
}

fn case_may_have_zero_substeps(case: &DeriveCase, json_steps: &[Value]) -> bool {
    if case.family == "fraction_combine" {
        return true;
    }
    !json_steps.is_empty()
        && json_steps.iter().all(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| {
                    is_self_explanatory_fraction_rule(rule)
                        || is_self_explanatory_identity_rule(rule)
                        || rule == "Simplificar fracción anidada"
                })
        })
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
    if case.expected_status == "derived"
        && web_substep_count == 0
        && !case_may_have_zero_substeps(case, &json_steps)
    {
        flags.push("no web substeps emitted".to_string());
    }
    if json_steps.iter().any(has_redundant_single_substep) {
        flags.push("redundant single substep duplicates parent step".to_string());
    }
    if json_steps.iter().any(has_repeated_substep_snapshot) {
        flags.push("multiple substeps share the same snapshot inside a step".to_string());
    }
    if json_steps
        .iter()
        .any(has_non_contextual_parent_snapshot_substep)
    {
        flags.push("non-contextual substep duplicates parent snapshot".to_string());
    }
    if json_steps
        .iter()
        .any(has_weak_generic_fraction_cleanup_title)
    {
        flags.push("generic fraction cleanup title remains in substeps".to_string());
    }
    if json_steps.iter().any(has_weak_editorial_substep_title) {
        flags.push("weak editorial substep title remains in derive".to_string());
    }
    if json_steps.iter().any(has_noisy_template_substep) {
        flags.push("noisy template substep remains in derive".to_string());
    }
    if json_steps.iter().any(has_redundant_fraction_rule_substeps) {
        flags.push("self-explanatory fraction rule still emits redundant substeps".to_string());
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

fn step_substep_titles(step: &Value) -> Vec<&str> {
    step.get("substeps")
        .and_then(Value::as_array)
        .map(|substeps| {
            substeps
                .iter()
                .filter_map(|substep| substep.get("title").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default()
}

fn step_by_rule<'a>(artifact: &'a AuditArtifact, rule: &str) -> &'a Value {
    artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|candidate| candidate == rule)
        })
        .unwrap_or_else(|| {
            let available_rules: Vec<&str> = artifact
                .json_steps
                .iter()
                .filter_map(|step| step.get("rule").and_then(Value::as_str))
                .collect();
            panic!("expected derive step with rule `{rule}`; available rules: {available_rules:?}")
        })
}

fn assert_case_step_titles(case_id: &str, rule: &str, expected_titles: &[&str]) {
    let artifact = audit_case(&derive_case_by_id(case_id));
    let step = step_by_rule(&artifact, rule);
    assert_eq!(step_substep_titles(step), expected_titles);
}

fn assert_case_step_has_no_substeps(case_id: &str, rule: &str) {
    let artifact = audit_case(&derive_case_by_id(case_id));
    let step = step_by_rule(&artifact, rule);
    assert!(
        step_substep_titles(step).is_empty(),
        "expected `{case_id}` / `{rule}` to stay direct without substeps"
    );
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
                .any(|flag| flag == "multiple substeps share the same snapshot inside a step"),
            "derive audit case {} has duplicate multi-substep snapshots; cli=\n{}\nflags={:?}",
            case.id,
            artifact.cli_lines.join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "non-contextual substep duplicates parent snapshot"),
            "derive audit case {} still has a non-contextual substep that duplicates the parent snapshot; cli=\n{}\nflags={:?}",
            case.id,
            artifact.cli_lines.join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "generic fraction cleanup title remains in substeps"),
            "derive audit case {} still uses a weak generic fraction cleanup title; cli=\n{}\nflags={:?}",
            case.id,
            artifact.cli_lines.join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "self-explanatory fraction rule still emits redundant substeps"),
            "derive audit case {} still emits redundant fraction-rule substeps; cli=\n{}\nflags={:?}",
            case.id,
            artifact.cli_lines.join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "weak editorial substep title remains in derive"),
            "derive audit case {} still uses a weak editorial substep title; cli=\n{}\nflags={:?}",
            case.id,
            artifact.cli_lines.join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "noisy template substep remains in derive"),
            "derive audit case {} still keeps a noisy template substep; cli=\n{}\nflags={:?}",
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
fn derive_didactic_gap_two_factorial_ratio_explains_expand_then_cancel() {
    let artifact = audit_case(&derive_case_by_id("consecutive_factorial_ratio_gap_two"));

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
            "Expandir el factorial superior hasta llegar al factorial inferior",
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

    assert_eq!(titles, vec!["Usar a^2 + 2ab + b^2 = (a + b)^2"]);
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

    assert_eq!(titles, vec!["Usar a^3 + 3a^2b + 3ab^2 + b^3 = (a + b)^3"]);
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

    assert_eq!(titles, vec!["Usar a^2 - 2ab + b^2 = (a - b)^2"]);
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

    assert_eq!(titles, vec!["Usar a^3 - 3a^2b + 3ab^2 - b^3 = (a - b)^3"]);
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
    assert_case_step_has_no_substeps("expand_sophie_germain", "Expandir la expresión");
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
            "Usar (a + b)^2 = a^2 + 2ab + b^2",
            "Aplicar la fórmula con a = a, b = -b"
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

    assert_eq!(titles, vec!["Usar (a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3"]);
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
            "Usar (a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3",
            "Aplicar la fórmula con a = a, b = -b"
        ]
    );
}

#[test]
fn derive_didactic_symbolic_trinomial_square_expansion_shows_real_intermediate() {
    assert_case_step_has_no_substeps("expand_symbolic_trinomial_square", "Expandir binomio");
}

#[test]
fn derive_didactic_signed_trinomial_square_expansion_shows_real_intermediate() {
    assert_case_step_has_no_substeps(
        "expand_symbolic_signed_trinomial_square",
        "Expandir binomio",
    );
}

#[test]
fn derive_didactic_symbolic_trinomial_cube_expansion_shows_real_intermediate() {
    assert_case_step_has_no_substeps("expand_symbolic_trinomial_cube", "Expandir binomio");
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

    assert_eq!(titles, vec!["Reconocer la forma a^3 - b^3"]);
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

    assert_eq!(titles, vec!["Reconocer la forma a^3 + b^3"]);
}

#[test]
fn derive_didactic_symbolic_sixth_power_difference_factorization_explains_identity() {
    let artifact = audit_case(&derive_case_by_id("factor_symbolic_sixth_power_difference"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected sixth-power difference factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sixth-power difference factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Aplicar a^6 - b^6 = (a^2 - b^2)(a^4 + a^2b^2 + b^4)"]
    );
}

#[test]
fn derive_didactic_symbolic_sixth_power_sum_factorization_explains_identity() {
    let artifact = audit_case(&derive_case_by_id("factor_symbolic_sixth_power_sum"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Factorizar")
        })
        .expect("expected sixth-power sum factorization step");

    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected sixth-power sum factorization substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert_eq!(
        titles,
        vec!["Aplicar a^6 + b^6 = (a^2 + b^2)(a^4 - a^2b^2 + b^4)"]
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
        .into_iter()
        .flatten()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert!(
        titles.is_empty(),
        "expected no tautological substeps, got: {titles:?}"
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
}

#[test]
fn derive_didactic_csc_squared_contraction_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_csc_squared"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Reconocer cosecante cuadrada")
        })
        .expect("expected scaled csc squared contraction step");

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
}

#[test]
fn derive_didactic_secant_reciprocal_expansion_uses_direct_identity_language() {
    let artifact = audit_case(&DeriveCase {
        id: "expand_trig_sec_reciprocal_inline".to_string(),
        family: "trig_expand".to_string(),
        source: "sec(x)".to_string(),
        target: "1/cos(x)".to_string(),
        expected_status: "derived".to_string(),
    });

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad trigonométrica recíproca")
        })
        .expect("expected reciprocal trig step");

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
}

#[test]
fn derive_didactic_tangent_quotient_contraction_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_tan_quotient"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Convertir un cociente trigonométrico en tangente")
        })
        .expect("expected trig quotient step");

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
}

#[test]
fn derive_didactic_csc_cot_pythagorean_to_one_uses_direct_identity_language() {
    let case = DeriveCase {
        id: "csc_cot_pythagorean_to_one_inline".to_string(),
        family: "simplify".to_string(),
        source: "csc(x)^2 - cot(x)^2".to_string(),
        target: "1".to_string(),
        expected_status: "derived".to_string(),
    };
    let artifact = audit_case(&case);

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad pitagórica recíproca")
        })
        .expect("expected reciprocal pythagorean step");

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    assert_eq!(titles, vec!["Usar arctan(u) + arctan(1/u) = pi/2"]);
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
        1,
        "expected the direct trig quotient contraction"
    );
    assert!(step_substep_titles(trig_steps[0]).is_empty());
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
fn derive_didactic_general_sine_sum_uses_sum_to_product_directly() {
    let artifact = audit_case(&derive_case_by_id(
        "expand_trig_sum_to_product_sin_sum_general",
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
        vec!["Usar sin(A) + sin(B) = 2 · sin((A+B)/2) · cos((A-B)/2)"]
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
    assert_case_step_has_no_substeps(
        "cancel_fraction_difference_squares",
        "Factorizar una diferencia de cuadrados y cancelar",
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
    let artifact = audit_case(&DeriveCase {
        id: "cancel_fraction_perfect_square_plus_inline".to_string(),
        family: "simplify".to_string(),
        source: "(x^2+2*x+1)/(x+1)".to_string(),
        target: "x+1".to_string(),
        expected_status: "derived".to_string(),
    });

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
        vec![
            "Reconocer que el numerador es un cuadrado perfecto",
            "Si (x + 1)^2 está dividido entre x + 1, queda una sola copia"
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
fn derive_didactic_representative_direct_log_expansion_cases_have_no_substeps() {
    for case_id in [
        "expand_log_product",
        "expand_log_quotient",
        "expand_log_product_over_quotient",
        "expand_log_powered_two_denominator_factors",
        "expand_log_general_base_product_over_quotient",
        "expand_log_general_base_powered_two_denominator_factors_with_powered_denominator",
    ] {
        assert_case_step_has_no_substeps(case_id, "Expandir logaritmos");
    }
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
fn derive_didactic_representative_direct_log_contraction_cases_have_no_substeps() {
    for case_id in [
        "contract_log_sum",
        "contract_log_difference",
        "contract_log_product_over_quotient",
        "contract_log_powered_two_denominator_factors",
        "contract_log_general_base_difference",
        "contract_log_general_base_powered_two_denominator_factors_with_powered_denominator",
    ] {
        assert_case_step_has_no_substeps(case_id, "Contraer logaritmos");
    }
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
fn derive_didactic_log_change_of_base_cases_stay_direct() {
    for (case_id, allowed_rules) in [
        (
            "contract_log_change_of_base_chain",
            &["Contraer cadena de logaritmos", "Contraer logaritmos"][..],
        ),
        (
            "expand_log_change_of_base_chain",
            &["Expandir cambio de base", "Expandir logaritmos"][..],
        ),
    ] {
        let artifact = audit_case(&derive_case_by_id(case_id));
        let step = artifact
            .json_steps
            .iter()
            .find(|step| {
                step.get("rule")
                    .and_then(Value::as_str)
                    .is_some_and(|rule| allowed_rules.contains(&rule))
            })
            .unwrap_or_else(|| panic!("expected change-of-base step for {case_id}"));
        assert!(
            step_substep_titles(step).is_empty(),
            "expected no substeps for {case_id}"
        );
    }
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
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

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
}

#[test]
fn derive_didactic_representative_direct_power_merge_cases_keep_merge_narrative() {
    let cases = [
        "merge_same_base_fractional_powers",
        "merge_same_base_fractional_powers_to_integer",
        "merge_same_base_integer_and_fractional_power",
        "merge_same_base_integer_and_symbolic_power",
    ];

    for case_id in cases {
        let artifact = audit_case(&derive_case_by_id(case_id));
        let step = step_by_rule(&artifact, "Sumar exponentes de la misma base");
        let titles = step_substep_titles(step);
        assert!(
            titles.is_empty(),
            "expected no template substeps for {case_id}, got {titles:?}"
        );
    }
}

#[test]
fn derive_didactic_representative_symbolic_power_merge_cases_render_grouped_exponents() {
    let cases = [
        ("merge_same_base_symbolic_powers", "x^a · x^b", "x^(a + b)"),
        (
            "merge_four_same_base_symbolic_powers",
            "x^a · x^b · x^c · x^d",
            "x^(a + b + c + d)",
        ),
    ];

    for (case_id, before, after) in cases {
        let artifact = audit_case(&derive_case_by_id(case_id));
        let step = step_by_rule(&artifact, "Sumar exponentes de la misma base");
        assert_eq!(step.get("before").and_then(Value::as_str), Some(before));
        assert_eq!(step.get("after").and_then(Value::as_str), Some(after));
        assert!(
            step_substep_titles(step).is_empty(),
            "expected no substeps for grouped symbolic merge {case_id}"
        );
    }
}

#[test]
fn derive_didactic_representative_root_power_merge_cases_explain_root_then_merge() {
    let cases = [
        "merge_mixed_root_and_fractional_power_five_sixths",
        "merge_mixed_root_and_symbolic_power",
    ];

    for case_id in cases {
        assert_case_step_has_no_substeps(case_id, "Sumar exponentes de la misma base");
    }
}

#[test]
fn derive_didactic_radical_notable_quotient_recognizes_pattern_with_internal_cleanup() {
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
            "Volver a poner t = sqrt(x)",
            "Deshacer sqrt(x)^2 como x dentro del resultado"
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
fn derive_didactic_representative_rationalize_cases_keep_conjugate_narrative() {
    let expected = &[
        "Cambiar el signo para formar el conjugado",
        "Multiplicar numerador y denominador por ese conjugado",
        "En el denominador aparece una diferencia de cuadrados",
    ];

    for case_id in [
        "rationalize_linear_root",
        "rationalize_linear_root_plus",
        "rationalize_shifted_linear_root",
        "rationalize_symbolic_linear_root",
        "rationalize_symbolic_linear_root_plus",
        "rationalize_symbolic_linear_root_alt_var",
    ] {
        assert_case_step_titles(case_id, "Racionalizar el denominador", expected);
    }
}

#[test]
fn derive_didactic_representative_rationalize_zero_case_keeps_direct_cancel() {
    assert_case_step_has_no_substeps(
        "rationalize_then_cancel_to_zero",
        "Racionalizar el denominador",
    );
    assert_case_step_has_no_substeps(
        "rationalize_then_cancel_to_zero",
        "Restar dos expresiones iguales",
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
    assert_eq!(expand_titles, vec!["Usar (a + b)^2 = a^2 + 2ab + b^2"]);
}

#[test]
fn derive_didactic_representative_direct_nested_fraction_cases_have_no_substeps() {
    assert_case_step_titles(
        "nested_fraction_one_over_sum",
        "Cancelar factores en una fracción",
        &["Primero simplificar la suma del denominador"],
    );
    assert_case_step_titles(
        "nested_fraction_one_over_three_reciprocals",
        "Cancelar factores en una fracción",
        &["Primero simplificar la suma del denominador"],
    );

    for case_id in [
        "nested_fraction_sum_over_reciprocal",
        "nested_fraction_sum_with_fraction_over_scalar_general",
    ] {
        let artifact = audit_case(&derive_case_by_id(case_id));
        let all_titles: Vec<&str> = artifact
            .json_steps
            .iter()
            .flat_map(|step| {
                step.get("substeps")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|substep| substep.get("title").and_then(Value::as_str))
            })
            .collect();
        assert_eq!(
            all_titles,
            Vec::<&str>::new(),
            "unexpected substeps for {case_id}"
        );
    }
}

#[test]
fn derive_didactic_structural_nested_fraction_cases_keep_single_denominator_sum_substep() {
    for case_id in [
        "nested_fraction_one_over_sum_with_fraction",
        "nested_fraction_fraction_over_sum_with_fraction_general",
    ] {
        assert_case_step_titles(
            case_id,
            "Cancelar factores en una fracción",
            &["Primero simplificar la suma del denominador"],
        );
    }
}

#[test]
fn derive_didactic_reverse_structural_nested_fraction_cases_keep_trace_direct() {
    for (case_id, expected_title) in [
        (
            "nested_fraction_one_over_sum_with_fraction_reverse",
            "Reescribir el denominador sacando factor común z",
        ),
        (
            "nested_fraction_fraction_over_sum_with_fraction_general_reverse",
            "Reescribir el denominador sacando factor común d",
        ),
        (
            "nested_fraction_sum_with_fraction_over_scalar_general_reverse",
            "Reescribir el numerador sacando factor común c",
        ),
    ] {
        let artifact = audit_case(&derive_case_by_id(case_id));
        assert!(
            artifact.json_steps.iter().any(|step| {
                step.get("rule")
                    .and_then(Value::as_str)
                    .is_some_and(|rule| rule == "Simplificar fracción anidada")
            }),
            "expected a nested-fraction humanized rule for {case_id}"
        );
        let all_titles: Vec<&str> = artifact
            .json_steps
            .iter()
            .flat_map(|step| {
                step.get("substeps")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|substep| substep.get("title").and_then(Value::as_str))
            })
            .collect();
        assert_eq!(
            all_titles,
            vec![expected_title],
            "unexpected substeps for {case_id}"
        );
    }

    for (case, expected_title) in [
        (
            DeriveCase {
                id: "nested_fraction_one_over_sum_with_fraction_compound_denominator_reverse_inline"
                    .to_string(),
                family: "nested_fraction".to_string(),
                source: "(c+d)/(a*(c+d)+b)".to_string(),
                target: "1/(a + b/(c+d))".to_string(),
                expected_status: "derived".to_string(),
            },
            "Reescribir el denominador sacando factor común c + d",
        ),
        (
            DeriveCase {
                id: "nested_fraction_sum_with_fraction_over_scalar_compound_denominator_reverse_inline"
                    .to_string(),
                family: "nested_fraction".to_string(),
                source: "(a*(c+d)+b)/(e*(c+d))".to_string(),
                target: "(a + b/(c+d))/e".to_string(),
                expected_status: "derived".to_string(),
            },
            "Reescribir el numerador sacando factor común c + d",
        ),
    ] {
        let artifact = audit_case(&case);
        let all_titles: Vec<&str> = artifact
            .json_steps
            .iter()
            .flat_map(|step| {
                step.get("substeps")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|substep| substep.get("title").and_then(Value::as_str))
            })
            .collect();
        assert_eq!(
            all_titles,
            vec![expected_title],
            "unexpected substeps for inline nested-fraction case {}",
            case.id
        );
    }
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
        .map(|substeps| {
            substeps
                .iter()
                .filter_map(|substep| substep.get("title").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(titles, Vec::<&str>::new());
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
        .map(|substeps| {
            substeps
                .iter()
                .filter_map(|substep| substep.get("title").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(titles, Vec::<&str>::new());
}

#[test]
fn derive_didactic_representative_telescoping_fraction_split_cases_keep_gap_narrative() {
    let cases = [
        (
            "split_telescoping_fraction_consecutive",
            vec!["Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)", "Aquí u = n"],
        ),
        (
            "split_telescoping_fraction_gap_two",
            vec![
                "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
                "Aquí u = n y k = 2",
            ],
        ),
        (
            "split_telescoping_fraction_negative_gap_two",
            vec![
                "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
                "Aquí u = n - 2 y k = 2",
            ],
        ),
        (
            "split_telescoping_fraction_affine_gap_two",
            vec![
                "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
                "Aquí u = 2 · n + 1 y k = 2",
            ],
        ),
        (
            "split_telescoping_fraction_affine_symbolic_shift_gap",
            vec![
                "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
                "Aquí u = a · n + b y k = c - b",
            ],
        ),
        (
            "split_telescoping_fraction_difference_squares_unfactored",
            vec![
                "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
                "Aquí u = x - 1 y k = 2",
            ],
        ),
        (
            "split_telescoping_fraction_symbolic_difference_squares_unfactored",
            vec![
                "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
                "Aquí u = x - a y k = 2 · a",
            ],
        ),
    ];

    for (case_id, expected) in cases {
        assert_case_step_titles(case_id, "Descomponer en fracciones telescópicas", &expected);
    }
}

#[test]
fn derive_didactic_representative_telescoping_fraction_combine_cases_keep_inverse_gap_narrative() {
    let cases = [
        (
            "combine_telescoping_fraction_consecutive",
            vec!["Usar 1 / u - 1 / (u + 1) = 1 / (u · (u + 1))", "Aquí u = n"],
        ),
        (
            "combine_telescoping_fraction_gap_two",
            vec![
                "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
                "Aquí u = n y k = 2",
            ],
        ),
        (
            "combine_telescoping_fraction_negative_gap_two",
            vec![
                "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
                "Aquí u = n - 2 y k = 2",
            ],
        ),
        (
            "combine_telescoping_fraction_affine_gap_two",
            vec![
                "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
                "Aquí u = 2 · n + 1 y k = 2",
            ],
        ),
        (
            "combine_telescoping_fraction_affine_symbolic_shift_gap",
            vec![
                "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
                "Aquí u = a · n + b y k = c - b",
            ],
        ),
        (
            "combine_telescoping_fraction_shifted_quadratic_unfactored",
            vec![
                "Llevar a denominador común",
                "Simplificar el numerador y el denominador",
            ],
        ),
        (
            "combine_telescoping_fraction_symbolic_difference_squares_unfactored",
            vec![
                "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
                "Aquí u = x - a y k = 2 · a",
            ],
        ),
    ];

    for (case_id, expected) in cases {
        let artifact = audit_case(&derive_case_by_id(case_id));
        let step = artifact
            .json_steps
            .iter()
            .find(|step| {
                step.get("rule")
                    .and_then(Value::as_str)
                    .is_some_and(|rule| {
                        rule == "Recomponer fracción telescópica" || rule == "Restar fracciones"
                    })
            })
            .unwrap_or_else(|| panic!("expected telescoping combine step for {case_id}"));
        assert_eq!(step_substep_titles(step), expected);
    }
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
fn derive_didactic_symbolic_argument_morrie_telescoping_avoids_tautological_substitution() {
    let artifact = audit_case(&derive_case_by_id(
        "integrate_prep_morrie_symbolic_argument",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar telescopado de cosenos")
        })
        .expect("expected symbolic-argument Morrie telescoping step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected symbolic-argument Morrie telescoping substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(titles, vec!["Usar el telescopado de cosenos"]);
}

#[test]
fn derive_didactic_symbolic_scale_morrie_telescoping_tracks_scaled_base_argument() {
    let artifact = audit_case(&derive_case_by_id("integrate_prep_morrie_symbolic_scale"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar telescopado de cosenos")
        })
        .expect("expected symbolic-scale Morrie telescoping step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected symbolic-scale Morrie telescoping substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar el telescopado de cosenos", "Aquí u = a · x"]
    );
}

#[test]
fn derive_didactic_reverse_morrie_telescoping_uses_expansion_language() {
    let artifact = audit_case(&derive_case_by_id("integrate_prep_morrie_reverse_basic"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar telescopado de cosenos")
        })
        .expect("expected reverse Morrie telescoping step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected reverse Morrie telescoping substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(titles, vec!["Expandir la ley de Morrie", "Aquí u = x"]);
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
fn derive_didactic_symbolic_argument_dirichlet_kernel_tracks_only_n_value() {
    let artifact = audit_case(&derive_case_by_id(
        "integrate_prep_dirichlet_symbolic_argument",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad del núcleo de Dirichlet")
        })
        .expect("expected symbolic-argument Dirichlet kernel step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected symbolic-argument Dirichlet kernel substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(titles, vec!["Usar el núcleo de Dirichlet", "Aquí n = 4"]);
}

#[test]
fn derive_didactic_symbolic_scale_dirichlet_kernel_tracks_scaled_argument() {
    let artifact = audit_case(&derive_case_by_id(
        "integrate_prep_dirichlet_symbolic_scale",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad del núcleo de Dirichlet")
        })
        .expect("expected symbolic-scale Dirichlet kernel step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected symbolic-scale Dirichlet kernel substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar el núcleo de Dirichlet", "Aquí n = 2 y u = a · x"]
    );
}

#[test]
fn derive_didactic_longer_symbolic_scale_dirichlet_kernel_tracks_scaled_argument_and_n() {
    let artifact = audit_case(&derive_case_by_id(
        "integrate_prep_dirichlet_symbolic_scale_longer",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad del núcleo de Dirichlet")
        })
        .expect("expected longer symbolic-scale Dirichlet kernel step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected longer symbolic-scale Dirichlet kernel substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Usar el núcleo de Dirichlet", "Aquí n = 3 y u = a · x"]
    );
}

#[test]
fn derive_didactic_reverse_dirichlet_kernel_uses_expansion_language() {
    let artifact = audit_case(&derive_case_by_id("integrate_prep_dirichlet_reverse_basic"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad del núcleo de Dirichlet")
        })
        .expect("expected reverse Dirichlet kernel step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected reverse Dirichlet kernel substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Expandir el núcleo de Dirichlet", "Aquí n = 2 y u = x"]
    );
}

#[test]
fn derive_didactic_reverse_symbolic_scale_dirichlet_kernel_tracks_scaled_argument_and_n() {
    let artifact = audit_case(&derive_case_by_id(
        "integrate_prep_dirichlet_reverse_symbolic_scale_longer",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad del núcleo de Dirichlet")
        })
        .expect("expected reverse symbolic-scale Dirichlet kernel step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected reverse symbolic-scale Dirichlet kernel substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec!["Expandir el núcleo de Dirichlet", "Aquí n = 3 y u = a · x"]
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
fn derive_didactic_representative_finite_telescoping_product_cases_keep_telescoping_narrative() {
    let direct_cases = [
        (
            "finite_telescoping_product_symbolic_shift_symbolic_lower",
            &[
                "Escribir los primeros y últimos factores del producto",
                "Los factores intermedios se cancelan por parejas",
            ][..],
        ),
        (
            "finite_telescoping_product_affine_symbolic_shift_symbolic_lower",
            &[
                "Escribir los primeros y últimos factores del producto",
                "Los factores intermedios se cancelan por parejas",
            ][..],
        ),
    ];

    for (case_id, expected_titles) in direct_cases {
        assert_case_step_titles(
            case_id,
            "Evaluar producto telescópico finito",
            expected_titles,
        );
    }

    let factorized_cases = [
        (
            "finite_telescoping_product_factorized_square_shifted_base_numeric_symbolic_lower",
            "Aquí u = k + 2",
        ),
        (
            "finite_telescoping_product_factorized_square_shifted_base_symbolic_symbolic_lower",
            "Aquí u = a + k",
        ),
    ];

    for (case_id, u_title) in factorized_cases {
        assert_case_step_titles(
            case_id,
            "Evaluar producto telescópico finito",
            &[
                "Usar (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2",
                u_title,
                "Los factores (u + 1) y (u - 1) se cancelan telescópicamente",
                "Solo quedan el primer factor u - 1 y el último factor u + 1",
            ],
        );
    }
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
fn derive_didactic_representative_finite_telescoping_sum_cases_keep_partial_fraction_narrative() {
    let unit_gap_cases = [(
        "finite_telescoping_sum_symbolic_shift_symbolic_lower",
        "Aquí u = a + k",
    )];

    for (case_id, u_title) in unit_gap_cases {
        assert_case_step_titles(
            case_id,
            "Evaluar suma telescópica finita",
            &[
                "Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)",
                u_title,
                "La suma telescópica cancela los términos intermedios",
            ],
        );
    }

    let affine_gap_cases = [
        (
            "finite_telescoping_sum_affine_symbolic_shift_symbolic_lower",
            "Aquí u = a * k + b y g = a",
        ),
        (
            "finite_telescoping_sum_affine_symbolic_arbitrary_shift_symbolic_lower",
            "Aquí u = a * k + b + c y g = a",
        ),
    ];

    for (case_id, ug_title) in affine_gap_cases {
        assert_case_step_titles(
            case_id,
            "Evaluar suma telescópica finita",
            &[
                "Usar 1 / (u · (u + g)) = 1 / g · (1 / u - 1 / (u + g))",
                ug_title,
                "La suma telescópica cancela los términos intermedios",
            ],
        );
    }
}

#[test]
fn derive_didactic_representative_same_denominator_fraction_cases_have_no_substeps() {
    let cases = [
        (
            "combine_fraction_part_with_same_denominator",
            "Sumar fracciones con mismo denominador",
        ),
        (
            "combine_three_same_denominator_fractions",
            "Sumar fracciones con mismo denominador",
        ),
        (
            "combine_same_denominator_fraction_sum",
            "Sumar fracciones con mismo denominador",
        ),
        (
            "combine_same_denominator_fraction_difference",
            "Restar fracciones con mismo denominador",
        ),
        (
            "combine_symbolic_same_denominator_fraction_subset_with_passthrough",
            "Sumar fracciones con mismo denominador",
        ),
    ];

    for (case_id, rule) in cases {
        assert_case_step_has_no_substeps(case_id, rule);
    }
}

#[test]
fn derive_didactic_representative_fraction_expansion_cases_keep_distribution_and_real_cancellation()
{
    let cases = [
        (
            "expand_fraction_part_with_same_denominator_three_terms",
            &["Repartir el mismo denominador sobre cada término del numerador"][..],
        ),
        (
            "expand_fraction_with_common_scalar_factor_in_denominator",
            &[
                "Usar (a + b) / d = a/d + b/d",
                "Cancelar los factores comunes en la fracción que queda",
            ][..],
        ),
        (
            "expand_fraction_mixed_variable_term_cancellation",
            &[
                "Usar (a + b) / d = a/d + b/d",
                "Cancelar los factores comunes en las fracciones resultantes",
            ][..],
        ),
        (
            "expand_fraction_three_factor_full_cancellation",
            &[
                "Repartir el mismo denominador sobre cada término del numerador",
                "Cancelar los factores comunes en las fracciones resultantes",
            ][..],
        ),
        (
            "expand_fraction_two_cancellations_plus_remainder",
            &[
                "Repartir el mismo denominador sobre cada término del numerador",
                "Cancelar los factores comunes en las fracciones resultantes",
            ][..],
        ),
        (
            "expand_fraction_three_factor_cross_cancellation_plus_remainder",
            &[
                "Repartir el mismo denominador sobre cada término del numerador",
                "Cancelar los factores comunes en las fracciones resultantes",
            ][..],
        ),
        (
            "expand_fraction_three_factor_three_cancellations_to_constant",
            &[
                "Repartir el mismo denominador sobre cada término del numerador",
                "Cancelar los factores comunes en las fracciones resultantes",
            ][..],
        ),
    ];

    for (case_id, expected_titles) in cases {
        assert_case_step_titles(case_id, "Repartir el denominador común", expected_titles);
    }
}

#[test]
fn derive_didactic_representative_fraction_decomposition_cases_keep_whole_plus_remainder_narrative()
{
    let csv_cases = [
        "split_fraction_into_whole_plus_remainder",
        "split_fraction_linear_over_scaled_linear",
        "split_fraction_symbolic_over_general_shift",
        "split_fraction_symbolic_over_scaled_general_linear",
        "split_fraction_symbolic_over_negative_scaled_general_linear",
    ];

    for case_id in csv_cases {
        assert_case_step_titles(
            case_id,
            "Separar parte entera y resto",
            &[
                "Reescribir el numerador como denominador · parte entera + resto",
                "Separar la parte entera de la fracción restante",
            ],
        );
    }

    let monic_cases = [
        DeriveCase {
            id: "split_fraction_monic_symbolic_offset_inline".to_string(),
            family: "fraction_decompose".to_string(),
            source: "(x+a)/(x+b)".to_string(),
            target: "1 + (a-b)/(x+b)".to_string(),
            expected_status: "derived".to_string(),
        },
        DeriveCase {
            id: "split_fraction_monic_scaled_symbolic_inline".to_string(),
            family: "fraction_decompose".to_string(),
            source: "(x+a)/(c*x+d)".to_string(),
            target: "1/c + (a-d/c)/(c*x+d)".to_string(),
            expected_status: "derived".to_string(),
        },
        DeriveCase {
            id: "split_fraction_negative_monic_scaled_symbolic_inline".to_string(),
            family: "fraction_decompose".to_string(),
            source: "(x+a)/(d-c*x)".to_string(),
            target: "-1/c + (a+d/c)/(d-c*x)".to_string(),
            expected_status: "derived".to_string(),
        },
    ];

    for case in monic_cases {
        let artifact = audit_case(&case);
        let step = step_by_rule(&artifact, "Separar parte entera y resto");
        assert_eq!(
            step_substep_titles(step),
            vec![
                "Reescribir el numerador como denominador · parte entera + resto",
                "Separar la parte entera de la fracción restante",
            ],
            "unexpected fraction decomposition narrative for inline monic case {}",
            case.id
        );
    }
}

#[test]
fn derive_didactic_representative_fraction_combination_cases_keep_whole_plus_remainder_narrative() {
    let csv_cases = [
        "combine_whole_plus_remainder_into_fraction",
        "combine_symbolic_whole_plus_remainder_into_fraction",
        "combine_scaled_symbolic_whole_plus_remainder_into_fraction",
        "combine_negative_scaled_symbolic_whole_plus_remainder_into_fraction",
    ];

    for case_id in csv_cases {
        assert_case_step_titles(
            case_id,
            "Unir parte entera y fracción",
            &[
                "Poner la parte entera sobre el mismo denominador",
                "Sumar los numeradores y conservar el denominador",
            ],
        );
    }

    let monic_cases = [
        DeriveCase {
            id: "combine_monic_scaled_symbolic_inline".to_string(),
            family: "fraction_combine".to_string(),
            source: "1/c + (a-d/c)/(c*x+d)".to_string(),
            target: "(x+a)/(c*x+d)".to_string(),
            expected_status: "derived".to_string(),
        },
        DeriveCase {
            id: "combine_negative_monic_scaled_symbolic_inline".to_string(),
            family: "fraction_combine".to_string(),
            source: "-1/c + (a+d/c)/(d-c*x)".to_string(),
            target: "(x+a)/(d-c*x)".to_string(),
            expected_status: "derived".to_string(),
        },
    ];

    for case in monic_cases {
        let artifact = audit_case(&case);
        let step = step_by_rule(&artifact, "Unir parte entera y fracción");
        assert_eq!(
            step_substep_titles(step),
            vec![
                "Poner la parte entera sobre el mismo denominador",
                "Sumar los numeradores y conservar el denominador",
            ],
            "unexpected fraction combination narrative for inline monic case {}",
            case.id
        );
    }
}

#[test]
fn derive_didactic_representative_odd_half_power_cases_use_root_split_narrative() {
    let case_ids = [
        "expand_odd_half_power",
        "expand_odd_half_power_after_simplify",
        "expand_higher_odd_half_power",
        "expand_higher_odd_half_power_after_simplify",
        "expand_higher_odd_half_power_alt_var",
    ];

    for case_id in case_ids {
        let artifact = audit_case(&derive_case_by_id(case_id));

        let step = artifact
            .json_steps
            .iter()
            .find(|step| {
                step.get("rule")
                    .and_then(Value::as_str)
                    .is_some_and(|rule| {
                        matches!(
                            rule,
                            "Reescribir potencia semientera impar"
                                | "Extraer potencia par de la raíz"
                        )
                    })
            })
            .expect("expected odd-half-power derive step");
        let rule = step
            .get("rule")
            .and_then(Value::as_str)
            .expect("expected odd-half-power rule");
        let titles: Vec<&str> = step
            .get("substeps")
            .and_then(Value::as_array)
            .expect("expected odd-half-power substeps")
            .iter()
            .filter_map(|substep| substep.get("title").and_then(Value::as_str))
            .collect();
        match rule {
            "Reescribir potencia semientera impar" => assert_eq!(
                titles,
                vec![
                    "Separar la mitad entera de la mitad radical",
                    "Usar que queda una raíz cuadrada del mismo factor",
                ],
                "unexpected odd-half-power narrative for {case_id}"
            ),
            "Extraer potencia par de la raíz" => {
                assert_eq!(
                    titles.first().copied(),
                    Some("Separar el radicando en una potencia par y un factor"),
                    "unexpected odd-half-power narrative for {case_id}"
                );
                assert!(
                    titles
                        .get(1)
                        .is_some_and(|title| title.starts_with("Como ") && title.contains("≥ 0")),
                    "unexpected odd-half-power narrative for {case_id}: {titles:?}"
                );
            }
            _ => panic!("unexpected odd-half-power rule for {case_id}: {rule}"),
        }
    }
}

#[test]
fn derive_didactic_representative_collect_cases_keep_focus_narrative() {
    let cases = [
        (
            "collect_linear",
            "Agrupar términos por variable",
            vec!["Agrupar los términos que llevan la misma potencia de x"],
        ),
        (
            "collect_linear_alt_variable",
            "Agrupar términos por variable",
            vec!["Agrupar los términos que llevan la misma potencia de y"],
        ),
        (
            "collect_multiple_power_groups",
            "Agrupar términos por variable",
            vec!["Agrupar los términos que llevan la misma potencia de x"],
        ),
        (
            "collect_common_symbolic_coefficients",
            "Agrupar términos por variable",
            vec!["Agrupar los términos que llevan la misma potencia de x"],
        ),
        (
            "collect_composite_monomial_factor",
            "Agrupar términos por factor común",
            vec!["Agrupar los términos que llevan el mismo factor x·y"],
        ),
        (
            "collect_two_composite_factor_groups",
            "Agrupar términos por factor común",
            vec!["Agrupar los términos que llevan el mismo factor x·y"],
        ),
    ];

    for (case_id, rule, expected_titles) in cases {
        let artifact = audit_case(&derive_case_by_id(case_id));
        let step = step_by_rule(&artifact, rule);
        assert_eq!(
            step_substep_titles(step),
            expected_titles,
            "unexpected collect narrative for {case_id}"
        );
    }
}

#[test]
fn derive_didactic_representative_factor_with_division_cases_keep_variable_specific_narrative() {
    let csv_cases = [
        (
            "factor_out_with_division",
            vec!["Si un término no lleva x, escribirlo como x · (t/x)"],
        ),
        (
            "factor_out_with_division_sparse_quintic",
            vec!["Si un término no lleva x, escribirlo como x · (t/x)"],
        ),
        (
            "factor_out_with_division_mixed_septic",
            vec!["Si un término no lleva x, escribirlo como x · (t/x)"],
        ),
        (
            "factor_out_square_with_division_quartic",
            vec!["Si un término no lleva x^2, escribirlo como x^2 · (t/x^2)"],
        ),
        (
            "factor_out_cube_with_division_septic",
            vec!["Si un término no lleva x^3, escribirlo como x^3 · (t/x^3)"],
        ),
    ];

    for (case_id, expected_titles) in csv_cases {
        let artifact = audit_case(&derive_case_by_id(case_id));
        let step = step_by_rule(&artifact, "Sacar factor usando división");
        assert_eq!(
            step_substep_titles(step),
            expected_titles,
            "unexpected factor-with-division narrative for {case_id}"
        );
    }

    let y_case = DeriveCase {
        id: "factor_out_with_division_quadratic_y_inline".to_string(),
        family: "conditional_factor".to_string(),
        source: "a*y^2 + b*y + c".to_string(),
        target: "y*(a*y + b + c/y)".to_string(),
        expected_status: "derived".to_string(),
    };
    let y_artifact = audit_case(&y_case);
    let y_step = step_by_rule(&y_artifact, "Sacar factor usando división");
    assert_eq!(
        step_substep_titles(y_step),
        vec!["Si un término no lleva y, escribirlo como y · (t/y)"],
        "unexpected factor-with-division narrative for inline y case"
    );
}

#[test]
fn derive_didactic_combine_like_terms_with_zero_sums_coefficients_without_noise() {
    let case = DeriveCase {
        id: "combine_like_terms_with_zero_inline".to_string(),
        family: "simplify".to_string(),
        source: "2*x + 3*x + 0".to_string(),
        target: "5*x".to_string(),
        expected_status: "derived".to_string(),
    };
    let artifact = audit_case(&case);

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Agrupar términos semejantes")
        })
        .expect("expected combine-like-terms step");
    assert!(step_substep_titles(step).is_empty());
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
    assert_case_step_has_no_substeps("expand_common_factor_sum", "Expandir la expresión");
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

    let steps: Vec<_> = artifact
        .json_steps
        .iter()
        .filter(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir la expresión")
        })
        .collect();
    assert_eq!(steps.len(), 2, "expected two direct distributive steps");
    for step in steps {
        assert!(step_substep_titles(step).is_empty());
    }
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
                .is_some_and(|rule| rule == "Cancelar un factor común")
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
            "Reducir la fracción que queda"
        ]
    );
}

#[test]
fn derive_didactic_monomial_common_factor_fraction_cancels_symbol_then_simplifies_coefficients() {
    assert_case_step_has_no_substeps(
        "cancel_fraction_monomial_common_factor",
        "Cancelar un factor común",
    );
}

#[test]
fn derive_didactic_representative_complete_square_cases_use_formula_and_track_coefficients() {
    let cases = [
        (
            "solve_prep_complete_square_monic_numeric",
            &[
                "Usar la fórmula de completar el cuadrado",
                "Aquí A = 1, B = 6 y C = 5",
            ][..],
        ),
        (
            "solve_prep_complete_square_symbolic_leading_coeff",
            &[
                "Usar la fórmula de completar el cuadrado",
                "Aquí A = a, B = b y C = c",
            ][..],
        ),
        (
            "solve_prep_complete_square_symbolic_monic_parametric",
            &[
                "Usar la fórmula de completar el cuadrado",
                "Aquí A = 1, B = 2 · b y C = c",
            ][..],
        ),
        (
            "solve_prep_complete_square_alt_variable_symbolic_leading_coeff",
            &[
                "Usar la fórmula de completar el cuadrado",
                "Aquí A = a, B = b y C = c",
            ][..],
        ),
        (
            "solve_prep_complete_square_symbolic_negative_linear_coeff",
            &[
                "Usar la fórmula de completar el cuadrado",
                "Aquí A = a, B = -b y C = c",
            ][..],
        ),
        (
            "solve_prep_complete_square_negative_symbolic_leading_coeff",
            &[
                "Usar la fórmula de completar el cuadrado",
                "Aquí A = -a, B = b y C = c",
            ][..],
        ),
        (
            "solve_prep_complete_square_fractional_monic_numeric",
            &[
                "Usar la fórmula de completar el cuadrado",
                "Aquí A = 1, B = 3 y C = 1",
            ][..],
        ),
        (
            "solve_prep_complete_square_fractional_symbolic_leading_coeff",
            &[
                "Usar la fórmula de completar el cuadrado",
                "Aquí A = a/2, B = b y C = c",
            ][..],
        ),
    ];

    for (case_id, expected_titles) in cases {
        assert_case_step_titles(case_id, "Completar el cuadrado", expected_titles);
    }
}

#[test]
fn derive_didactic_cube_sum_product_uses_sum_of_cubes_identity_language() {
    let artifact = audit_case(&derive_case_by_id("expand_cube_sum_product"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir la expresión")
        })
        .expect("expected polynomial product derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected polynomial product substeps")
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
fn derive_didactic_cube_difference_product_uses_difference_of_cubes_identity_language() {
    assert_case_step_has_no_substeps("expand_cube_difference_product", "Expandir la expresión");
}

#[test]
fn derive_didactic_quadratic_difference_of_squares_product_keeps_cancellation_story() {
    assert_case_step_has_no_substeps(
        "expand_difference_of_squares_quadratic_product",
        "Expandir la expresión",
    );
}

#[test]
fn derive_didactic_sixth_power_plus_product_keeps_pairwise_cancellation_story() {
    let artifact = audit_case(&derive_case_by_id("expand_sixth_power_plus_product"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir y reagrupar un producto polinómico")
        })
        .expect("expected polynomial product derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected polynomial product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Distribuir cada término del producto",
            "Agrupar los términos del mismo grado",
            "Los términos intermedios se cancelan por parejas"
        ]
    );
}

#[test]
fn derive_didactic_sixth_power_minus_product_keeps_pairwise_cancellation_story() {
    let artifact = audit_case(&derive_case_by_id("expand_sixth_power_minus_product"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir y reagrupar un producto polinómico")
        })
        .expect("expected polynomial product derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected polynomial product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Distribuir cada término del producto",
            "Agrupar los términos del mismo grado",
            "Los términos intermedios se cancelan por parejas"
        ]
    );
}

#[test]
fn derive_didactic_eighth_power_minus_multifactor_product_uses_summary_cancellation_story() {
    assert_case_step_has_no_substeps(
        "expand_eighth_power_minus_multifactor_product",
        "Expandir la expresión",
    );
}

#[test]
fn derive_didactic_ninth_power_plus_product_keeps_pairwise_cancellation_story() {
    assert_case_step_has_no_substeps("expand_ninth_power_plus_product", "Expandir la expresión");
}

#[test]
fn derive_didactic_symbolic_cube_sum_product_uses_sum_of_cubes_identity_language() {
    let artifact = audit_case(&derive_case_by_id("expand_symbolic_cube_sum_product"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir la expresión")
        })
        .expect("expected polynomial product derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected polynomial product substeps")
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
fn derive_didactic_symbolic_cube_difference_product_uses_difference_of_cubes_identity_language() {
    let artifact = audit_case(&derive_case_by_id(
        "expand_symbolic_cube_difference_product",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir la expresión")
        })
        .expect("expected polynomial product derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected polynomial product substeps")
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
fn derive_didactic_symbolic_sixth_power_plus_product_uses_sixth_power_identity_language() {
    let artifact = audit_case(&derive_case_by_id(
        "expand_symbolic_sixth_power_plus_product",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir la expresión")
        })
        .expect("expected polynomial product derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected polynomial product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Reconocer el patrón (a^2 + b^2)(a^4 - a^2b^2 + b^4)",
            "Aplicar (a^2 + b^2)(a^4 - a^2b^2 + b^4) = a^6 + b^6"
        ]
    );
}

#[test]
fn derive_didactic_symbolic_sixth_power_minus_product_uses_sixth_power_identity_language() {
    let artifact = audit_case(&derive_case_by_id(
        "expand_symbolic_sixth_power_minus_product",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir la expresión")
        })
        .expect("expected polynomial product derive step");
    let titles: Vec<&str> = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected polynomial product substeps")
        .iter()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();
    assert_eq!(
        titles,
        vec![
            "Reconocer el patrón (a^2 - b^2)(a^4 + a^2b^2 + b^4)",
            "Aplicar (a^2 - b^2)(a^4 + a^2b^2 + b^4) = a^6 - b^6"
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
