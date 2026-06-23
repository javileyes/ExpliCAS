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
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

#[derive(Debug, Clone)]
struct DeriveCase {
    id: String,
    family: String,
    source: String,
    target: String,
    expected_status: String,
}

#[derive(Debug, Clone)]
struct AuditArtifact {
    result: String,
    step_count: usize,
    web_substep_count: usize,
    json_steps: Vec<Value>,
    flags: Vec<String>,
}

struct TimedAuditArtifact {
    artifact: AuditArtifact,
    seconds: f64,
}

struct TimedCliLines {
    lines: Vec<String>,
    seconds: f64,
}

#[derive(Debug, Default)]
struct FamilyFlagSummary {
    cases: usize,
    flagged_cases: usize,
    no_substep_flags: usize,
    web_substeps: usize,
}

static AUDIT_CASE_CACHE: LazyLock<Mutex<BTreeMap<String, AuditArtifact>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

static DERIVE_CASES: LazyLock<Vec<DeriveCase>> = LazyLock::new(parse_derive_cases);
static DERIVE_CASES_BY_ID: LazyLock<BTreeMap<String, DeriveCase>> = LazyLock::new(|| {
    DERIVE_CASES
        .iter()
        .cloned()
        .map(|case| (case.id.clone(), case))
        .collect()
});

fn parse_derive_cases() -> Vec<DeriveCase> {
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

fn load_derive_cases() -> &'static [DeriveCase] {
    DERIVE_CASES.as_slice()
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
        time_budget_ms: None,
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

fn timed_audit_case(case: &DeriveCase) -> TimedAuditArtifact {
    let start = Instant::now();
    let artifact = audit_case(case);
    TimedAuditArtifact {
        artifact,
        seconds: start.elapsed().as_secs_f64(),
    }
}

fn timed_run_cli_lines(case: &DeriveCase) -> TimedCliLines {
    let start = Instant::now();
    let lines = run_cli_lines(case);
    TimedCliLines {
        lines,
        seconds: start.elapsed().as_secs_f64(),
    }
}

fn derive_audit_worker_count(case_count: usize) -> usize {
    if case_count <= 1 {
        return 1;
    }
    if let Ok(raw) = env::var("DERIVE_AUDIT_THREADS") {
        if let Ok(parsed) = raw.parse::<usize>() {
            return parsed.clamp(1, case_count);
        }
    }

    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(4)
        .clamp(1, case_count)
}

fn map_derive_cases_parallel<T, F>(cases: &[DeriveCase], f: F) -> Vec<T>
where
    T: Send,
    F: Fn(&DeriveCase) -> T + Sync,
{
    let worker_count = derive_audit_worker_count(cases.len());
    if worker_count == 1 {
        return cases.iter().map(f).collect();
    }

    let next_index = AtomicUsize::new(0);
    let mut empty_slots = Vec::with_capacity(cases.len());
    empty_slots.resize_with(cases.len(), || None);
    let results = Mutex::new(empty_slots);

    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| loop {
                let index = next_index.fetch_add(1, Ordering::Relaxed);
                if index >= cases.len() {
                    break;
                }

                let item = f(&cases[index]);
                results.lock().expect("derive audit result mutex poisoned")[index] = Some(item);
            });
        }
    });

    results
        .into_inner()
        .expect("derive audit result mutex poisoned")
        .into_iter()
        .map(|item| item.expect("derive audit worker should fill every result slot"))
        .collect()
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

fn is_self_explanatory_collect_rule(rule: &str) -> bool {
    matches!(
        rule,
        "Agrupar términos por variable" | "Agrupar términos por factor común"
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
            | "Reescribir secante como recíproco del coseno"
            | "Reescribir cosecante como recíproco del seno"
            | "Reescribir cotangente como coseno entre seno"
            | "Reconocer secante desde un recíproco"
            | "Reconocer cosecante desde un recíproco"
            | "Reconocer cotangente desde un cociente"
            | "Cancelar funciones trigonométricas recíprocas"
            | "Reconocer tangente por cotangente como 1"
            | "Reconocer seno por cosecante como 1"
            | "Reconocer coseno por secante como 1"
            | "Aplicar identidad pitagórica recíproca"
            | "Aplicar la identidad pitagórica"
            | "Aplicar identidad de cofunción"
            | "Aplicar suma a producto"
            | "Aplicar identidad del cuadrado trigonométrico"
            | "Repartir el denominador común"
            | "Aplicar identidad de ángulo mitad"
            | "Aplicar identidad de tangente de ángulo mitad"
            | "Aplicar identidad de tangente de ángulo doble"
            | "Aplicar identidad de tangente de suma/diferencia de ángulos"
            | "Expandir tangente como seno entre coseno"
            | "Convertir un cociente trigonométrico en tangente"
            | "Aplicar identidad de arctangentes"
            | "Expandir logaritmos"
            | "Contraer logaritmos"
            | "Sacar un exponente fuera del logaritmo"
            | "Expandir cambio de base"
            | "Contraer cadena de logaritmos"
            | "Cancelar logaritmo natural y exponencial inversos"
            | "Cancelar exponencial y logaritmo inversos"
            | "Expandir binomio"
            | "Negative Base Power"
            | "Simplificar potencia con base negativa"
            | "Invertir una resta dentro de una potencia par"
            | "Expandir ángulo doble"
            | "Contraer ángulo doble"
            | "Aplicar identidad hiperbólica de ángulo doble"
            | "Aplicar identidad hiperbólica de ángulo triple"
            | "Aplicar identidad exponencial hiperbólica"
            | "Aplicar identidad pitagórica hiperbólica"
            | "Reconocer tangente hiperbólica desde un cociente"
            | "Evaluar valor hiperbólico especial"
            | "Evaluar valor trigonométrico especial"
            | "Reescribir la raíz como potencia fraccionaria"
            | "Sumar exponentes de la misma base"
            | "Combinar raíces en un producto"
            | "Reconocer un cociente notable"
            | "Merge Sqrt Product"
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
                        || is_self_explanatory_collect_rule(rule)
                        || is_self_explanatory_identity_rule(rule)
                        || rule == "Simplificar fracción anidada"
                })
        })
}

fn audit_case_cache_key(case: &DeriveCase) -> String {
    format!(
        "{}|{}|{}|{}|{}",
        case.id, case.family, case.source, case.target, case.expected_status
    )
}

fn audit_case(case: &DeriveCase) -> AuditArtifact {
    let cache_key = audit_case_cache_key(case);
    if let Some(cached) = AUDIT_CASE_CACHE
        .lock()
        .expect("audit case cache poisoned")
        .get(&cache_key)
        .cloned()
    {
        return cached;
    }

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

    let artifact = AuditArtifact {
        result,
        step_count,
        web_substep_count,
        json_steps,
        flags,
    };

    AUDIT_CASE_CACHE
        .lock()
        .expect("audit case cache poisoned")
        .insert(cache_key, artifact.clone());

    artifact
}

fn derive_case_by_id(id: &str) -> DeriveCase {
    DERIVE_CASES_BY_ID
        .get(id)
        .cloned()
        .unwrap_or_else(|| panic!("missing derive audit case {id}"))
}

fn phase_family_hotspots(cases: &[DeriveCase], seconds_by_case: &[f64]) -> String {
    assert_eq!(
        cases.len(),
        seconds_by_case.len(),
        "derive audit timing expects one timing per case"
    );

    let mut by_family = BTreeMap::<String, f64>::new();
    for (case, seconds) in cases.iter().zip(seconds_by_case.iter()) {
        *by_family.entry(case.family.clone()).or_default() += *seconds;
    }
    let mut rows = by_family.into_iter().collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    rows.into_iter()
        .take(6)
        .map(|(family, seconds)| format!("{family}:{seconds:.3}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn phase_case_hotspots(cases: &[DeriveCase], seconds_by_case: &[f64]) -> String {
    assert_eq!(
        cases.len(),
        seconds_by_case.len(),
        "derive audit timing expects one timing per case"
    );

    let mut rows = cases
        .iter()
        .zip(seconds_by_case.iter())
        .map(|(case, seconds)| (case.id.clone(), *seconds))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    rows.into_iter()
        .take(6)
        .map(|(id, seconds)| format!("{id}:{seconds:.3}"))
        .collect::<Vec<_>>()
        .join(",")
}

const QUICK_DERIVE_AUDIT_PER_FAMILY: usize = 1;

fn quick_derive_audit_quota_for_family(family: &str) -> usize {
    if matches!(
        family,
        "expand" | "log_contract" | "trig_expand" | "trig_contract" | "finite_telescoping"
    ) {
        QUICK_DERIVE_AUDIT_PER_FAMILY + 1
    } else {
        QUICK_DERIVE_AUDIT_PER_FAMILY
    }
}

fn quick_derive_audit_preferred_case_ids_for_family(family: &str) -> &'static [&'static str] {
    match family {
        // Broad smoke coverage for collect only needs a direct linear grouping case.
        "collect" => &["collect_linear"],
        // Keep the factor-with-division family in the quick audit, but prefer the
        // smallest direct representative instead of the septic variant picked by sort order.
        "conditional_factor" => &["factor_out_with_division"],
        // Broad smoke coverage for same-denominator combines only needs one direct pair.
        "fraction_combine" => &["combine_fraction_part_with_same_denominator"],
        // The simple numerator split is enough here; heavier exact-division and cancellation
        // variants stay pinned by dedicated regressors.
        "fraction_expand" => &["expand_fraction_simple"],
        // The whole-plus-remainder split is already the cheapest useful smoke case.
        "fraction_decompose" => &["split_fraction_into_whole_plus_remainder"],
        // Broad smoke coverage for power merges only needs the direct symbolic pair.
        "power_merge" => &["merge_same_base_symbolic_powers"],
        // The broad quick audit only needs one smoke case for redundant-substep checks here.
        // Prefer the already-pinned fast direct complete-square variant instead of the much
        // heavier symbolic-leading-coefficient case.
        "solve_prep" => &["solve_prep_complete_square_monic_numeric"],
        // Broad smoke coverage for polynomial products does not need the slower cube-product
        // variant because dedicated regressors already pin that narrative.
        "polynomial_product" => &["expand_difference_of_squares_quadratic_product"],
        // Keep two direct factorization samples that stay cheap while still exercising
        // distinct identity-driven factor routes in the broad smoke audit.
        "factor" => &[
            "factor_common_factor_sum",
            "factor_perfect_square_trinomial",
        ],
        _ => &[],
    }
}

fn derive_audit_cases(sampled: bool) -> Vec<DeriveCase> {
    let offset = env::var("DERIVE_AUDIT_OFFSET")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(0);
    let limit = env::var("DERIVE_AUDIT_LIMIT")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok());

    let derived_cases = load_derive_cases()
        .iter()
        .filter(|case| case.expected_status == "derived")
        .cloned();

    let cases: Vec<_> = if sampled {
        let mut by_family = BTreeMap::<String, Vec<DeriveCase>>::new();
        for case in derived_cases {
            by_family.entry(case.family.clone()).or_default().push(case);
        }

        let mut sampled_cases = Vec::new();
        for (family, cases) in by_family {
            let quota = quick_derive_audit_quota_for_family(&family);
            let mut selected_ids = HashSet::new();

            for preferred_id in quick_derive_audit_preferred_case_ids_for_family(&family) {
                if selected_ids.len() >= quota {
                    break;
                }
                if let Some(case) = cases.iter().find(|case| case.id == *preferred_id) {
                    sampled_cases.push(case.clone());
                    selected_ids.insert(case.id.clone());
                }
            }

            for case in cases {
                if selected_ids.len() >= quota {
                    break;
                }
                if selected_ids.insert(case.id.clone()) {
                    sampled_cases.push(case);
                }
            }
        }

        sampled_cases
    } else {
        derived_cases.collect()
    };

    cases
        .into_iter()
        .skip(offset)
        .take(limit.unwrap_or(usize::MAX))
        .collect()
}

fn assert_cases_render_without_redundant_single_substeps(cases: &[DeriveCase]) {
    let trace = env::var("DERIVE_AUDIT_TRACE").ok().as_deref() == Some("1");

    for case in cases {
        if trace {
            eprintln!("audit-case {}", case.id);
        }
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
            run_cli_lines(case).join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "multiple substeps share the same snapshot inside a step"),
            "derive audit case {} has duplicate multi-substep snapshots; cli=\n{}\nflags={:?}",
            case.id,
            run_cli_lines(case).join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "non-contextual substep duplicates parent snapshot"),
            "derive audit case {} still has a non-contextual substep that duplicates the parent snapshot; cli=\n{}\nflags={:?}",
            case.id,
            run_cli_lines(case).join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "generic fraction cleanup title remains in substeps"),
            "derive audit case {} still uses a weak generic fraction cleanup title; cli=\n{}\nflags={:?}",
            case.id,
            run_cli_lines(case).join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "self-explanatory fraction rule still emits redundant substeps"),
            "derive audit case {} still emits redundant fraction-rule substeps; cli=\n{}\nflags={:?}",
            case.id,
            run_cli_lines(case).join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "weak editorial substep title remains in derive"),
            "derive audit case {} still uses a weak editorial substep title; cli=\n{}\nflags={:?}",
            case.id,
            run_cli_lines(case).join("\n"),
            artifact.flags
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "noisy template substep remains in derive"),
            "derive audit case {} still keeps a noisy template substep; cli=\n{}\nflags={:?}",
            case.id,
            run_cli_lines(case).join("\n"),
            artifact.flags
        );
    }
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

fn first_substep_latex(step: &Value) -> (&str, &str) {
    let substep = step
        .get("substeps")
        .and_then(Value::as_array)
        .and_then(|substeps| substeps.first())
        .expect("expected at least one substep");
    (
        substep
            .get("before_latex")
            .and_then(Value::as_str)
            .expect("expected substep before_latex"),
        substep
            .get("after_latex")
            .and_then(Value::as_str)
            .expect("expected substep after_latex"),
    )
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
    assert_eq!(
        step_substep_titles(step),
        expected_titles,
        "unexpected substep titles for `{case_id}` / `{rule}`"
    );
}

fn assert_case_step_has_no_substeps(case_id: &str, rule: &str) {
    let artifact = audit_case(&derive_case_by_id(case_id));
    let step = step_by_rule(&artifact, rule);
    assert!(
        step_substep_titles(step).is_empty(),
        "expected `{case_id}` / `{rule}` to stay direct without substeps"
    );
}

fn assert_case_has_no_no_web_substeps_flag(case_id: &str) {
    let artifact = audit_case(&derive_case_by_id(case_id));
    assert!(
        !artifact
            .flags
            .iter()
            .any(|flag| flag == "no web substeps emitted"),
        "{case_id} should be accepted as a direct self-explanatory step"
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

fn build_report(
    cases: &[DeriveCase],
    artifacts: &[AuditArtifact],
    cli_lines_by_case: &[Vec<String>],
) -> String {
    assert_eq!(
        cases.len(),
        artifacts.len(),
        "derive audit report expects one artifact per case"
    );
    assert_eq!(
        cases.len(),
        cli_lines_by_case.len(),
        "derive audit report expects one CLI transcript per case"
    );

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
    let mut flag_counts = BTreeMap::<String, usize>::new();
    let mut family_summaries = BTreeMap::<String, FamilyFlagSummary>::new();
    for (case, artifact) in cases.iter().zip(artifacts.iter()) {
        let summary = family_summaries.entry(case.family.clone()).or_default();
        summary.cases += 1;
        summary.web_substeps += artifact.web_substep_count;
        if !artifact.flags.is_empty() {
            summary.flagged_cases += 1;
        }
        for flag in &artifact.flags {
            *flag_counts.entry(flag.clone()).or_insert(0) += 1;
            if flag == "no web substeps emitted" {
                summary.no_substep_flags += 1;
            }
        }
    }
    let flagged_cases = artifacts
        .iter()
        .filter(|artifact| !artifact.flags.is_empty())
        .count();
    let no_substep_flag_cases = artifacts
        .iter()
        .filter(|artifact| {
            artifact
                .flags
                .iter()
                .any(|flag| flag == "no web substeps emitted")
        })
        .count();

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

    writeln!(out, "## Flag Summary").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "- Cases with flags: `{flagged_cases}`").unwrap();
    writeln!(
        out,
        "- Cases flagged as no web substeps emitted: `{no_substep_flag_cases}`"
    )
    .unwrap();
    writeln!(out).unwrap();
    if flag_counts.is_empty() {
        writeln!(out, "No audit flags emitted.").unwrap();
        writeln!(out).unwrap();
    } else {
        let mut sorted_flags = flag_counts.into_iter().collect::<Vec<_>>();
        sorted_flags.sort_by(|(left_flag, left_count), (right_flag, right_count)| {
            right_count
                .cmp(left_count)
                .then_with(|| left_flag.cmp(right_flag))
        });
        writeln!(out, "| flag | cases |").unwrap();
        writeln!(out, "| --- | ---: |").unwrap();
        for (flag, count) in sorted_flags {
            writeln!(out, "| {flag} | {count} |").unwrap();
        }
        writeln!(out).unwrap();
    }

    let mut sorted_families = family_summaries.into_iter().collect::<Vec<_>>();
    sorted_families.sort_by(|(left_family, left), (right_family, right)| {
        right
            .no_substep_flags
            .cmp(&left.no_substep_flags)
            .then_with(|| right.flagged_cases.cmp(&left.flagged_cases))
            .then_with(|| left_family.cmp(right_family))
    });
    writeln!(
        out,
        "| family | cases | flagged | no-substeps flag | web substeps |"
    )
    .unwrap();
    writeln!(out, "| --- | ---: | ---: | ---: | ---: |").unwrap();
    for (family, summary) in sorted_families {
        writeln!(
            out,
            "| `{family}` | {} | {} | {} | {} |",
            summary.cases, summary.flagged_cases, summary.no_substep_flags, summary.web_substeps
        )
        .unwrap();
    }
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

    for ((case, artifact), cli_lines) in cases
        .iter()
        .zip(artifacts.iter())
        .zip(cli_lines_by_case.iter())
    {
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
        for line in cli_lines {
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
fn derive_didactic_report_summarizes_audit_flags_by_family() {
    let cases = vec![
        DeriveCase {
            id: "simplify_clear".to_string(),
            family: "simplify".to_string(),
            source: "x".to_string(),
            target: "x".to_string(),
            expected_status: "derived".to_string(),
        },
        DeriveCase {
            id: "simplify_opaque".to_string(),
            family: "simplify".to_string(),
            source: "sin(x)^2+cos(x)^2".to_string(),
            target: "1".to_string(),
            expected_status: "derived".to_string(),
        },
        DeriveCase {
            id: "factor_noisy".to_string(),
            family: "factor".to_string(),
            source: "x^2-1".to_string(),
            target: "(x-1)*(x+1)".to_string(),
            expected_status: "derived".to_string(),
        },
    ];
    let artifacts = vec![
        AuditArtifact {
            result: "x".to_string(),
            step_count: 1,
            web_substep_count: 1,
            json_steps: Vec::new(),
            flags: Vec::new(),
        },
        AuditArtifact {
            result: "1".to_string(),
            step_count: 1,
            web_substep_count: 0,
            json_steps: Vec::new(),
            flags: vec!["no web substeps emitted".to_string()],
        },
        AuditArtifact {
            result: "(x - 1) * (x + 1)".to_string(),
            step_count: 1,
            web_substep_count: 1,
            json_steps: Vec::new(),
            flags: vec!["weak editorial substep title remains in derive".to_string()],
        },
    ];
    let cli_lines = cases
        .iter()
        .map(|case| vec![format!("derive {}", case.id)])
        .collect::<Vec<_>>();

    let report = build_report(&cases, &artifacts, &cli_lines);

    assert!(report.contains("## Flag Summary"));
    assert!(report.contains("- Cases with flags: `2`"));
    assert!(report.contains("- Cases flagged as no web substeps emitted: `1`"));
    assert!(report.contains("| no web substeps emitted | 1 |"));
    assert!(report.contains("| weak editorial substep title remains in derive | 1 |"));
    assert!(report.contains("| `simplify` | 2 | 1 | 1 | 1 |"));
    assert!(report.contains("| `factor` | 1 | 1 | 0 | 1 |"));
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
        .iter()
        .filter(|case| case.expected_status == "derived")
    {
        assert!(
            ids.insert(case.id.clone()),
            "duplicate derive didactic audit case id: {}",
            case.id
        );
    }
}

fn sampled_derive_audit_cases_for_families(families: &[&str]) -> Vec<DeriveCase> {
    let wanted = families.iter().copied().collect::<HashSet<_>>();
    derive_audit_cases(true)
        .into_iter()
        .filter(|case| wanted.contains(case.family.as_str()))
        .collect()
}

fn assert_sampled_derive_audit_cases_for_families_render_without_redundant_single_substeps(
    families: &[&str],
) {
    let cases = sampled_derive_audit_cases_for_families(families);
    assert!(
        !cases.is_empty(),
        "quick derive didactic audit family slice must not be empty"
    );
    assert_cases_render_without_redundant_single_substeps(&cases);
}

const QUICK_DERIVE_AUDIT_ALGEBRAIC_FAMILIES: &[&str] = &[
    "collect",
    "conditional_factor",
    "expand",
    "factor",
    "fraction_combine",
    "fraction_decompose",
    "fraction_expand",
    "polynomial_product",
    "power_merge",
    "solve_prep",
];

const QUICK_DERIVE_AUDIT_LOG_TRIG_FAMILIES: &[&str] = &[
    "integrate_prep",
    "log_contract",
    "log_exp_inverse",
    "log_expand",
    "log_inverse_power",
    "trig_contract",
    "trig_expand",
];

const QUICK_DERIVE_AUDIT_STRUCTURAL_FAMILIES: &[&str] = &[
    "finite_aggregate",
    "finite_telescoping",
    "nested_fraction",
    "number_theory",
    "radical_power",
    "rationalize",
    "simplify",
    "telescoping_fraction",
];

#[test]
fn derive_didactic_quick_audit_family_partition_covers_sampled_cases() {
    let all_cases = derive_audit_cases(true);
    let mut covered_ids = HashSet::new();

    for families in [
        QUICK_DERIVE_AUDIT_ALGEBRAIC_FAMILIES,
        QUICK_DERIVE_AUDIT_LOG_TRIG_FAMILIES,
        QUICK_DERIVE_AUDIT_STRUCTURAL_FAMILIES,
    ] {
        for case in sampled_derive_audit_cases_for_families(families) {
            assert!(
                covered_ids.insert(case.id.clone()),
                "quick derive didactic audit family partition duplicated case {}",
                case.id
            );
        }
    }

    assert!(
        !all_cases.is_empty(),
        "quick derive didactic audit corpus must not be empty"
    );
    assert_eq!(
        covered_ids.len(),
        all_cases.len(),
        "quick derive didactic audit family partition must cover the full sampled corpus"
    );
}

#[test]
fn derive_didactic_quick_audit_algebraic_cases_render_steps_without_redundant_single_substeps() {
    assert_sampled_derive_audit_cases_for_families_render_without_redundant_single_substeps(
        QUICK_DERIVE_AUDIT_ALGEBRAIC_FAMILIES,
    );
}

#[test]
fn derive_didactic_quick_audit_log_trig_cases_render_steps_without_redundant_single_substeps() {
    assert_sampled_derive_audit_cases_for_families_render_without_redundant_single_substeps(
        QUICK_DERIVE_AUDIT_LOG_TRIG_FAMILIES,
    );
}

#[test]
fn derive_didactic_quick_audit_structural_cases_render_steps_without_redundant_single_substeps() {
    assert_sampled_derive_audit_cases_for_families_render_without_redundant_single_substeps(
        QUICK_DERIVE_AUDIT_STRUCTURAL_FAMILIES,
    );
}

#[test]
fn derive_didactic_finite_aggregate_closed_forms_explain_symbolic_evaluation() {
    for (case_id, rule, expected_titles) in [
        (
            "finite_aggregate_sum_first_integers_symbolic",
            "Aplicar fórmula de suma de enteros",
            &[
                "Escribir la suma con sus extremos",
                "Usar la fórmula cerrada para la suma de enteros",
            ][..],
        ),
        (
            "finite_aggregate_sum_of_squares_symbolic",
            "Aplicar fórmula de suma de cuadrados",
            &[
                "Escribir la suma con sus extremos",
                "Usar la fórmula cerrada para la suma de cuadrados",
            ][..],
        ),
        (
            "finite_aggregate_sum_geometric_power_base_two_symbolic",
            "Aplicar fórmula de suma geométrica",
            &[
                "Escribir la suma con sus extremos",
                "Usar la fórmula cerrada para la suma geométrica",
            ][..],
        ),
        (
            "finite_aggregate_product_first_integers_symbolic",
            "Aplicar producto factorial",
            &[
                "Escribir el producto con sus extremos",
                "Usar factorial para el producto de enteros consecutivos",
            ][..],
        ),
        (
            "finite_aggregate_product_of_squares_symbolic",
            "Aplicar producto de potencias",
            &[
                "Escribir el producto con sus extremos",
                "Convertir el producto de potencias en potencia de factoriales",
            ][..],
        ),
        (
            "finite_aggregate_product_constant_symbolic",
            "Aplicar producto de constante",
            &[
                "Escribir el producto con sus extremos",
                "Contar factores iguales en el producto",
            ][..],
        ),
    ] {
        assert_case_step_titles(case_id, rule, expected_titles);
    }

    for case_id in [
        "finite_aggregate_product_constant_symbolic",
        "finite_aggregate_product_constant_symbolic_lower_bound",
        "finite_aggregate_product_first_integers_symbolic",
        "finite_aggregate_product_first_integers_symbolic_lower_bound",
        "finite_aggregate_product_of_cubes_symbolic_lower_bound",
        "finite_aggregate_product_of_squares_symbolic",
        "finite_aggregate_product_of_squares_symbolic_lower_bound",
        "finite_aggregate_sum_constant_symbolic",
        "finite_aggregate_sum_constant_symbolic_lower_bound",
        "finite_aggregate_sum_first_integers_symbolic",
        "finite_aggregate_sum_first_integers_symbolic_lower_bound",
        "finite_aggregate_sum_geometric_power_base_two_symbolic",
        "finite_aggregate_sum_geometric_power_base_two_symbolic_lower_bound",
        "finite_aggregate_sum_of_cubes_symbolic",
        "finite_aggregate_sum_of_cubes_symbolic_lower_bound",
        "finite_aggregate_sum_of_squares_symbolic",
        "finite_aggregate_sum_of_squares_symbolic_lower_bound",
    ] {
        let artifact = audit_case(&derive_case_by_id(case_id));
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "no web substeps emitted"),
            "{case_id} should expose the finite aggregate closed-form step"
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag.contains("duplicates")),
            "{case_id} should not duplicate the parent snapshot: {:?}",
            artifact.flags
        );
    }
}

#[test]
#[ignore = "full derive didactic audit is expensive; run manually before broad didactic refactors"]
fn derive_didactic_cases_render_steps_without_redundant_single_substeps() {
    let cases = derive_audit_cases(false);
    assert_cases_render_without_redundant_single_substeps(&cases);
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
    assert_case_step_titles(
        "factor_perfect_square_trinomial",
        "Factorizar",
        &["Usar a^2 + 2ab + b^2 = (a + b)^2"],
    );
}

#[test]
fn derive_didactic_symbolic_perfect_square_factorization_explains_pattern() {
    assert_case_step_titles(
        "factor_perfect_square_trinomial_symbolic",
        "Factorizar",
        &["Usar a^2 + 2ab + b^2 = (a + b)^2"],
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

    assert_eq!(titles, vec!["Usar a^3 + 3a^2b + 3ab^2 + b^3 = (a + b)^3"]);
}

#[test]
fn derive_didactic_negative_perfect_square_factorization_explains_pattern() {
    assert_case_step_titles(
        "factor_perfect_square_trinomial_minus",
        "Factorizar",
        &["Usar a^2 - 2ab + b^2 = (a - b)^2"],
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

    assert_eq!(titles, vec!["Usar a^3 - 3a^2b + 3ab^2 - b^3 = (a - b)^3"]);
}

#[test]
fn derive_didactic_geometric_difference_factorization_explains_series_identity() {
    let case = DeriveCase {
        id: "factor_geometric_difference_power_5_inline".to_string(),
        family: "factor".to_string(),
        source: "x^5 - 1".to_string(),
        target: "(x-1)*(x^4 + x^3 + x^2 + x + 1)".to_string(),
        expected_status: "derived".to_string(),
    };
    let artifact = audit_case(&case);
    let step = step_by_rule(&artifact, "Factorizar");
    assert_eq!(
        step_substep_titles(step),
        ["Aquí la diferencia de potencias usa base x y exponente 5"],
        "expected geometric-difference factorization to identify the concrete pattern"
    );
    assert_case_step_titles(
        "factor_geometric_difference_power_6",
        "Factorizar",
        &["Aquí la diferencia de potencias usa base x y exponente 6"],
    );
    assert_case_step_titles(
        "factor_full_cyclotomic_sixth_power_difference",
        "Factorizar",
        &["Aquí la diferencia de sexto grado se factoriza completamente con base x"],
    );
}

#[test]
fn derive_didactic_difference_of_squares_factorization_identifies_bases() {
    assert_case_step_titles(
        "factor_difference_squares",
        "Factorizar",
        &["Aquí la diferencia de cuadrados usa bases a y b"],
    );
    assert_case_step_titles(
        "factor_difference_squares_with_passthrough",
        "Factorizar",
        &["Aquí la diferencia de potencias usa base x y exponente 2"],
    );
}

#[test]
fn derive_didactic_sophie_germain_factorization_explains_identity() {
    assert_case_step_titles(
        "factor_sophie_germain",
        "Factorizar",
        &[
            "Convertir la suma en diferencia de cuadrados",
            "Factorizar la diferencia de cuadrados",
        ],
    );
    assert!(
        audit_case(&derive_case_by_id("factor_sophie_germain"))
            .flags
            .is_empty(),
        "`factor_sophie_germain` should be audit-clean after Sophie Germain substeps"
    );
}

#[test]
fn derive_didactic_sophie_germain_expansion_explains_identity() {
    assert_case_step_titles(
        "expand_sophie_germain",
        "Expandir la expresión",
        &[
            "Reconocer el patrón de Sophie Germain",
            "Aplicar la identidad de Sophie Germain",
        ],
    );
    assert!(
        audit_case(&derive_case_by_id("expand_sophie_germain"))
            .flags
            .is_empty(),
        "`expand_sophie_germain` should be audit-clean after Sophie Germain substeps"
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
    assert_case_step_has_no_substeps("expand_symbolic_binomial_minus", "Expandir binomio");
}

#[test]
fn derive_didactic_symbolic_binomial_cube_expansion_explains_the_identity() {
    assert_case_step_has_no_substeps("expand_symbolic_binomial_cube", "Expandir binomio");
}

#[test]
fn derive_didactic_negative_symbolic_binomial_cube_expansion_explains_the_identity() {
    assert_case_step_has_no_substeps("expand_symbolic_binomial_cube_minus", "Expandir binomio");
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
fn derive_didactic_direct_binomial_expansions_are_audit_clean_without_padding() {
    for case_id in [
        "expand_binomial",
        "expand_symbolic_binomial",
        "expand_fractional_binomial_square",
        "expand_symbolic_binomial_cube",
        "expand_symbolic_binomial_minus",
        "expand_symbolic_binomial_cube_minus",
        "expand_symbolic_trinomial_square",
        "expand_symbolic_signed_trinomial_square",
        "expand_symbolic_trinomial_cube",
    ] {
        let artifact = audit_case(&derive_case_by_id(case_id));
        assert!(
            artifact.flags.is_empty(),
            "expected direct binomial expansion `{case_id}` to stay audit-clean without padded substeps; cli=\n{}\nflags={:?}",
            run_cli_lines(&derive_case_by_id(case_id)).join("\n"),
            artifact.flags
        );
    }
}

#[test]
fn derive_didactic_alternating_cubic_vandermonde_factorization_explains_zero_factors_then_linear_part(
) {
    assert_case_step_titles(
        "factor_alternating_cubic_vandermonde",
        "Factorizar",
        &[
            "Si a = b, aparece el factor a - b",
            "Si a = c, aparece el factor a - c",
            "Si b = c, aparece el factor b - c",
            "El cociente restante es a + b + c",
        ],
    );
    assert_case_has_no_no_web_substeps_flag("factor_alternating_cubic_vandermonde");
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
    assert_case_has_no_no_web_substeps_flag("pythagorean_identity");
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
    let artifact = audit_case(&derive_case_by_id("expand_trig_sec_reciprocal"));
    assert!(
        artifact.flags.is_empty(),
        "direct secant expansion should be audit-clean without padding: {:?}",
        artifact.flags
    );

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Reescribir secante como recíproco del coseno")
        })
        .expect("expected secant reciprocal expansion step");

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
}

#[test]
fn derive_didactic_cosecant_reciprocal_expansion_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("expand_trig_csc_reciprocal"));
    assert!(
        artifact.flags.is_empty(),
        "direct cosecant expansion should be audit-clean without padding: {:?}",
        artifact.flags
    );

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Reescribir cosecante como recíproco del seno")
        })
        .expect("expected cosecant reciprocal expansion step");

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
}

#[test]
fn derive_didactic_cotangent_quotient_expansion_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("expand_trig_cot_quotient"));
    assert!(
        artifact.flags.is_empty(),
        "direct cotangent expansion should be audit-clean without padding: {:?}",
        artifact.flags
    );

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Reescribir cotangente como coseno entre seno")
        })
        .expect("expected cotangent quotient expansion step");

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
}

#[test]
fn derive_didactic_reciprocal_cosine_contraction_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_sec_reciprocal"));
    assert!(
        artifact.flags.is_empty(),
        "direct secant contraction should be audit-clean without padding: {:?}",
        artifact.flags
    );

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Reconocer secante desde un recíproco")
        })
        .expect("expected reciprocal secant contraction step");

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
}

#[test]
fn derive_didactic_reciprocal_sine_contraction_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_csc_reciprocal"));
    assert!(
        artifact.flags.is_empty(),
        "direct cosecant contraction should be audit-clean without padding: {:?}",
        artifact.flags
    );

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Reconocer cosecante desde un recíproco")
        })
        .expect("expected reciprocal cosecant contraction step");

    let titles = step_substep_titles(step);
    assert!(titles.is_empty());
}

#[test]
fn derive_didactic_cotangent_quotient_contraction_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("contract_trig_cot_quotient"));
    assert!(
        artifact.flags.is_empty(),
        "direct cotangent contraction should be audit-clean without padding: {:?}",
        artifact.flags
    );

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Reconocer cotangente desde un cociente")
        })
        .expect("expected cotangent quotient contraction step");

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
fn derive_didactic_tangent_quotient_expansion_uses_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("expand_trig_tan_to_sin_cos"));
    assert!(
        artifact.flags.is_empty(),
        "direct tangent expansion should be audit-clean without padding: {:?}",
        artifact.flags
    );

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir tangente como seno entre coseno")
        })
        .expect("expected tangent quotient expansion step");

    assert!(step_substep_titles(step).is_empty());
}

#[test]
fn derive_didactic_reciprocal_trig_product_to_one_uses_direct_identity_language() {
    for (case_id, expected_rule) in [
        (
            "reciprocal_trig_product_to_one",
            "Reconocer tangente por cotangente como 1",
        ),
        (
            "reciprocal_trig_sin_csc_product_to_one",
            "Reconocer seno por cosecante como 1",
        ),
        (
            "reciprocal_trig_cos_sec_product_to_one",
            "Reconocer coseno por secante como 1",
        ),
    ] {
        let artifact = audit_case(&derive_case_by_id(case_id));
        let step = artifact
            .json_steps
            .iter()
            .find(|step| {
                step.get("rule")
                    .and_then(Value::as_str)
                    .is_some_and(|rule| rule == expected_rule)
            })
            .unwrap_or_else(|| panic!("expected reciprocal product step for {case_id}"));

        let titles = step_substep_titles(step);
        assert!(
            titles.is_empty(),
            "{case_id} should not need padded substeps"
        );
    }
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
        .into_iter()
        .flatten()
        .filter_map(|substep| substep.get("title").and_then(Value::as_str))
        .collect();

    assert!(
        titles.is_empty(),
        "expected a direct inverse tangent identity step, got {:?}",
        titles
    );
    assert!(
        artifact.flags.is_empty(),
        "direct inverse tangent identity should not be flagged as opaque: {:?}",
        artifact.flags
    );
}

#[test]
fn derive_didactic_inverse_trig_sum_uses_complement_language() {
    let artifact = audit_case(&derive_case_by_id(
        "inverse_trig_arcsin_arccos_complement_sum",
    ));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar identidad complementaria arcsin/arccos")
        })
        .expect("expected inverse trig complement sum step");

    assert_eq!(
        step_substep_titles(step),
        vec!["Aquí arcsin(x) y arccos(x) suman pi/2"]
    );
    assert!(
        artifact.flags.is_empty(),
        "inverse trig complement sum should not be flagged: {:?}",
        artifact.flags
    );

    let cli_output = run_cli_lines(&derive_case_by_id(
        "inverse_trig_arcsin_arccos_complement_sum",
    ))
    .join("\n");
    assert!(
        cli_output.contains("[Aplicar identidad complementaria arcsin/arccos]"),
        "expected visible inverse-trig complement rule suffix in CLI output:\n{cli_output}"
    );
    assert!(
        !cli_output.contains("[Inverse Trig Sum Identity]"),
        "raw internal inverse-trig rule suffix leaked into CLI output:\n{cli_output}"
    );
}

#[test]
fn derive_didactic_log_exp_inverse_uses_human_direct_identity_language() {
    let artifact = audit_case(&derive_case_by_id("log_exp_inverse_ln_exp"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Cancelar logaritmo natural y exponencial inversos")
        })
        .expect("expected log-exp inverse step");

    assert_eq!(step_substep_titles(step), Vec::<&str>::new());
    assert!(
        artifact.flags.is_empty(),
        "direct log-exp inverse should not be flagged as opaque: {:?}",
        artifact.flags
    );
}

#[test]
fn derive_didactic_log_exp_power_inverse_has_visible_normalization_substep() {
    let artifact = audit_case(&derive_case_by_id("log_exp_inverse_ln_exp_power"));

    let power_step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Multiplicar exponentes")
        })
        .expect("expected power normalization step");

    assert_eq!(
        step_substep_titles(power_step),
        vec!["Usar (e^A)^n = e^(n·A)"]
    );
    assert!(artifact.json_steps.iter().any(|step| {
        step.get("rule")
            .and_then(Value::as_str)
            .is_some_and(|rule| rule == "Cancelar logaritmo natural y exponencial inversos")
    }));
    assert!(
        artifact.flags.is_empty(),
        "log-exp power inverse should not be flagged as opaque: {:?}",
        artifact.flags
    );
}

#[test]
fn derive_didactic_log_exp_product_inverse_shows_post_expansion_cancellation() {
    let artifact = audit_case(&derive_case_by_id("log_exp_inverse_ln_exp_product"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Expandir logaritmos")
        })
        .expect("expected expand-log step");

    assert_eq!(
        step_substep_titles(step),
        vec!["Cancelar cada logaritmo natural con su exponencial"]
    );
    assert!(
        artifact.flags.is_empty(),
        "log-exp product inverse should not be flagged as opaque: {:?}",
        artifact.flags
    );
}

#[test]
fn derive_didactic_inverse_hyperbolic_log_identity_has_concrete_substeps() {
    assert_case_step_titles(
        "inverse_hyperbolic_atanh_square_ratio_log",
        "Convertir tangente hiperbólica inversa en logaritmo",
        &["Identificar el argumento como (u^2 - 1)/(u^2 + 1)"],
    );
    assert_case_has_no_no_web_substeps_flag("inverse_hyperbolic_atanh_square_ratio_log");
}

#[test]
fn derive_didactic_hyperbolic_composition_has_concrete_substeps() {
    assert_case_step_titles(
        "hyperbolic_composition_sinh_asinh",
        "Cancelar funciones hiperbólicas inversas",
        &["Usar que sinh y asinh son funciones inversas", "Aquí u = x"],
    );
    assert_case_has_no_no_web_substeps_flag("hyperbolic_composition_sinh_asinh");
}

#[test]
fn derive_didactic_hyperbolic_special_value_is_direct_and_unflagged() {
    let artifact = audit_case(&derive_case_by_id("hyperbolic_special_value_sinh_zero"));

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Evaluar valor hiperbólico especial")
        })
        .expect("expected hyperbolic special value step");

    assert_eq!(step_substep_titles(step), Vec::<&str>::new());
    assert!(
        artifact.flags.is_empty(),
        "special hyperbolic value should stay direct and unflagged: {:?}",
        artifact.flags
    );
}

#[test]
fn derive_didactic_trig_special_value_is_direct_and_unflagged() {
    for case_id in [
        "trig_special_value_sin_zero",
        "reciprocal_trig_special_value_sec_pi_fourth",
        "inverse_trig_special_value_arctan_sqrt_three",
        "trig_special_value_cos_two_pi_thirds_negative_half",
    ] {
        let artifact = audit_case(&derive_case_by_id(case_id));

        let step = artifact
            .json_steps
            .iter()
            .find(|step| {
                step.get("rule")
                    .and_then(Value::as_str)
                    .is_some_and(|rule| rule == "Evaluar valor trigonométrico especial")
            })
            .expect("expected trig special value step");

        assert_eq!(step_substep_titles(step), Vec::<&str>::new());
        assert!(
            artifact.flags.is_empty(),
            "special trig value `{case_id}` should stay direct and unflagged: {:?}",
            artifact.flags
        );
    }
}

#[test]
fn derive_didactic_arcsin_arctan_composition_uses_inverse_trig_language() {
    let case = derive_case_by_id("arcsin_sin_arctan_safe_composition");
    let artifact = audit_case(&case);

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar composición trigonométrica inversa")
        })
        .expect("expected inverse-trig composition step");

    assert_eq!(
        step_substep_titles(step),
        [
            "Reconocer x/sqrt(1+x^2) como sin(arctan(x))",
            "Sustituir dentro de arcsin",
            "Usar asin(sin(u)) = u en el rango principal"
        ]
    );
    let cli_output = run_cli_lines(&case).join("\n");
    assert!(
        cli_output.contains("[Aplicar composición trigonométrica inversa]"),
        "expected visible inverse-trig composition rule suffix in CLI output:\n{cli_output}"
    );
    assert!(
        !cli_output.contains("[Inverse Trig Composition]"),
        "raw internal inverse-trig composition rule suffix leaked into CLI output:\n{cli_output}"
    );
}

#[test]
fn derive_didactic_direct_inverse_trig_composition_uses_inverse_language() {
    let case = derive_case_by_id("inverse_trig_composition_sin_arcsin");
    let artifact = audit_case(&case);

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Aplicar composición trigonométrica inversa")
        })
        .expect("expected inverse-trig composition step");

    assert_eq!(
        step_substep_titles(step),
        ["Usar que sin y arcsin son funciones inversas", "Aquí u = x"]
    );
    assert_case_has_no_no_web_substeps_flag("inverse_trig_composition_sin_arcsin");
    let cli_output = run_cli_lines(&case).join("\n");
    assert!(
        cli_output.contains("[Aplicar composición trigonométrica inversa]"),
        "expected visible inverse-trig composition rule suffix in CLI output:\n{cli_output}"
    );
    assert!(
        !cli_output.contains("[Inverse Trig Composition]"),
        "raw internal inverse-trig composition rule suffix leaked into CLI output:\n{cli_output}"
    );
}

#[test]
fn derive_didactic_arctan_right_triangle_compositions_use_inverse_trig_language() {
    for (case_id, projection_title) in [
        (
            "sin_arctan_right_triangle_projection",
            "Leer el seno desde ese triángulo",
        ),
        (
            "cos_arctan_right_triangle_projection",
            "Leer el coseno desde ese triángulo",
        ),
    ] {
        assert_case_step_titles(
            case_id,
            "Aplicar composición trigonométrica inversa",
            &[
                "Calcular la hipotenusa del triángulo asociado a arctan(x)",
                projection_title,
            ],
        );
        assert_case_has_no_no_web_substeps_flag(case_id);
        let cli_output = run_cli_lines(&derive_case_by_id(case_id)).join("\n");
        assert!(
            cli_output.contains("[Aplicar composición trigonométrica inversa]"),
            "expected visible inverse-trig composition rule suffix in CLI output for {case_id}:\n{cli_output}"
        );
        assert!(
            !cli_output.contains("[Inverse Trig Composition]"),
            "raw internal inverse-trig composition rule suffix leaked into CLI output for {case_id}:\n{cli_output}"
        );
    }
}

#[test]
fn derive_didactic_arcsin_projection_steps_use_inverse_trig_language() {
    for (case_id, inverse_name, projection_title) in [
        (
            "cos_arcsin_complement_projection",
            "arcsin(x)",
            "Leer el coseno desde ese triángulo",
        ),
        (
            "sin_arccos_complement_projection",
            "arccos(x)",
            "Leer el seno desde ese triángulo",
        ),
        (
            "tan_arcsin_tangent_projection",
            "arcsin(x)",
            "Leer la tangente desde ese triángulo",
        ),
    ] {
        let first_title =
            format!("Calcular el cateto restante del triángulo asociado a {inverse_name}");
        assert_case_step_titles(
            case_id,
            "Aplicar composición trigonométrica inversa",
            &[first_title.as_str(), projection_title],
        );
        assert_case_has_no_no_web_substeps_flag(case_id);
        let cli_output = run_cli_lines(&derive_case_by_id(case_id)).join("\n");
        assert!(
            cli_output.contains("[Aplicar composición trigonométrica inversa]"),
            "expected visible inverse-trig composition rule suffix in CLI output for {case_id}:\n{cli_output}"
        );
        assert!(
            !cli_output.contains("[Inverse Trig Composition]"),
            "raw internal inverse-trig composition rule suffix leaked into CLI output for {case_id}:\n{cli_output}"
        );
    }
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
    assert_case_step_has_no_substeps("contract_trig_sin_diff_special", "Aplicar suma a producto");
    assert_case_has_no_no_web_substeps_flag("contract_trig_sin_diff_special");
}

#[test]
fn derive_didactic_general_sine_sum_uses_sum_to_product_directly() {
    assert_case_step_has_no_substeps(
        "expand_trig_sum_to_product_sin_sum_general",
        "Aplicar suma a producto",
    );
    assert_case_has_no_no_web_substeps_flag("expand_trig_sum_to_product_sin_sum_general");
}

#[test]
fn derive_didactic_general_cosine_sum_uses_sum_to_product_directly() {
    assert_case_step_has_no_substeps(
        "expand_trig_sum_to_product_cos_sum_general",
        "Aplicar suma a producto",
    );
    assert_case_has_no_no_web_substeps_flag("expand_trig_sum_to_product_cos_sum_general");
}

#[test]
fn derive_didactic_general_cosine_difference_uses_sum_to_product_directly() {
    assert_case_step_has_no_substeps(
        "expand_trig_sum_to_product_cos_diff_general",
        "Aplicar suma a producto",
    );
    assert_case_has_no_no_web_substeps_flag("expand_trig_sum_to_product_cos_diff_general");
}

#[test]
fn derive_didactic_inverse_trig_double_angle_expansions_show_projection() {
    for case_id in [
        "expand_trig_double_sin_inverse_arcsin",
        "expand_trig_double_cos_inverse_arcsin",
        "expand_trig_double_sin_inverse_arccos",
        "expand_trig_double_cos_inverse_arccos",
        "expand_trig_double_sin_arctan_projection",
    ] {
        assert_case_step_titles(
            case_id,
            "Expandir ángulo doble",
            &[
                "Expandir con la identidad de ángulo doble",
                "Sustituir las razones trigonométricas inversas",
            ],
        );
        assert_case_has_no_no_web_substeps_flag(case_id);
    }
}

#[test]
fn derive_didactic_trig_square_identity_uses_direct_step_without_audit_flag() {
    for case_id in [
        "expand_trig_sin_cos_square_sum",
        "expand_trig_sin_cos_square_diff",
    ] {
        assert_case_step_has_no_substeps(case_id, "Aplicar identidad del cuadrado trigonométrico");
        assert_case_has_no_no_web_substeps_flag(case_id);
    }
}

#[test]
fn derive_didactic_quadruple_angle_cosine_has_formula_substep() {
    assert_case_step_titles(
        "expand_trig_quadruple_angle_cosine",
        "Reescribir ángulo cuádruple",
        &["Usar cos(4u) = 8 · cos(u)^4 - 8 · cos(u)^2 + 1, con u = x"],
    );
    assert_case_has_no_no_web_substeps_flag("expand_trig_quadruple_angle_cosine");
}

#[test]
fn derive_didactic_quadruple_angle_sine_has_formula_substep() {
    for case_id in [
        "expand_trig_quadruple_angle_sine_expanded_product",
        "contract_trig_quadruple_angle_sine_expanded_product",
    ] {
        assert_case_step_titles(
            case_id,
            "Reescribir ángulo cuádruple",
            &["Usar sin(4u) = 4 · sin(u) · cos(u)^3 - 4 · sin(u)^3 · cos(u), con u = x"],
        );
        assert_case_has_no_no_web_substeps_flag(case_id);
    }
}

#[test]
fn derive_didactic_square_double_angle_contraction_has_formula_substep() {
    assert_case_step_titles(
        "contract_trig_square_double_angle_sine_cosine_product",
        "Contraer cuadrado de ángulo doble",
        &["Usar sin²(u)·cos²(u) = sin²(2u) / 4, con u = x"],
    );
    assert_case_has_no_no_web_substeps_flag(
        "contract_trig_square_double_angle_sine_cosine_product",
    );
}

#[test]
fn derive_didactic_difference_of_squares_fraction_cancel_recaps_factor_then_cancel() {
    for case_id in [
        "cancel_fraction_difference_squares",
        "cancel_fraction_difference_squares_with_passthrough",
    ] {
        assert_case_step_titles(
            case_id,
            "Factorizar una diferencia de cuadrados y cancelar",
            &[
                "Factorizar el numerador como diferencia de cuadrados",
                "Ahora se cancela el factor a - b",
            ],
        );

        let cli_output = run_cli_lines(&derive_case_by_id(case_id)).join("\n");
        assert!(
            cli_output.contains("[Factorizar una diferencia de cuadrados y cancelar]"),
            "derive CLI should show the visible difference-of-squares cancel rule name for {case_id}: {cli_output}"
        );
        assert!(
            !cli_output.contains("[Pre-order Difference of Squares Cancel]"),
            "derive CLI should not expose the internal difference-of-squares cancel rule name for {case_id}: {cli_output}"
        );
    }

    assert!(
        audit_case(&derive_case_by_id("cancel_fraction_difference_squares"))
            .flags
            .is_empty()
    );
    assert!(audit_case(&derive_case_by_id(
        "cancel_fraction_difference_squares_with_passthrough"
    ))
    .flags
    .is_empty());
}

#[test]
fn derive_didactic_difference_of_cubes_fraction_cancel_recaps_factor_then_cancel() {
    for case_id in [
        "cancel_fraction_difference_cubes",
        "cancel_fraction_difference_cubes_with_passthrough",
    ] {
        let case = derive_case_by_id(case_id);
        let artifact = audit_case(&case);

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
                "Ahora se cancela el factor (a - b)",
                "Reemplazar ese bloque en la expresión"
            ]
        );
        let cli_output = run_cli_lines(&case).join("\n");
        assert!(
            cli_output.contains("[Factorizar cubos y cancelar]"),
            "expected visible cubes rule suffix in CLI output for {case_id}:\n{cli_output}"
        );
        assert!(
            !cli_output.contains("[Cancel Sum/Difference of Cubes Fraction]"),
            "raw internal cubes rule suffix leaked into CLI output for {case_id}:\n{cli_output}"
        );
    }
}

#[test]
fn derive_didactic_sum_of_cubes_fraction_cancel_recaps_factor_then_cancel() {
    let case = derive_case_by_id("cancel_fraction_sum_cubes");
    let artifact = audit_case(&case);

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
            "Ahora se cancela el factor (a + b)",
            "Reemplazar ese bloque en la expresión"
        ]
    );
    let cli_output = run_cli_lines(&case).join("\n");
    assert!(
        cli_output.contains("[Factorizar cubos y cancelar]"),
        "expected visible cubes rule suffix in CLI output:\n{cli_output}"
    );
    assert!(
        !cli_output.contains("[Cancel Sum/Difference of Cubes Fraction]"),
        "raw internal cubes rule suffix leaked into CLI output:\n{cli_output}"
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

    let substeps = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected perfect-square cancellation substeps");
    assert_eq!(
        substeps[1].get("before_latex").and_then(Value::as_str),
        Some("\\frac{{(x + 1)}^{2}}{x + 1}")
    );
    assert_eq!(
        substeps[1].get("after_latex").and_then(Value::as_str),
        Some("x + 1")
    );
}

#[test]
fn derive_didactic_perfect_square_fraction_cancel_minus_symbolic_uses_repeated_factor_rule() {
    let case = derive_case_by_id("cancel_fraction_perfect_square_minus_symbolic");
    let artifact = audit_case(&case);

    let step = artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| rule == "Cancelar un cuadrado perfecto con el mismo binomio")
        })
        .expect("expected symbolic negative perfect-square fraction cancel step");
    assert!(
        artifact
            .json_steps
            .iter()
            .all(|step| step.get("rule").and_then(Value::as_str)
                != Some("Pre-order Perfect Square Minus Cancel")),
        "derive web/json should not expose the internal perfect-square-minus rule name"
    );

    let cli_output = run_cli_lines(&case).join("\n");
    assert!(
        cli_output.contains("[Cancelar un cuadrado perfecto con el mismo binomio]"),
        "derive CLI should show the visible perfect-square-minus rule name: {cli_output}"
    );
    assert!(
        !cli_output.contains("[Pre-order Perfect Square Minus Cancel]"),
        "derive CLI should not expose the internal perfect-square-minus rule name: {cli_output}"
    );

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

    let substeps = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected symbolic negative perfect-square cancellation substeps");
    assert_eq!(
        substeps[1].get("before_latex").and_then(Value::as_str),
        Some("\\frac{{(a - b)}^{2}}{a - b}")
    );
    assert_eq!(
        substeps[1].get("after_latex").and_then(Value::as_str),
        Some("a - b")
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
fn derive_didactic_choose_number_theory_cases_explain_binomial_identities() {
    for (case_id, rule, expected_titles) in [
        (
            "choose_numeric_binomial_coefficient",
            "Calcular coeficiente binomial",
            &[
                "Usar C(5,2) = 5! / (2! · 3!)",
                "Calcular 5! / (2! · 3!) = 10",
            ][..],
        ),
        (
            "choose_numeric_pascal_identity",
            "Aplicar identidad de Pascal",
            &["Usar C(4,1) + C(4,2) = C(5,2)"][..],
        ),
        (
            "choose_numeric_symmetry",
            "Aplicar simetría del coeficiente binomial",
            &["Usar C(6,1) = C(6,6-1)", "Calcular 6-1 = 5"][..],
        ),
    ] {
        assert_case_step_titles(case_id, rule, expected_titles);
        let artifact = audit_case(&derive_case_by_id(case_id));
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "no web substeps emitted"),
            "{case_id} should explain the number-theory transformation"
        );
    }
}

#[test]
fn derive_didactic_exponential_log_product_shows_split_then_cancel() {
    assert_case_step_titles(
        "collapse_exponential_log_product",
        "Reescribir exponenciales",
        &[
            "Separar la suma o resta del exponente en productos de exponenciales",
            "Cancelar e^(k·ln(u)) como potencia en cada factor",
        ],
    );
}

#[test]
fn derive_didactic_scaled_exponential_log_product_shows_powers_after_cancellation() {
    assert_case_step_titles(
        "collapse_exponential_scaled_log_product",
        "Reescribir exponenciales",
        &[
            "Separar la suma o resta del exponente en productos de exponenciales",
            "Cancelar e^(k·ln(u)) como potencia en cada factor",
        ],
    );

    let artifact = audit_case(&derive_case_by_id(
        "collapse_exponential_scaled_log_product",
    ));
    let step = step_by_rule(&artifact, "Reescribir exponenciales");
    let substeps = step
        .get("substeps")
        .and_then(Value::as_array)
        .expect("expected exponential-log substeps");
    assert_eq!(
        substeps[0].get("after_latex").and_then(Value::as_str),
        Some("e^{2\\cdot \\ln(x)}\\cdot e^{3\\cdot \\ln(y)}")
    );
    assert_eq!(
        substeps[1].get("after_latex").and_then(Value::as_str),
        Some("{x}^{2}\\cdot {y}^{3}")
    );
}

#[test]
fn derive_didactic_log_inverse_power_unary_alias_stays_explainable() {
    assert_case_step_titles(
        "log_inverse_power_unary_natural_alias",
        "Convertir potencia logarítmica inversa",
        &[
            "Usar que e^(ln(u)) = u",
            "El exponente exterior cancela el ln del exponente interior",
        ],
    );
    assert_case_has_no_no_web_substeps_flag("log_inverse_power_unary_natural_alias");
}

#[test]
fn derive_didactic_natural_log_power_alias_shows_inverse_then_outer_exponent() {
    assert_case_step_titles(
        "log_exp_inverse_natural_log_power_alias",
        "Cancelar exponencial con logaritmo y conservar exponente",
        &[
            "Usar que e^(ln(u)) = u",
            "Aplicar el factor exterior como exponente",
        ],
    );
    assert_case_has_no_no_web_substeps_flag("log_exp_inverse_natural_log_power_alias");
}

#[test]
fn derive_didactic_log10_power_alias_shows_inverse_then_outer_exponent() {
    assert_case_step_titles(
        "log_exp_inverse_log10_power_alias",
        "Cancelar exponencial con logaritmo y conservar exponente",
        &[
            "Usar que 10^(log10(u)) = u",
            "Aplicar el factor exterior como exponente",
        ],
    );
    assert_case_has_no_no_web_substeps_flag("log_exp_inverse_log10_power_alias");
}

#[test]
fn derive_didactic_basic_exponential_laws_show_concrete_identities() {
    for (case_id, rule, expected_title) in [
        (
            "contract_exponential_sum",
            "Reescribir exponenciales",
            "Usar e^A · e^B = e^(A+B)",
        ),
        (
            "expand_exponential_sum",
            "Reescribir exponenciales",
            "Usar e^(A+B) = e^A · e^B",
        ),
        (
            "contract_exponential_difference",
            "Reescribir exponenciales",
            "Usar e^A / e^B = e^(A-B)",
        ),
        (
            "contract_exponential_reciprocal",
            "Reescribir recíproco exponencial",
            "Usar 1/e^A = e^(-A)",
        ),
        (
            "expand_exponential_reciprocal",
            "Reescribir recíproco exponencial",
            "Usar e^(-A) = 1/e^A",
        ),
        (
            "contract_exponential_power",
            "Reescribir potencia exponencial",
            "Usar (e^A)^n = e^(n·A)",
        ),
        (
            "expand_exponential_power",
            "Reescribir potencia exponencial",
            "Usar e^(n·A) = (e^A)^n",
        ),
    ] {
        assert_case_step_titles(case_id, rule, &[expected_title]);
    }
}

#[test]
fn derive_didactic_even_power_log_expansion_stays_direct_without_redundant_substep() {
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
        .map(|substeps| {
            substeps
                .iter()
                .filter_map(|substep| substep.get("title").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();

    assert!(
        titles.is_empty(),
        "even-power log expansion step should stay direct, got substeps: {titles:?}"
    );
    assert_case_has_no_no_web_substeps_flag("expand_log_even_power_abs");
}

#[test]
fn derive_didactic_general_base_log_power_expansion_stays_direct_without_redundant_substep() {
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
        .map(|substeps| {
            substeps
                .iter()
                .filter_map(|substep| substep.get("title").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();

    assert!(
        titles.is_empty(),
        "general-base log power expansion step should stay direct, got substeps: {titles:?}"
    );
    assert_case_has_no_no_web_substeps_flag("expand_log_general_base_power");
}

#[test]
fn derive_didactic_even_power_log_contraction_stays_direct_without_substeps() {
    assert_case_step_has_no_substeps("contract_log_even_power_abs", "Contraer logaritmos");
    assert_case_has_no_no_web_substeps_flag("contract_log_even_power_abs");
}

#[test]
fn derive_didactic_general_base_log_power_contraction_stays_direct_without_substeps() {
    assert_case_step_has_no_substeps("contract_log_general_base_power", "Contraer logaritmos");
    assert_case_has_no_no_web_substeps_flag("contract_log_general_base_power");
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
fn derive_didactic_scaled_log_sum_contraction_stays_direct_without_substeps() {
    assert_case_step_has_no_substeps("contract_log_sum_with_scaled_powers", "Contraer logaritmos");
}

#[test]
fn derive_didactic_scaled_general_base_log_difference_contraction_stays_direct_without_substeps() {
    assert_case_step_has_no_substeps(
        "contract_log_general_base_difference_with_scaled_powers",
        "Contraer logaritmos",
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
fn derive_didactic_direct_log_change_of_base_cases_expose_components() {
    for (case_id, expected_titles) in [
        (
            "expand_log_change_of_base_direct",
            &[
                "Poner el argumento en el numerador",
                "Poner la base en el denominador",
            ][..],
        ),
        (
            "contract_log_change_of_base_direct",
            &[
                "Leer el argumento desde el numerador",
                "Leer la base desde el denominador",
            ][..],
        ),
    ] {
        assert_case_step_titles(case_id, "Aplicar cambio de base", expected_titles);

        let cli_output = run_cli_lines(&derive_case_by_id(case_id)).join("\n");
        assert!(
            cli_output.contains("[Aplicar cambio de base]"),
            "derive CLI should show the visible change-of-base rule name for {case_id}: {cli_output}"
        );
        assert!(
            !cli_output.contains("[Change of Base]"),
            "derive CLI should not expose the internal change-of-base rule name for {case_id}: {cli_output}"
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
fn derive_didactic_half_scaled_sine_double_angle_contracts_directly() {
    assert_case_step_has_no_substeps(
        "contract_trig_half_scaled_double_sin",
        "Expandir ángulo doble",
    );
    assert_case_has_no_no_web_substeps_flag("contract_trig_half_scaled_double_sin");
}

#[test]
fn derive_didactic_trig_angle_sum_diff_explains_direct_identities() {
    for (case_id, expected_title) in [
        (
            "expand_trig_angle_sum_sine",
            "Usar sin(A+B) = sin(A) · cos(B) + cos(A) · sin(B)",
        ),
        (
            "expand_trig_angle_sum_cosine",
            "Usar cos(A+B) = cos(A) · cos(B) - sin(A) · sin(B)",
        ),
        (
            "contract_trig_angle_diff_sine",
            "Usar sin(A-B) = sin(A) · cos(B) - cos(A) · sin(B)",
        ),
        (
            "contract_trig_angle_diff_cosine",
            "Usar cos(A-B) = cos(A) · cos(B) + sin(A) · sin(B)",
        ),
    ] {
        assert_case_step_titles(
            case_id,
            "Aplicar suma/diferencia de ángulos",
            &[expected_title],
        );
    }
}

#[test]
fn derive_didactic_tangent_angle_sum_diff_routes_use_specific_identity() {
    for case_id in [
        "expand_trig_tangent_angle_sum",
        "contract_trig_tangent_angle_sum",
        "expand_trig_tangent_angle_difference",
        "contract_trig_tangent_angle_difference",
    ] {
        assert_case_step_has_no_substeps(
            case_id,
            "Aplicar identidad de tangente de suma/diferencia de ángulos",
        );
        assert_case_has_no_no_web_substeps_flag(case_id);
    }
}

#[test]
fn derive_didactic_phase_shift_explains_contract_and_expand_templates() {
    for (case_id, expected_title) in [
        (
            "contract_trig_phase_shift_sum_to_shifted_sine",
            "Usar a·sin(u) + b·cos(u) = R·sin(u + φ)",
        ),
        (
            "expand_trig_phase_shift_shifted_sine_to_sum",
            "Expandir R·sin(u + φ)",
        ),
    ] {
        assert_case_step_titles(case_id, "Aplicar identidad de desfase", &[expected_title]);
    }
}

#[test]
fn derive_didactic_phase_shift_shifted_sine_to_shifted_cosine_explains_complementary_angle() {
    for case_id in [
        "contract_trig_phase_shift_shifted_sine_to_shifted_cosine",
        "contract_trig_phase_shift_general_shifted_sine_to_shifted_cosine",
    ] {
        assert_case_step_titles(
            case_id,
            "Aplicar identidad de desfase",
            &["Usar sin(u + φ) = cos(u - (π/2 - φ))"],
        );
    }
}

#[test]
fn derive_didactic_phase_shift_passthrough_focuses_changed_block() {
    for case_id in [
        "contract_trig_phase_shift_shifted_terms_with_passthrough",
        "contract_trig_phase_shift_general_shifted_terms_with_passthrough",
        "contract_trig_phase_shift_sum_to_shifted_sine_with_passthrough",
        "contract_trig_phase_shift_general_sum_to_shifted_sine_with_passthrough",
    ] {
        assert_case_step_titles(
            case_id,
            "Aplicar identidad de desfase",
            &["Aplicar la identidad de desfase al bloque que cambia"],
        );
    }
}

#[test]
fn derive_didactic_cofunction_identity_uses_visible_rule_without_padding() {
    for case_id in [
        "expand_trig_cofunction_sine_minus",
        "expand_trig_cofunction_cosine_minus",
    ] {
        let artifact = audit_case(&derive_case_by_id(case_id));
        let step = step_by_rule(&artifact, "Aplicar identidad de cofunción");
        assert!(
            step_substep_titles(step).is_empty(),
            "{case_id}: cofunction identity should stay direct without redundant substeps"
        );
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "no web substeps emitted"),
            "{case_id}: cofunction identity should be accepted as a self-explanatory direct step"
        );
    }
}

#[test]
fn derive_didactic_hyperbolic_angle_sum_diff_explains_direct_identities() {
    for (case_id, expected_title) in [
        (
            "expand_hyperbolic_sinh_sum",
            "Usar sinh(A+B) = sinh(A) · cosh(B) + cosh(A) · sinh(B)",
        ),
        (
            "contract_hyperbolic_sinh_difference",
            "Usar sinh(A-B) = sinh(A) · cosh(B) - cosh(A) · sinh(B)",
        ),
        (
            "expand_hyperbolic_cosh_sum",
            "Usar cosh(A+B) = cosh(A) · cosh(B) + sinh(A) · sinh(B)",
        ),
        (
            "contract_hyperbolic_cosh_difference",
            "Usar cosh(A-B) = cosh(A) · cosh(B) - sinh(A) · sinh(B)",
        ),
        (
            "expand_hyperbolic_tanh_sum",
            "Usar tanh(A+B) = (tanh(A) + tanh(B)) / (1 + tanh(A)·tanh(B))",
        ),
        (
            "contract_hyperbolic_tanh_sum",
            "Usar tanh(A+B) = (tanh(A) + tanh(B)) / (1 + tanh(A)·tanh(B))",
        ),
        (
            "expand_hyperbolic_tanh_difference",
            "Usar tanh(A-B) = (tanh(A) - tanh(B)) / (1 - tanh(A)·tanh(B))",
        ),
        (
            "contract_hyperbolic_tanh_difference",
            "Usar tanh(A-B) = (tanh(A) - tanh(B)) / (1 - tanh(A)·tanh(B))",
        ),
    ] {
        assert_case_step_titles(
            case_id,
            "Aplicar identidad hiperbólica de suma/diferencia de ángulos",
            &[expected_title],
        );
    }
}

#[test]
fn derive_didactic_hyperbolic_recursive_six_explains_angle_addition_decomposition() {
    for (case_id, expected_title) in [
        (
            "expand_recursive_hyperbolic_sinh_sum",
            "Usar sinh(5u+u) = sinh(5u) · cosh(u) + cosh(5u) · sinh(u), con u = x",
        ),
        (
            "expand_recursive_hyperbolic_cosh_sum",
            "Usar cosh(5u+u) = cosh(5u) · cosh(u) + sinh(5u) · sinh(u), con u = x",
        ),
    ] {
        assert_case_step_titles(
            case_id,
            "Aplicar identidad hiperbólica de suma/diferencia de ángulos",
            &[expected_title],
        );
    }
}

#[test]
fn derive_didactic_hyperbolic_product_to_sum_explains_direct_identities() {
    for (case_id, expected_title) in [
        (
            "expand_hyperbolic_sinh_sum_to_product_exact",
            "Usar sinh(A)+sinh(B) = 2·sinh((A+B)/2)·cosh((A-B)/2)",
        ),
        (
            "expand_hyperbolic_cosh_sum_to_product_exact",
            "Usar cosh(A)+cosh(B) = 2·cosh((A+B)/2)·cosh((A-B)/2)",
        ),
        (
            "expand_hyperbolic_cosh_difference_to_product_exact",
            "Usar cosh(A)-cosh(B) = 2·sinh((A+B)/2)·sinh((A-B)/2)",
        ),
        (
            "expand_hyperbolic_product_to_sum_to_sinh_cubic_polynomial",
            "Usar 2·sinh(A)·cosh(B) = sinh(A+B) + sinh(A-B)",
        ),
        (
            "expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial_with_passthrough",
            "Usar 2·sinh(A)·sinh(B) = cosh(A+B) - cosh(A-B)",
        ),
    ] {
        assert_case_step_titles(
            case_id,
            "Aplicar identidad hiperbólica de producto a suma",
            &[expected_title],
        );
    }
}

#[test]
fn derive_didactic_direct_hyperbolic_identity_rules_use_visible_titles_without_padding() {
    for (case_id, expected_rule) in [
        (
            "hyperbolic_contract_sinh_double_angle",
            "Aplicar identidad hiperbólica de ángulo doble",
        ),
        (
            "hyperbolic_contract_sinh_triple_angle",
            "Aplicar identidad hiperbólica de ángulo triple",
        ),
        (
            "hyperbolic_expand_sinh_to_exp_definition",
            "Aplicar identidad exponencial hiperbólica",
        ),
        (
            "hyperbolic_pythagorean_identity",
            "Aplicar identidad pitagórica hiperbólica",
        ),
        (
            "hyperbolic_tanh_pythagorean_reverse",
            "Aplicar identidad pitagórica hiperbólica",
        ),
    ] {
        assert_case_step_has_no_substeps(case_id, expected_rule);
        let artifact = audit_case(&derive_case_by_id(case_id));
        assert!(
            !artifact
                .flags
                .iter()
                .any(|flag| flag == "no web substeps emitted"),
            "{case_id} should be accepted as a direct self-explanatory identity step"
        );
    }
}

#[test]
fn derive_didactic_hyperbolic_negative_parity_uses_specific_identity() {
    assert_case_step_titles(
        "hyperbolic_negative_tanh_parity",
        "Aplicar paridad hiperbólica",
        &["Usar que una función impar cumple f(-u) = -f(u)"],
    );
}

#[test]
fn derive_didactic_hyperbolic_quotient_uses_specific_identity() {
    let case_id = "hyperbolic_contract_tanh_quotient";
    assert_case_step_has_no_substeps(case_id, "Reconocer tangente hiperbólica desde un cociente");
    let artifact = audit_case(&derive_case_by_id(case_id));
    assert!(
        artifact.flags.is_empty(),
        "{case_id} should be audit-clean as a direct self-explanatory quotient rewrite: {:?}",
        artifact.flags
    );
}

#[test]
fn derive_didactic_hyperbolic_half_angle_squares_use_specific_identity() {
    for (case_id, expected_substep) in [
        (
            "hyperbolic_half_angle_cosh_forward",
            "Usar cosh²(u/2) = (cosh(u) + 1) / 2",
        ),
        (
            "hyperbolic_half_angle_sinh_forward",
            "Usar sinh²(u/2) = (cosh(u) - 1) / 2",
        ),
    ] {
        assert_case_step_titles(
            case_id,
            "Aplicar identidad hiperbólica de ángulo mitad",
            &[expected_substep],
        );
    }
}

#[test]
fn derive_didactic_trig_product_to_sum_explains_direct_identities() {
    for (case_id, expected_title) in [
        (
            "expand_trig_product_to_sum_cos_cos",
            "Usar 2·cos(A)·cos(B) = cos(A+B) + cos(A-B)",
        ),
        (
            "expand_trig_product_to_sum_cos_sin",
            "Usar 2·cos(A)·sin(B) = sin(A+B) - sin(A-B)",
        ),
        (
            "expand_trig_product_to_sum_sin_cos",
            "Usar 2·sin(A)·cos(B) = sin(A+B) + sin(A-B)",
        ),
        (
            "expand_trig_product_to_sum_sin_sin",
            "Usar 2·sin(A)·sin(B) = cos(A-B) - cos(A+B)",
        ),
        (
            "expand_trig_product_to_sum_to_cosine_difference_polynomial_with_passthrough",
            "Usar 2·sin(A)·sin(B) = cos(A-B) - cos(A+B)",
        ),
    ] {
        assert_case_step_titles(case_id, "Aplicar producto a suma", &[expected_title]);
    }
}

#[test]
fn derive_didactic_trig_negative_parity_uses_specific_identity() {
    assert_case_step_titles(
        "expand_trig_negative_tangent_parity",
        "Aplicar paridad trigonométrica",
        &["Usar que una función impar cumple f(-u) = -f(u)"],
    );
}

#[test]
fn derive_didactic_trig_triple_angle_explains_direct_contract_and_composed_steps() {
    for (case_id, expected_title) in [
        (
            "expand_trig_triple_angle_cosine",
            "Usar cos(3u) = 4 · cos(u)^3 - 3 · cos(u), con u = x",
        ),
        (
            "expand_trig_triple_angle_sine",
            "Usar sin(3u) = 3 · sin(u) - 4 · sin(u)^3, con u = x",
        ),
        (
            "expand_trig_triple_angle_tangent",
            "Usar tan(3u) = (3 · tan(u) - tan(u)^3) / (1 - 3 · tan(u)^2), con u = x",
        ),
        (
            "contract_trig_triple_angle_cosine",
            "Usar cos(3u) = 4 · cos(u)^3 - 3 · cos(u), con u = x",
        ),
        (
            "contract_trig_triple_angle_sine",
            "Usar sin(3u) = 3 · sin(u) - 4 · sin(u)^3, con u = x",
        ),
        (
            "contract_trig_triple_angle_tangent",
            "Usar tan(3u) = (3 · tan(u) - tan(u)^3) / (1 - 3 · tan(u)^2), con u = x",
        ),
        (
            "expand_trig_product_to_sum_to_cosine_difference_polynomial_with_passthrough",
            "Usar cos(3u) = 4 · cos(u)^3 - 3 · cos(u), con u = x",
        ),
        (
            "expand_trig_product_to_sum_to_sine_difference_mixed_polynomial_with_passthrough",
            "Usar sin(3u) = 3 · sin(u) - 4 · sin(u)^3, con u = x",
        ),
    ] {
        assert_case_step_titles(case_id, "Reescribir ángulo triple", &[expected_title]);
    }
}

#[test]
fn derive_didactic_trig_power_reduction_explains_base_identity() {
    for case_id in [
        "expand_trig_sine_fourth_power_reduction",
        "expand_trig_sine_sixth_power_reduction",
        "expand_trig_sine_eighth_power_reduction",
        "expand_trig_sine_tenth_power_reduction",
        "expand_trig_sine_twelfth_power_reduction",
        "expand_trig_sine_fourteenth_power_reduction",
        "expand_trig_sine_sixteenth_power_reduction",
        "expand_trig_sine_eighteenth_power_reduction",
        "expand_trig_sine_twentieth_power_reduction",
        "expand_trig_sine_twenty_second_power_reduction",
        "expand_trig_sine_twenty_fourth_power_reduction",
    ] {
        assert_case_step_titles(
            case_id,
            "Aplicar reducción de potencias",
            &["Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x"],
        );
    }

    for case_id in [
        "expand_trig_cosine_fourth_power_reduction",
        "expand_trig_cosine_sixth_power_reduction",
        "expand_trig_cosine_eighth_power_reduction",
        "expand_trig_cosine_tenth_power_reduction",
        "expand_trig_cosine_twelfth_power_reduction",
        "expand_trig_cosine_fourteenth_power_reduction",
        "expand_trig_cosine_sixteenth_power_reduction",
        "expand_trig_cosine_eighteenth_power_reduction",
        "expand_trig_cosine_twentieth_power_reduction",
        "expand_trig_cosine_twenty_second_power_reduction",
        "expand_trig_cosine_twenty_fourth_power_reduction",
    ] {
        assert_case_step_titles(
            case_id,
            "Aplicar reducción de potencias",
            &["Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x"],
        );
    }

    assert_case_step_titles(
        "expand_trig_sine_cosine_square_product_reduction",
        "Aplicar reducción de potencias",
        &["Usar sin²(u)·cos²(u) = (1 - cos(4u)) / 8, con u = x"],
    );
}

#[test]
fn derive_didactic_trig_quintuple_angle_explains_base_identity() {
    for (case_id, expected_title) in [
        (
            "expand_trig_quintuple_angle_sine",
            "Usar sin(5u) = 5 · sin(u) - 20 · sin(u)^3 + 16 · sin(u)^5, con u = x",
        ),
        (
            "contract_trig_quintuple_angle_sine",
            "Usar sin(5u) = 5 · sin(u) - 20 · sin(u)^3 + 16 · sin(u)^5, con u = x",
        ),
        (
            "expand_trig_quintuple_angle_cosine",
            "Usar cos(5u) = 16 · cos(u)^5 - 20 · cos(u)^3 + 5 · cos(u), con u = x",
        ),
        (
            "contract_trig_quintuple_angle_cosine",
            "Usar cos(5u) = 16 · cos(u)^5 - 20 · cos(u)^3 + 5 · cos(u), con u = x",
        ),
    ] {
        assert_case_step_titles(case_id, "Reescribir ángulo quíntuple", &[expected_title]);
    }
}

#[test]
fn derive_didactic_trig_recursive_six_explains_angle_addition_decomposition() {
    for (case_id, expected_title) in [
        (
            "expand_trig_recursive_six_sine",
            "Usar sin(5u+u) = sin(5u) · cos(u) + cos(5u) · sin(u), con u = x",
        ),
        (
            "contract_trig_recursive_six_sine",
            "Usar sin(5u+u) = sin(5u) · cos(u) + cos(5u) · sin(u), con u = x",
        ),
        (
            "expand_trig_recursive_six_cosine",
            "Usar cos(5u+u) = cos(5u) · cos(u) - sin(5u) · sin(u), con u = x",
        ),
        (
            "contract_trig_recursive_six_cosine",
            "Usar cos(5u+u) = cos(5u) · cos(u) - sin(5u) · sin(u), con u = x",
        ),
    ] {
        assert_case_step_titles(
            case_id,
            "Aplicar suma/diferencia de ángulos",
            &[expected_title],
        );
    }
}

#[test]
fn derive_didactic_hyperbolic_direct_tanh_triple_angle_stays_direct_without_redundant_substep() {
    assert_case_step_has_no_substeps(
        "expand_hyperbolic_tanh_triple_angle",
        "Aplicar identidad hiperbólica de ángulo triple",
    );
}

#[test]
fn derive_didactic_hyperbolic_triple_angle_explains_combined_cosh_steps() {
    // The passthrough variant keeps the discrete triple-angle step (the extra
    // `+a` term blocks the one-pass normalization). The bare variant no longer
    // surfaces it: the soundness fix (cosh(3x)−cosh(x) no longer collapses to 0)
    // lets the product-to-sum output normalize straight to the cubic polynomial
    // in a single step. Follow-up: restore the explicit triple-angle step in the
    // bare derive narrative so this educational identity stays visible there too.
    assert_case_step_titles(
        "expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial_with_passthrough",
        "Aplicar identidad hiperbólica de ángulo triple",
        &["Usar cosh(3·x) = 4·cosh(x)^3 - 3·cosh(x)"],
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
fn derive_didactic_half_angle_tangent_simplified_argument_uses_specific_identity() {
    for case_id in [
        "expand_trig_half_angle_tangent_sin_over_one_plus_cos",
        "expand_trig_half_angle_tangent_one_minus_cos_over_sin",
        "expand_trig_tangent_half_angle_substitution_sine",
    ] {
        assert_case_step_has_no_substeps(case_id, "Aplicar identidad de tangente de ángulo mitad");
        assert_case_has_no_no_web_substeps_flag(case_id);
    }
}

#[test]
fn derive_didactic_tangent_double_angle_routes_use_specific_identity() {
    for case_id in ["expand_trig_double_tangent", "contract_trig_double_tangent"] {
        assert_case_step_has_no_substeps(case_id, "Aplicar identidad de tangente de ángulo doble");
        assert_case_has_no_no_web_substeps_flag(case_id);
    }
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

    let cli_output =
        run_cli_lines(&derive_case_by_id("merge_same_base_symbolic_powers")).join("\n");
    assert!(
        cli_output.contains("[Sumar exponentes de la misma base]"),
        "expected visible power-merge rule suffix in CLI output:\n{cli_output}"
    );
    assert!(
        !cli_output.contains("[Combine powers with same base (n-ary)]"),
        "raw internal power-merge rule suffix leaked into CLI output:\n{cli_output}"
    );
}

#[test]
fn derive_didactic_quotient_power_merge_shows_negative_exponent_then_merge() {
    assert_case_step_titles(
        "merge_same_base_symbolic_quotient_powers",
        "Sumar exponentes de la misma base",
        &[
            "Reescribir la división como potencia negativa",
            "Sumar los exponentes de la misma base",
        ],
    );
    assert_case_has_no_no_web_substeps_flag("merge_same_base_symbolic_quotient_powers");
}

#[test]
fn derive_didactic_representative_root_power_merge_cases_explain_root_then_merge() {
    let cases = [
        "merge_mixed_root_and_fractional_power_five_sixths",
        "merge_mixed_root_and_symbolic_power",
        "merge_mixed_root_and_fractional_powers_to_integer_with_passthrough",
    ];

    for case_id in cases {
        assert_case_step_has_no_substeps(case_id, "Sumar exponentes de la misma base");
    }
}

#[test]
fn derive_didactic_radical_notable_quotient_recognizes_pattern_with_internal_cleanup() {
    assert_case_step_has_no_substeps("radical_notable_quotient", "Reconocer un cociente notable");
    assert_case_has_no_no_web_substeps_flag("radical_notable_quotient");
}

#[test]
fn derive_didactic_perfect_square_root_explains_square_then_absolute_value() {
    for (case_id, expected_titles) in [
        (
            "perfect_square_root_to_abs",
            vec![
                "Reescribir el radicando como un cuadrado perfecto",
                "La raíz de un cuadrado da un valor absoluto",
            ],
        ),
        (
            "perfect_square_root_direct_power_to_abs",
            vec![
                "Identificar la base del cuadrado",
                "La raíz de un cuadrado da un valor absoluto",
            ],
        ),
        (
            "perfect_square_root_to_abs_with_passthrough",
            vec![
                "Reescribir el radicando como un cuadrado perfecto",
                "La raíz de un cuadrado da un valor absoluto",
            ],
        ),
    ] {
        let artifact = audit_case(&derive_case_by_id(case_id));

        let step = artifact
            .json_steps
            .iter()
            .find(|step| {
                step.get("rule")
                    .and_then(Value::as_str)
                    .is_some_and(|rule| rule == "Reconocer un cuadrado perfecto bajo la raíz")
            })
            .unwrap_or_else(|| panic!("expected perfect-square root derive step for {case_id}"));
        let titles: Vec<&str> = step
            .get("substeps")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("expected perfect-square root substeps for {case_id}"))
            .iter()
            .filter_map(|substep| substep.get("title").and_then(Value::as_str))
            .collect();
        assert_eq!(titles, expected_titles);
    }
}

#[test]
fn derive_didactic_nested_radical_denesting_explains_square_then_abs_cleanup() {
    assert_case_step_titles(
        "nested_radical_denesting",
        "Reconocer un cuadrado perfecto bajo la raíz",
        &[
            "Reescribir el radicando como un cuadrado perfecto",
            "La raíz de un cuadrado da un valor absoluto",
        ],
    );
    assert_case_step_has_no_substeps(
        "nested_radical_denesting",
        "Quitar valor absoluto de una expresión no negativa",
    );
    assert_case_has_no_no_web_substeps_flag("nested_radical_denesting");

    let cli_output = run_cli_lines(&derive_case_by_id("nested_radical_denesting")).join("\n");
    assert!(
        cli_output.contains("[Quitar valor absoluto de una expresión no negativa]"),
        "expected visible nonnegative-absolute-value rule suffix in CLI output:\n{cli_output}"
    );
    assert!(
        !cli_output.contains("[Abs Of Sum Of Squares]"),
        "raw internal absolute-value rule suffix leaked into CLI output:\n{cli_output}"
    );
}

#[test]
fn derive_didactic_square_of_square_root_explains_domain_condition() {
    assert_case_step_titles(
        "square_of_square_root_requires_nonnegative",
        "Deshacer raíz y potencia",
        &[
            "Identificar el radicando de la raíz principal",
            "El cuadrado deshace la raíz bajo la condición u ≥ 0",
        ],
    );
}

#[test]
fn derive_didactic_sqrt_product_merge_shows_both_source_factors() {
    let artifact = audit_case(&derive_case_by_id(
        "merge_sqrt_product_requires_nonnegative",
    ));
    let step = step_by_rule(&artifact, "Combinar raíces en un producto");

    assert_eq!(
        step.get("before").and_then(Value::as_str),
        Some("sqrt(x) · sqrt(y)")
    );
    assert_eq!(
        step.get("after").and_then(Value::as_str),
        Some("sqrt(x · y)")
    );
    assert_case_has_no_no_web_substeps_flag("merge_sqrt_product_requires_nonnegative");

    let cli_lines = run_cli_lines(&derive_case_by_id(
        "merge_sqrt_product_requires_nonnegative",
    ));
    let cli_output = cli_lines.join("\n");
    assert!(
        cli_output.contains("[Combinar raíces en un producto]"),
        "expected visible sqrt-product rule suffix in CLI output:\n{cli_output}"
    );
    assert!(
        !cli_output.contains("[Merge Sqrt Product]"),
        "raw internal sqrt-product rule suffix leaked into CLI output:\n{cli_output}"
    );
    assert!(
        cli_lines.iter().any(|line| line == "  • x ≥ 0"),
        "CLI derive report should preserve source sqrt domain for x: {cli_lines:?}"
    );
    assert!(
        cli_lines.iter().any(|line| line == "  • y ≥ 0"),
        "CLI derive report should preserve source sqrt domain for y: {cli_lines:?}"
    );
    assert!(
        !cli_lines.iter().any(|line| line == "  • x * y ≥ 0"),
        "CLI derive report should not replace source sqrt domains with only target-domain witness: {cli_lines:?}"
    );
}

#[test]
fn derive_didactic_sqrt_arithmetic_sum_combines_like_radicals() {
    assert_case_step_titles(
        "simplify_sqrt_arithmetic_sum",
        "Agrupar términos semejantes",
        &["Sumar los coeficientes que acompañan a sqrt(2)"],
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
        "rationalize_shifted_linear_root",
        "rationalize_symbolic_linear_root_plus",
    ] {
        assert_case_step_titles(case_id, "Racionalizar el denominador", expected);
    }

    let cli_output = run_cli_lines(&derive_case_by_id("rationalize_linear_root")).join("\n");
    assert!(
        cli_output.contains("[Racionalizar el denominador]"),
        "expected visible rationalize rule suffix in CLI output:\n{cli_output}"
    );
    assert!(
        !cli_output.contains("[Rationalize Linear Sqrt Denominator]"),
        "raw internal rationalize rule suffix leaked into CLI output:\n{cli_output}"
    );
}

#[test]
fn derive_didactic_cube_root_rationalization_explains_cubic_conjugate() {
    assert_case_step_titles(
        "rationalize_cube_root_sum_denominator",
        "Racionalizar el denominador",
        &[
            "Multiplicar por el conjugado cúbico",
            "Aplicar suma de cubos en el denominador",
        ],
    );
    assert_case_has_no_no_web_substeps_flag("rationalize_cube_root_sum_denominator");
}

#[test]
fn derive_didactic_representative_rationalize_zero_case_keeps_direct_cancel() {
    assert_case_step_titles(
        "rationalize_then_cancel_to_zero",
        "Racionalizar el denominador",
        &[
            "Cambiar el signo para formar el conjugado",
            "Multiplicar numerador y denominador por ese conjugado",
            "En el denominador aparece una diferencia de cuadrados",
        ],
    );
    assert_case_step_has_no_substeps(
        "rationalize_then_cancel_to_zero",
        "Restar dos expresiones iguales",
    );
    assert_case_has_no_no_web_substeps_flag("rationalize_then_cancel_to_zero");
}

#[test]
fn derive_didactic_representative_direct_nested_fraction_cases_have_expected_substeps() {
    assert_case_step_titles(
        "nested_fraction_one_over_sum",
        "Cancelar factores en una fracción",
        &[
            "Llevar a denominador común dentro del denominador",
            "Invertir la fracción del denominador",
        ],
    );
    assert_case_step_titles(
        "nested_fraction_one_over_three_reciprocals",
        "Cancelar factores en una fracción",
        &[
            "Llevar a denominador común dentro del denominador",
            "Invertir la fracción del denominador",
        ],
    );
    assert_case_step_titles(
        "nested_fraction_one_over_sum_with_passthrough",
        "Cancelar factores en una fracción",
        &[
            "Llevar a denominador común dentro del denominador",
            "Invertir la fracción del denominador",
        ],
    );
    assert_case_has_no_no_web_substeps_flag("nested_fraction_one_over_sum_with_passthrough");
    assert_case_step_titles(
        "nested_fraction_one_over_sum_with_subtractive_passthrough",
        "Cancelar factores en una fracción",
        &[
            "Llevar a denominador común dentro del denominador",
            "Invertir la fracción del denominador",
        ],
    );
    assert_case_has_no_no_web_substeps_flag(
        "nested_fraction_one_over_sum_with_subtractive_passthrough",
    );

    assert_case_step_titles(
        "nested_fraction_sum_over_reciprocal",
        "Cancelar factores en una fracción",
        &[
            "Invertir la fracción del denominador",
            "Simplificar el producto resultante",
        ],
    );
    assert_case_step_titles(
        "nested_fraction_sum_with_fraction_over_scalar_general",
        "Cancelar factores en una fracción",
        &[
            "Llevar el numerador a denominador común",
            "Incorporar el denominador externo",
        ],
    );
}

#[test]
fn derive_didactic_nested_fraction_reciprocal_inverse_shows_invert_then_cleanup() {
    assert_case_step_titles(
        "nested_fraction_reciprocal_inverse",
        "Cancelar factores en una fracción",
        &[
            "Invertir la fracción del denominador",
            "Simplificar el producto resultante",
        ],
    );
    assert_case_has_no_no_web_substeps_flag("nested_fraction_reciprocal_inverse");
}

#[test]
fn derive_didactic_structural_nested_fraction_cases_keep_single_denominator_sum_substep() {
    assert_case_step_titles(
        "nested_fraction_one_over_sum_with_fraction",
        "Cancelar factores en una fracción",
        &[
            "Llevar a denominador común dentro del denominador",
            "Invertir la fracción del denominador",
        ],
    );
    assert_case_step_titles(
        "nested_fraction_fraction_over_sum_with_fraction_general",
        "Cancelar factores en una fracción",
        &[
            "Llevar a denominador común dentro del denominador",
            "Dividir entre una fracción es multiplicar por su inversa",
        ],
    );
}

#[test]
fn derive_didactic_reverse_structural_nested_fraction_cases_keep_trace_direct() {
    for (case_id, expected_title) in [
        (
            "nested_fraction_one_over_sum_with_fraction_reverse",
            "Reescribir el denominador sacando factor común z",
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

    let (case, expected_title) = (
        DeriveCase {
            id: "nested_fraction_one_over_sum_with_fraction_compound_denominator_reverse_inline"
                .to_string(),
            family: "nested_fraction".to_string(),
            source: "(c+d)/(a*(c+d)+b)".to_string(),
            target: "1/(a + b/(c+d))".to_string(),
            expected_status: "derived".to_string(),
        },
        "Reescribir el denominador sacando factor común c + d",
    );
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
    assert_case_step_titles(
        "split_telescoping_fraction_consecutive",
        "Descomponer en fracciones telescópicas",
        &[
            "Introducir el numerador telescópico",
            "Separar sobre el denominador común",
        ],
    );

    let cases = [
        "split_telescoping_fraction_gap_two",
        "split_telescoping_fraction_negative_gap_two",
        "split_telescoping_fraction_affine_gap_two",
        "split_telescoping_fraction_affine_symbolic_shift_gap",
        "split_telescoping_fraction_difference_squares_unfactored",
        "split_telescoping_fraction_symbolic_difference_squares_unfactored",
    ];

    for case_id in cases {
        assert_case_step_titles(
            case_id,
            "Descomponer en fracciones telescópicas",
            &[
                "Introducir el numerador telescópico",
                "Separar sobre el denominador común",
            ],
        );
    }
}

#[test]
fn derive_didactic_representative_telescoping_fraction_combine_cases_keep_inverse_gap_narrative() {
    assert_case_step_titles(
        "combine_telescoping_fraction_consecutive",
        "Recomponer fracción telescópica",
        &[
            "Llevar las fracciones al denominador común",
            "Simplificar el numerador telescópico",
        ],
    );

    let no_substep_cases = [
        "combine_telescoping_fraction_gap_two",
        "combine_telescoping_fraction_negative_gap_two",
        "combine_telescoping_fraction_affine_gap_two",
        "combine_telescoping_fraction_affine_symbolic_shift_gap",
        "combine_telescoping_fraction_symbolic_difference_squares_unfactored",
    ];

    for case_id in no_substep_cases {
        assert_case_step_titles(
            case_id,
            "Recomponer fracción telescópica",
            &[
                "Llevar las fracciones al denominador común",
                "Simplificar el numerador telescópico",
            ],
        );
    }

    let cases = [(
        "combine_telescoping_fraction_shifted_quadratic_unfactored",
        vec![
            "Llevar a denominador común",
            "Simplificar el numerador y el denominador",
        ],
    )];

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
    assert_eq!(
        step_substep_titles(step),
        vec!["Usar el telescopado de cosenos con u = x"]
    );
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
    assert_eq!(
        step_substep_titles(step),
        vec!["Usar el telescopado de cosenos"]
    );
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
    assert_eq!(
        step_substep_titles(step),
        vec!["Usar el telescopado de cosenos con u = a · x"]
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
    assert_eq!(
        step_substep_titles(step),
        vec!["Expandir la ley de Morrie con u = x"]
    );
    assert_eq!(
        first_substep_latex(step),
        (
            "\\frac{\\sin(8u)}{8\\cdot \\sin(u)}",
            "\\cos(u)\\cdot \\cos(2u)\\cdot \\cos(4u)"
        )
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
    assert_eq!(
        step_substep_titles(step),
        vec!["Usar el núcleo de Dirichlet con n = 2 y u = x"]
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
    assert_eq!(
        step_substep_titles(step),
        vec!["Usar el núcleo de Dirichlet con n = 3 y u = x"]
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
    assert_eq!(
        step_substep_titles(step),
        vec!["Usar el núcleo de Dirichlet con n = 4"]
    );
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
    assert_eq!(
        step_substep_titles(step),
        vec!["Usar el núcleo de Dirichlet con n = 2 y u = a · x"]
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
    assert_eq!(
        step_substep_titles(step),
        vec!["Usar el núcleo de Dirichlet con n = 3 y u = a · x"]
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
    assert_eq!(
        step_substep_titles(step),
        vec!["Expandir el núcleo de Dirichlet con n = 2 y u = x"]
    );
    assert_eq!(
        first_substep_latex(step),
        (
            "\\frac{\\sin((2+\\frac{1}{2})u)}{\\sin(\\frac{u}{2})}",
            "1 + 2\\cdot \\sum_{k=1}^{2}\\cos(k\\cdot u)"
        )
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
    assert_eq!(
        step_substep_titles(step),
        vec!["Expandir el núcleo de Dirichlet con n = 3 y u = a · x"]
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
        "finite_telescoping_product_factorized_square_shifted_base_numeric_symbolic_lower",
        "finite_telescoping_product_factorized_square_shifted_base_symbolic_symbolic_lower",
    ];

    for case_id in factorized_cases {
        assert_case_step_titles(
            case_id,
            "Evaluar producto telescópico finito",
            &[
                "Usar (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2",
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
            "La suma telescópica cancela los términos intermedios",
        ]
    );
}

#[test]
fn derive_didactic_representative_finite_telescoping_sum_cases_keep_partial_fraction_narrative() {
    let affine_case = DeriveCase {
        id: "finite_telescoping_sum_affine_numeric_gap_symbolic_shift_inline".to_string(),
        family: "finite_telescoping".to_string(),
        source: "sum(1/((2*k+b+c)*(2*k+b+c+2)), k, m, n)".to_string(),
        target: "1/2*(1/(2*m+b+c) - 1/(2*n+b+c+2))".to_string(),
        expected_status: "derived".to_string(),
    };
    let artifact = audit_case(&affine_case);
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
            "Usar 1 / (u · (u + g)) = 1 / g · (1 / u - 1 / (u + g))",
            "La suma telescópica cancela los términos intermedios",
        ]
    );
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
        assert!(
            audit_case(&derive_case_by_id(case_id)).flags.is_empty(),
            "`{case_id}` should be accepted as a direct self-explanatory collect step"
        );
    }
}

#[test]
fn derive_didactic_representative_fraction_expansion_cases_keep_distribution_and_real_cancellation()
{
    for case_id in [
        "expand_fraction_simple",
        "expand_fraction_part_with_same_denominator_three_terms",
    ] {
        assert_case_step_has_no_substeps(case_id, "Repartir el denominador común");
        assert_case_has_no_no_web_substeps_flag(case_id);
    }
    assert_case_step_titles(
        "expand_fraction_with_common_scalar_factor_in_denominator",
        "Repartir el denominador común",
        &["Cancelar los factores comunes en la fracción que queda"],
    );

    let plural_cases = [
        "expand_fraction_mixed_variable_term_cancellation",
        "expand_fraction_three_factor_full_cancellation",
        "expand_fraction_two_cancellations_plus_remainder",
        "expand_fraction_three_factor_cross_cancellation_plus_remainder",
        "expand_fraction_three_factor_three_cancellations_to_constant",
    ];
    for case_id in plural_cases {
        assert_case_step_titles(
            case_id,
            "Repartir el denominador común",
            &["Cancelar los factores comunes en las fracciones resultantes"],
        );
    }
}

#[test]
fn derive_didactic_representative_fraction_decomposition_cases_keep_whole_plus_remainder_narrative()
{
    let csv_cases = [
        "split_fraction_into_whole_plus_remainder",
        "split_fraction_symbolic_over_scaled_general_linear",
    ];
    let expected_titles = [
        "Reescribir el numerador como parte entera por denominador más resto",
        "Separar la suma del numerador sobre el denominador",
    ];

    for case_id in csv_cases {
        assert_case_step_titles(case_id, "Separar parte entera y resto", &expected_titles);
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
            expected_titles,
            "unexpected fraction decomposition narrative for inline monic case {}",
            case.id
        );
    }
}

#[test]
fn derive_didactic_representative_fraction_combination_cases_keep_whole_plus_remainder_narrative() {
    let csv_cases = [
        "combine_whole_plus_remainder_into_fraction",
        "combine_scaled_symbolic_whole_plus_remainder_into_fraction",
    ];

    for case_id in csv_cases {
        assert_case_step_has_no_substeps(case_id, "Unir parte entera y fracción");
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
            Vec::<&str>::new(),
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
            "Reescribir potencia semientera impar" => {
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

        let substeps = step
            .get("substeps")
            .and_then(Value::as_array)
            .expect("expected odd-half-power substeps");
        assert!(
            substeps.iter().all(|substep| {
                let before = substep
                    .get("before_latex")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let after = substep
                    .get("after_latex")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                !before.contains("u")
                    && !before.contains("k")
                    && !after.contains("u")
                    && !after.contains("k")
            }),
            "odd-half-power substeps should use concrete math, got: {:?}",
            substeps
        );
    }
}

#[test]
fn derive_didactic_representative_collect_cases_keep_focus_narrative() {
    let cases = [
        ("collect_linear", "Agrupar términos por variable"),
        (
            "collect_linear_alt_variable",
            "Agrupar términos por variable",
        ),
        (
            "collect_multiple_power_groups",
            "Agrupar términos por variable",
        ),
        (
            "collect_common_symbolic_coefficients",
            "Agrupar términos por variable",
        ),
        (
            "collect_composite_monomial_factor",
            "Agrupar términos por factor común",
        ),
        (
            "collect_two_composite_factor_groups",
            "Agrupar términos por factor común",
        ),
    ];

    for (case_id, rule) in cases {
        assert_case_step_has_no_substeps(case_id, rule);
    }
}

#[test]
fn derive_didactic_representative_factor_with_division_cases_explain_hidden_divisions() {
    let cases = [
        ("factor_out_with_division", "x"),
        ("factor_out_square_with_division_quartic", "x^2"),
        ("factor_out_cube_with_division_septic", "x^3"),
    ];

    for (case_id, factor) in cases {
        assert_case_step_titles(
            case_id,
            "Sacar factor usando división",
            &[
                &format!("Reescribir cada término con el factor común {factor}"),
                &format!("Sacar el factor común {factor}"),
            ],
        );
        assert!(
            audit_case(&derive_case_by_id(case_id)).flags.is_empty(),
            "`{case_id}` should not be flagged after factor-with-division substeps"
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
    let y_step = y_artifact
        .json_steps
        .iter()
        .find(|step| {
            step.get("rule")
                .and_then(Value::as_str)
                .is_some_and(|rule| {
                    rule == "Sacar factor usando división" || rule == "Completar el cuadrado"
                })
        })
        .expect("expected direct factor-with-division or complete-square step");
    assert_eq!(
        step_substep_titles(y_step),
        [
            "Reescribir cada término con el factor común y",
            "Sacar el factor común y"
        ],
        "unexpected factor-with-division narrative for inline y case"
    );
    assert!(
        y_artifact.flags.is_empty(),
        "inline y factor-with-division case should not be flagged"
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
    assert_case_step_titles(
        "factor_common_factor_sum",
        "Factorizar",
        &["Aquí el factor común es a"],
    );
}

#[test]
fn derive_didactic_common_factor_expansion_sum_explains_distributive_law() {
    assert_case_step_titles(
        "expand_common_factor_sum",
        "Expandir la expresión",
        &[
            "Identificar los productos que genera la distributiva",
            "Escribir los productos con los signos originales",
        ],
    );
}

#[test]
fn derive_didactic_three_term_common_factor_factorization_explains_shared_factor() {
    assert_case_step_titles(
        "factor_common_factor_sum_three_terms",
        "Factorizar",
        &["Aquí el factor común es x"],
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
    assert_eq!(steps.len(), 2, "expected two distributive steps");
    for step in steps {
        assert_eq!(
            step_substep_titles(step),
            [
                "Identificar los productos que genera la distributiva",
                "Escribir los productos con los signos originales",
            ],
            "unexpected distributive substeps"
        );
    }
    assert!(
        artifact.flags.is_empty(),
        "expected three-term distributive expansion to be audit-clean; cli=\n{}\nflags={:?}",
        run_cli_lines(&derive_case_by_id(
            "expand_common_factor_difference_three_terms"
        ))
        .join("\n"),
        artifact.flags
    );
}

#[test]
fn derive_didactic_common_factor_expansion_variants_are_audit_clean() {
    for case_id in [
        "expand_common_factor_difference",
        "expand_common_factor_difference_three_terms",
        "expand_common_factor_sum",
        "expand_common_factor_sum_three_terms",
    ] {
        let artifact = audit_case(&derive_case_by_id(case_id));
        assert!(
            artifact.flags.is_empty(),
            "expected distributive expansion `{case_id}` to be audit-clean; cli=\n{}\nflags={:?}",
            run_cli_lines(&derive_case_by_id(case_id)).join("\n"),
            artifact.flags
        );
    }
}

#[test]
fn derive_didactic_numeric_common_factor_fraction_cancels_then_simplifies_remaining_fraction() {
    let case = derive_case_by_id("cancel_fraction_common_factor_numeric");
    let artifact = audit_case(&case);

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
    let cli_output = run_cli_lines(&case).join("\n");
    assert!(
        cli_output.contains("[Cancelar un factor común]"),
        "expected visible common-factor rule suffix in CLI output:\n{cli_output}"
    );
    assert!(
        !cli_output.contains("[Pre-order Common Factor Cancel]"),
        "raw internal common-factor rule suffix leaked into CLI output:\n{cli_output}"
    );
}

#[test]
fn derive_didactic_monomial_common_factor_fraction_cancels_symbol_then_simplifies_coefficients() {
    let case = derive_case_by_id("cancel_fraction_monomial_common_factor");
    assert_case_step_titles(
        &case.id,
        "Cancelar un factor común",
        &[
            "Descomponer x^2 para exponer el factor común x",
            "Cancelar el factor común x",
        ],
    );
    assert!(audit_case(&case).flags.is_empty());
    let cli_output = run_cli_lines(&case).join("\n");
    assert!(
        cli_output.contains("[Cancelar un factor común]"),
        "expected visible common-factor rule suffix in CLI output:\n{cli_output}"
    );
    assert!(
        !cli_output.contains("[Pre-order Common Factor Cancel]"),
        "raw internal common-factor rule suffix leaked into CLI output:\n{cli_output}"
    );
}

#[test]
fn derive_didactic_complete_square_monic_cases_show_hidden_square_balance() {
    for case_id in [
        "solve_prep_complete_square_monic_numeric",
        "solve_prep_complete_square_fractional_monic_numeric",
        "solve_prep_complete_square_symbolic_monic_parametric",
    ] {
        assert_case_step_titles(
            case_id,
            "Completar el cuadrado",
            &[
                "Añadir y restar el cuadrado del semicoeficiente",
                "Agrupar el trinomio como cuadrado perfecto",
            ],
        );
    }

    let cli_output = run_cli_lines(&derive_case_by_id(
        "solve_prep_complete_square_monic_numeric",
    ))
    .join("\n");
    assert!(
        cli_output.contains("[Completar el cuadrado]"),
        "expected visible complete-square rule suffix in CLI output:\n{cli_output}"
    );
    assert!(
        !cli_output.contains("[Complete the Square]"),
        "raw internal complete-square rule suffix leaked into CLI output:\n{cli_output}"
    );
}

#[test]
fn derive_didactic_complete_square_non_monic_cases_factor_then_balance_inside_parentheses() {
    for case_id in [
        "solve_prep_complete_square_alt_variable_symbolic_leading_coeff",
        "solve_prep_complete_square_fractional_symbolic_leading_coeff",
        "solve_prep_complete_square_negative_symbolic_leading_coeff",
        "solve_prep_complete_square_symbolic_leading_coeff",
        "solve_prep_complete_square_symbolic_negative_linear_coeff",
    ] {
        let artifact = audit_case(&derive_case_by_id(case_id));
        let step = step_by_rule(&artifact, "Completar el cuadrado");
        assert_eq!(
            step_substep_titles(step),
            [
                "Extraer el coeficiente líder de los términos cuadráticos",
                "Añadir y restar el cuadrado del semicoeficiente dentro del paréntesis",
                "Agrupar el trinomio como cuadrado perfecto",
            ],
            "unexpected complete-square narrative for `{case_id}`"
        );
        assert!(
            artifact.flags.is_empty(),
            "expected `{case_id}` to be clean after non-monic complete-square substeps; cli=\n{}\nflags={:?}",
            run_cli_lines(&derive_case_by_id(case_id)).join("\n"),
            artifact.flags
        );
    }
}

#[test]
fn derive_didactic_complete_square_negative_linear_coeff_shows_hidden_square_balance() {
    let case = DeriveCase {
        id: "solve_prep_complete_square_monic_negative_linear_inline".to_string(),
        family: "solve_prep".to_string(),
        source: "x^2 - 2*x + c".to_string(),
        target: "(x-1)^2 + c - 1".to_string(),
        expected_status: "derived".to_string(),
    };
    let artifact = audit_case(&case);
    let step = step_by_rule(&artifact, "Completar el cuadrado");
    assert_eq!(
        step_substep_titles(step),
        [
            "Añadir y restar el cuadrado del semicoeficiente",
            "Agrupar el trinomio como cuadrado perfecto"
        ],
        "unexpected complete-square narrative for `{}`",
        case.id
    );
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
    assert_case_step_titles(
        "expand_cube_difference_product",
        "Expandir la expresión",
        &[
            "Reconocer el patrón (a - b)(a^2 + ab + b^2)",
            "Aplicar (a - b)(a^2 + ab + b^2) = a^3 - b^3",
        ],
    );
}

#[test]
fn derive_didactic_quadratic_difference_of_squares_product_keeps_cancellation_story() {
    assert_case_step_titles(
        "expand_difference_of_squares_quadratic_product",
        "Expandir la expresión",
        &[
            "Aplicar el producto de conjugados",
            "Simplificar las potencias",
        ],
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
    assert_case_step_titles(
        "expand_eighth_power_minus_multifactor_product",
        "Expandir la expresión",
        &[
            "Aplicar el producto de conjugados",
            "Simplificar las potencias",
        ],
    );
}

#[test]
fn derive_didactic_ninth_power_plus_product_uses_composite_sum_of_cubes_identity_language() {
    assert_case_step_titles(
        "expand_ninth_power_plus_product",
        "Expandir la expresión",
        &[
            "Reconocer el patrón (a + b)(a^2 - ab + b^2)",
            "Aplicar (a + b)(a^2 - ab + b^2) = a^3 + b^3",
        ],
    );
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
        .iter()
        .filter(|case| case.expected_status == "derived")
        .cloned()
        .collect();
    let worker_count = derive_audit_worker_count(cases.len());
    let total_start = Instant::now();
    let artifacts_start = Instant::now();
    let timed_artifacts = map_derive_cases_parallel(&cases, timed_audit_case);
    let artifacts_elapsed = artifacts_start.elapsed();
    let artifact_seconds_by_case = timed_artifacts
        .iter()
        .map(|timed| timed.seconds)
        .collect::<Vec<_>>();
    let artifacts = timed_artifacts
        .into_iter()
        .map(|timed| timed.artifact)
        .collect::<Vec<_>>();
    let flagged_cases = artifacts
        .iter()
        .filter(|artifact| !artifact.flags.is_empty())
        .count();
    let no_web_steps = artifacts
        .iter()
        .filter(|artifact| {
            artifact
                .flags
                .iter()
                .any(|flag| flag == "no web steps emitted")
        })
        .count();
    let no_web_substeps = artifacts
        .iter()
        .filter(|artifact| {
            artifact
                .flags
                .iter()
                .any(|flag| flag == "no web substeps emitted")
        })
        .count();
    let total_web_substeps = artifacts
        .iter()
        .map(|artifact| artifact.web_substep_count)
        .sum::<usize>();
    let total_steps = artifacts
        .iter()
        .map(|artifact| artifact.step_count)
        .sum::<usize>();
    let mean_step_count = if artifacts.is_empty() {
        0.0
    } else {
        total_steps as f64 / artifacts.len() as f64
    };
    let cli_start = Instant::now();
    let timed_cli_lines = map_derive_cases_parallel(&cases, timed_run_cli_lines);
    let cli_elapsed = cli_start.elapsed();
    let cli_seconds_by_case = timed_cli_lines
        .iter()
        .map(|timed| timed.seconds)
        .collect::<Vec<_>>();
    let cli_lines_by_case = timed_cli_lines
        .into_iter()
        .map(|timed| timed.lines)
        .collect::<Vec<_>>();
    let report_start = Instant::now();
    let report = build_report(&cases, &artifacts, &cli_lines_by_case);
    let path = report_output_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create report dir");
    }
    fs::write(&path, report).expect("write derive didactic audit report");
    let report_elapsed = report_start.elapsed();
    let total_elapsed = total_start.elapsed();
    eprintln!(
        "derive didactic audit summary: cases={} flagged={} no_web_substeps={} no_web_steps={} total_web_substeps={} mean_step_count={:.2}",
        artifacts.len(),
        flagged_cases,
        no_web_substeps,
        no_web_steps,
        total_web_substeps,
        mean_step_count
    );
    eprintln!(
        "derive didactic audit timings: artifacts_seconds={:.3} cli_seconds={:.3} report_seconds={:.3} total_seconds={:.3} worker_count={}",
        artifacts_elapsed.as_secs_f64(),
        cli_elapsed.as_secs_f64(),
        report_elapsed.as_secs_f64(),
        total_elapsed.as_secs_f64(),
        worker_count
    );
    eprintln!(
        "derive didactic audit family hotspots: artifacts={} cli={}",
        phase_family_hotspots(&cases, &artifact_seconds_by_case),
        phase_family_hotspots(&cases, &cli_seconds_by_case)
    );
    eprintln!(
        "derive didactic audit case hotspots: artifacts={} cli={}",
        phase_case_hotspots(&cases, &artifact_seconds_by_case),
        phase_case_hotspots(&cases, &cli_seconds_by_case)
    );
    eprintln!("wrote {}", path.display());
}
