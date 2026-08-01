//! `metamorphic_simplification_tests`: familia `substitution_runs`.
//!
//! Ver la cabecera de `metamorphic_simplification_tests.rs` para el contexto.

use super::*;

fn classify_substitution_combo_locally(
    lhs_text: &str,
    rhs_text: &str,
    free_var: &str,
    filters: &[FilterSpec],
    config: &MetatestConfig,
    proof_flavor: MetamorphicProofFlavor,
    shortcut_mode: MetatestShortcutMode,
) -> SubstitutionComboOutcome {
    let pre_nf_engine_preproof = shortcut_mode.allows_pre_nf_proof_shortcuts()
        || matches!(shortcut_mode, MetatestShortcutMode::NfFirstPressure);
    if pre_nf_engine_preproof && prove_zero_from_engine_texts_in_child_process(lhs_text, rhs_text) {
        return SubstitutionComboOutcome {
            kind: "proved".to_string(),
            residual: String::new(),
            cause: String::new(),
            cycles: 0,
        };
    }

    let mut simplifier = Simplifier::with_default_rules();
    let exp_parsed = match parse(lhs_text, &mut simplifier.context) {
        Ok(expr) => expr,
        Err(_) => {
            return SubstitutionComboOutcome {
                kind: "parse_error".to_string(),
                residual: String::new(),
                cause: String::new(),
                cycles: 0,
            };
        }
    };
    let simp_parsed = match parse(rhs_text, &mut simplifier.context) {
        Ok(expr) => expr,
        Err(_) => {
            return SubstitutionComboOutcome {
                kind: "parse_error".to_string(),
                residual: String::new(),
                cause: String::new(),
                cycles: 0,
            };
        }
    };

    let opts = cas_solver::runtime::SimplifyOptions::default();
    let mut cycles = 0usize;
    let (mut e, _, stats_e) = simplifier.simplify_with_stats(exp_parsed, opts.clone());
    cycles += stats_e.cycle_events.len();
    let (mut s, _, stats_s) = simplifier.simplify_with_stats(simp_parsed, opts.clone());
    cycles += stats_s.cycle_events.len();

    let cfg = cas_solver::runtime::EvalConfig::default();
    let mut budget = cas_solver::runtime::Budget::preset_cli();
    if let Ok(result) = cas_solver::api::fold_constants(
        &mut simplifier.context,
        e,
        &cfg,
        cas_solver::api::ConstFoldMode::Safe,
        &mut budget,
    ) {
        e = result.expr;
    }
    if let Ok(result) = cas_solver::api::fold_constants(
        &mut simplifier.context,
        s,
        &cfg,
        cas_solver::api::ConstFoldMode::Safe,
        &mut budget,
    ) {
        s = result.expr;
    }

    if cas_solver::runtime::compare_expr(&simplifier.context, e, s) == std::cmp::Ordering::Equal {
        return SubstitutionComboOutcome {
            kind: "nf".to_string(),
            residual: String::new(),
            cause: String::new(),
            cycles,
        };
    }

    if matches!(shortcut_mode, MetatestShortcutMode::NfFirstPressure)
        && prove_zero_from_engine_texts_in_child_process(lhs_text, rhs_text)
    {
        return SubstitutionComboOutcome {
            kind: "proved".to_string(),
            residual: String::new(),
            cause: String::new(),
            cycles,
        };
    }

    if prove_zero_from_metamorphic_texts_with_flavor(
        &mut simplifier,
        lhs_text,
        rhs_text,
        e,
        s,
        proof_flavor,
    ) {
        return SubstitutionComboOutcome {
            kind: "proved".to_string(),
            residual: String::new(),
            cause: String::new(),
            cycles,
        };
    }

    if matches!(proof_flavor, MetamorphicProofFlavor::RawPressure) {
        if let Some(reason) = known_raw_domain_frontier_reason(lhs_text, rhs_text) {
            return SubstitutionComboOutcome {
                kind: "domain_frontier".to_string(),
                residual: reason.to_string(),
                cause: String::new(),
                cycles,
            };
        }
    }

    let free_var_owned = free_var.to_string();
    match classify_numeric_equiv_for_vars(
        &simplifier.context,
        e,
        s,
        std::slice::from_ref(&free_var_owned),
        filters,
        config,
    ) {
        NumericCheckOutcome::Pass => {
            let d = simplifier.context.add(cas_ast::Expr::Sub(e, s));
            let (d_simp, _) = simplifier.simplify(d);
            let residual = cas_formatter::LaTeXExpr {
                context: &simplifier.context,
                id: d_simp,
            }
            .to_latex();
            let shape = expr_shape_signature(&simplifier.context, d_simp);
            let cause = numeric_only_cause_for_vars(
                &simplifier.context,
                e,
                s,
                std::slice::from_ref(&free_var_owned),
                filters,
                config,
                &shape,
            )
            .label()
            .to_string();
            if let Some(reason) =
                known_domain_frontier_reason_for_numeric_cause(&cause, lhs_text, rhs_text)
            {
                return SubstitutionComboOutcome {
                    kind: "domain_frontier".to_string(),
                    residual: reason.to_string(),
                    cause: String::new(),
                    cycles,
                };
            }
            SubstitutionComboOutcome {
                kind: "numeric".to_string(),
                residual,
                cause,
                cycles,
            }
        }
        NumericCheckOutcome::Inconclusive(reason) => SubstitutionComboOutcome {
            kind: "inconclusive".to_string(),
            residual: reason,
            cause: String::new(),
            cycles,
        },
        NumericCheckOutcome::Failed(_) => SubstitutionComboOutcome {
            kind: "failed".to_string(),
            residual: String::new(),
            cause: String::new(),
            cycles,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn classify_substitution_combo_in_thread(
    lhs_text: &str,
    rhs_text: &str,
    free_var: &str,
    filters: &[FilterSpec],
    config: &MetatestConfig,
    proof_flavor: MetamorphicProofFlavor,
    shortcut_mode: MetatestShortcutMode,
    timeout: std::time::Duration,
) -> Option<SubstitutionComboOutcome> {
    let lhs = lhs_text.to_string();
    let rhs = rhs_text.to_string();
    let free_var = free_var.to_string();
    let filters = filters.to_vec();
    let config = config.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let _handle = std::thread::Builder::new()
        .stack_size(METATEST_WORKER_STACK_SIZE_BYTES)
        .spawn(move || {
            let outcome = classify_substitution_combo_locally(
                &lhs,
                &rhs,
                &free_var,
                &filters,
                &config,
                proof_flavor,
                shortcut_mode,
            );
            let _ = tx.send(outcome);
        })
        .ok()?;

    rx.recv_timeout(timeout).ok()
}

fn classify_substitution_combo_in_child_process(
    lhs_text: &str,
    rhs_text: &str,
    free_var: &str,
    filters: &[FilterSpec],
    proof_flavor: MetamorphicProofFlavor,
    shortcut_mode: MetatestShortcutMode,
    timeout: std::time::Duration,
) -> Option<SubstitutionComboOutcome> {
    let Ok(current_exe) = std::env::current_exe() else {
        return None;
    };

    let outcome_path = std::env::temp_dir().join(format!(
        "metatest_substitution_combo_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let mut child = match std::process::Command::new(current_exe)
        .arg("metatest_child_substitution_combo_classify")
        .arg("--ignored")
        .arg("--exact")
        .arg("--nocapture")
        .env(METATEST_CHILD_SUBSTITUTION_LHS_ENV, lhs_text)
        .env(METATEST_CHILD_SUBSTITUTION_RHS_ENV, rhs_text)
        .env(METATEST_CHILD_SUBSTITUTION_VAR_ENV, free_var)
        .env(
            METATEST_CHILD_SUBSTITUTION_FILTERS_ENV,
            encode_child_filters(filters),
        )
        .env(
            METATEST_CHILD_SUBSTITUTION_PROOF_ENV,
            proof_flavor.child_label(),
        )
        .env(
            METATEST_CHILD_SUBSTITUTION_MODE_ENV,
            shortcut_mode.child_label(),
        )
        .env(
            METATEST_CHILD_SUBSTITUTION_OUTCOME_ENV,
            outcome_path.to_string_lossy().to_string(),
        )
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return None,
    };

    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    let _ = std::fs::remove_file(&outcome_path);
                    if prove_zero_from_engine_texts_in_child_process(lhs_text, rhs_text) {
                        return Some(SubstitutionComboOutcome {
                            kind: "proved".to_string(),
                            residual: String::new(),
                            cause: String::new(),
                            cycles: 0,
                        });
                    }
                    return None;
                }
                let payload = std::fs::read_to_string(&outcome_path)
                    .ok()
                    .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())?;
                let _ = std::fs::remove_file(&outcome_path);
                return Some(SubstitutionComboOutcome {
                    kind: payload
                        .get("kind")
                        .and_then(Value::as_str)
                        .unwrap_or("inconclusive")
                        .to_string(),
                    residual: payload
                        .get("residual")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    cause: payload
                        .get("cause")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    cycles: payload.get("cycles").and_then(Value::as_u64).unwrap_or(0) as usize,
                });
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = std::fs::remove_file(&outcome_path);
                    if prove_zero_from_engine_texts_in_child_process(lhs_text, rhs_text) {
                        return Some(SubstitutionComboOutcome {
                            kind: "proved".to_string(),
                            residual: String::new(),
                            cause: String::new(),
                            cycles: 0,
                        });
                    }
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(&outcome_path);
                if prove_zero_from_engine_texts_in_child_process(lhs_text, rhs_text) {
                    return Some(SubstitutionComboOutcome {
                        kind: "proved".to_string(),
                        residual: String::new(),
                        cause: String::new(),
                        cycles: 0,
                    });
                }
                return None;
            }
        }
    }
}

#[test]
#[ignore]
fn metatest_child_substitution_combo_classify() {
    let lhs = std::env::var(METATEST_CHILD_SUBSTITUTION_LHS_ENV)
        .expect("missing child substitution lhs env");
    let rhs = std::env::var(METATEST_CHILD_SUBSTITUTION_RHS_ENV)
        .expect("missing child substitution rhs env");
    let free_var = std::env::var(METATEST_CHILD_SUBSTITUTION_VAR_ENV)
        .expect("missing child substitution var env");
    let filters = decode_child_filters(
        &std::env::var(METATEST_CHILD_SUBSTITUTION_FILTERS_ENV)
            .expect("missing child substitution filters env"),
    );
    let proof_flavor = MetamorphicProofFlavor::from_child_label(
        &std::env::var(METATEST_CHILD_SUBSTITUTION_PROOF_ENV)
            .expect("missing child substitution proof env"),
    )
    .expect("invalid child substitution proof flavor");
    let shortcut_mode = MetatestShortcutMode::from_child_label(
        &std::env::var(METATEST_CHILD_SUBSTITUTION_MODE_ENV)
            .expect("missing child substitution mode env"),
    )
    .expect("invalid child substitution shortcut mode");
    let outcome_path = std::env::var(METATEST_CHILD_SUBSTITUTION_OUTCOME_ENV)
        .expect("missing child substitution outcome env");

    let handle = std::thread::Builder::new()
        .stack_size(METATEST_DEEP_WORKER_STACK_SIZE_BYTES)
        .spawn(move || {
            classify_substitution_combo_locally(
                &lhs,
                &rhs,
                &free_var,
                &filters,
                &metatest_config(),
                proof_flavor,
                shortcut_mode,
            )
        })
        .expect("spawn substitution child worker");
    let outcome = handle.join().expect("substitution child worker panicked");
    let payload = serde_json::json!({
        "kind": outcome.kind,
        "residual": outcome.residual,
        "cause": outcome.cause,
        "cycles": outcome.cycles,
    });
    std::fs::write(
        outcome_path,
        serde_json::to_string(&payload).expect("serialize substitution payload"),
    )
    .expect("write substitution child outcome");
}

#[test]
fn atan_double_angle_identity_matches_substituted_cosine_pair() {
    let lhs = "cos(2*arctan((cos(u))))";
    let rhs = "(1-(cos(u))^2)/(1+(cos(u))^2)";
    assert!(atan_double_angle_identity_matches(lhs, rhs));
    assert!(atan_double_angle_identity_matches(rhs, lhs));
}

#[test]
fn half_angle_identity_matches_substituted_linear_pair() {
    let lhs = "2*sin(((2*u+3))/2)^2";
    let rhs = "1-cos((2*u+3))";
    assert!(half_angle_identity_matches(lhs, rhs));
    assert!(half_angle_identity_matches(rhs, lhs));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug builds are not performance-representative for raw-pressure contextual proofs; component coverage remains in contextual smoke and raw-pressure direct regressions"
)]
fn raw_pressure_proof_can_use_original_engine_texts_for_contextual_pair() {
    let lhs = "((x^2 + y^2)*(a^2 + b^2)) + (sec((1/(u - 1) + 1/(u + 1)))^2 - tan((1/(u - 1) + 1/(u + 1)))^2)";
    let rhs = "((x*a + y*b)^2 + (x*b - y*a)^2) + 1";

    assert!(prove_zero_from_contextual_block_strategies_text(lhs, rhs));

    let mut simplifier = Simplifier::with_default_rules();
    let lhs_expr = parse(lhs, &mut simplifier.context).expect("lhs parses");
    let rhs_expr = parse(rhs, &mut simplifier.context).expect("rhs parses");
    let (lhs_simp_raw, _) = simplifier.simplify(lhs_expr);
    let lhs_simp = fold_constants_safe(&mut simplifier.context, lhs_simp_raw);
    let (rhs_simp_raw, _) = simplifier.simplify(rhs_expr);
    let rhs_simp = fold_constants_safe(&mut simplifier.context, rhs_simp_raw);

    assert!(prove_zero_from_metamorphic_texts_with_flavor(
        &mut simplifier,
        lhs,
        rhs,
        lhs_simp,
        rhs_simp,
        MetamorphicProofFlavor::RawPressure
    ));
}

/// Word-boundary-aware text substitution.
/// Replaces all occurrences of `var` as a standalone word in `template`
/// with `replacement`, wrapping in parentheses for safety.
/// Uses simple word-boundary logic: a match is valid if the chars
/// before and after are not alphanumeric or underscore.
pub(super) fn text_substitute(template: &str, var: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(template.len() * 2);
    let chars: Vec<char> = template.chars().collect();
    let var_chars: Vec<char> = var.chars().collect();
    let var_len = var_chars.len();
    let mut i = 0;

    while i < chars.len() {
        // Check if var matches at position i
        if i + var_len <= chars.len() && chars[i..i + var_len] == var_chars[..] {
            // Check word boundary before
            let before_ok = if i == 0 {
                true
            } else {
                let c = chars[i - 1];
                !c.is_alphanumeric() && c != '_'
            };
            // Check word boundary after
            let after_ok = if i + var_len >= chars.len() {
                true
            } else {
                let c = chars[i + var_len];
                !c.is_alphanumeric() && c != '_'
            };

            if before_ok && after_ok {
                result.push('(');
                result.push_str(replacement);
                result.push(')');
                i += var_len;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

/// Load substitution identity pairs from CSV
fn parse_substitution_identities() -> Vec<IdentityPair> {
    let csv_path = find_test_data_file("substitution_identities.csv");
    let content =
        std::fs::read_to_string(csv_path).expect("Failed to read substitution_identities.csv");

    let mut pairs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Each row")
                && !label.starts_with("Substitution-Based")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            let vars: Vec<String> = parts[2]
                .trim()
                .split(';')
                .map(|s| s.trim().to_string())
                .collect();
            let mode = if parts.len() >= 4 {
                parse_domain_mode(parts[3].trim())
            } else {
                DomainRequirement::Generic
            };
            pairs.push(IdentityPair {
                exp: parts[0].trim().to_string(),
                simp: parts[1].trim().to_string(),
                vars,
                mode,
                bucket: Bucket::ConditionalRequires,
                branch_mode: BranchMode::default(),
                filter_spec: FilterSpec::None,
                family: current_family.clone(),
            });
        }
    }
    pairs
}

pub(super) fn load_substitution_identities() -> Vec<IdentityPair> {
    static SUBSTITUTION_IDENTITIES: OnceLock<Vec<IdentityPair>> = OnceLock::new();
    SUBSTITUTION_IDENTITIES
        .get_or_init(parse_substitution_identities)
        .clone()
}

/// Load substitution expressions from CSV
fn parse_substitution_expressions_from(filename: &str) -> Vec<SubstitutionExpr> {
    let csv_path = find_test_data_file(filename);
    let content = std::fs::read_to_string(&csv_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", filename));

    let mut exprs = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.splitn(4, ',').collect();
        if parts.len() >= 3 {
            let var = parts[1].trim().to_string();
            let filters = if parts.len() >= 4 {
                parse_filter_specs(parts[3], 1, line_num)
            } else {
                vec![FilterSpec::None]
            };
            exprs.push(SubstitutionExpr {
                expr: parts[0].trim().to_string(),
                var,
                label: parts[2].trim().to_string(),
                filters,
            });
        }
    }
    exprs
}

fn load_substitution_expressions() -> Vec<SubstitutionExpr> {
    static SUBSTITUTION_EXPRESSIONS: OnceLock<Vec<SubstitutionExpr>> = OnceLock::new();
    SUBSTITUTION_EXPRESSIONS
        .get_or_init(|| parse_substitution_expressions_from("substitution_expressions.csv"))
        .clone()
}

fn load_structural_substitution_expressions() -> Vec<SubstitutionExpr> {
    static STRUCTURAL_SUBSTITUTION_EXPRESSIONS: OnceLock<Vec<SubstitutionExpr>> = OnceLock::new();
    STRUCTURAL_SUBSTITUTION_EXPRESSIONS
        .get_or_init(|| {
            parse_substitution_expressions_from("substitution_structural_expressions.csv")
        })
        .clone()
}

fn filter_substitutions_by_labels(
    substitutions: Vec<SubstitutionExpr>,
    labels: &[&str],
) -> Vec<SubstitutionExpr> {
    let allowed: std::collections::HashSet<&str> = labels.iter().copied().collect();
    substitutions
        .into_iter()
        .filter(|sub| allowed.contains(sub.label.as_str()))
        .collect()
}

pub(super) fn load_contextual_pairs() -> Vec<ContextualPair> {
    static CONTEXTUAL_PAIRS: OnceLock<Vec<ContextualPair>> = OnceLock::new();
    CONTEXTUAL_PAIRS
        .get_or_init(|| parse_direct_pairs("contextual_pairs.csv"))
        .clone()
}

pub(super) fn load_contextual_rational_pairs() -> Vec<ContextualPair> {
    static CONTEXTUAL_RATIONAL_PAIRS: OnceLock<Vec<ContextualPair>> = OnceLock::new();
    CONTEXTUAL_RATIONAL_PAIRS
        .get_or_init(|| parse_direct_pairs("contextual_rational_pairs.csv"))
        .clone()
}

pub(super) fn load_contextual_trig_pairs() -> Vec<ContextualPair> {
    static CONTEXTUAL_TRIG_PAIRS: OnceLock<Vec<ContextualPair>> = OnceLock::new();
    CONTEXTUAL_TRIG_PAIRS
        .get_or_init(|| parse_direct_pairs("contextual_trig_pairs.csv"))
        .clone()
}

pub(super) fn load_contextual_polynomial_pairs() -> Vec<ContextualPair> {
    static CONTEXTUAL_POLYNOMIAL_PAIRS: OnceLock<Vec<ContextualPair>> = OnceLock::new();
    CONTEXTUAL_POLYNOMIAL_PAIRS
        .get_or_init(|| parse_direct_pairs("contextual_polynomial_pairs.csv"))
        .clone()
}

pub(super) fn load_contextual_radical_pairs() -> Vec<ContextualPair> {
    static CONTEXTUAL_RADICAL_PAIRS: OnceLock<Vec<ContextualPair>> = OnceLock::new();
    CONTEXTUAL_RADICAL_PAIRS
        .get_or_init(|| parse_direct_pairs("contextual_radical_pairs.csv"))
        .clone()
}

fn substitution_filters_for_mode(
    sub: &SubstitutionExpr,
    use_declared_filters: bool,
) -> Vec<FilterSpec> {
    if use_declared_filters {
        return sub.filters.clone();
    }

    vec![FilterSpec::None; sub.filters.len().max(1)]
}

/// Run substitution-based metamorphic tests
fn run_substitution_tests_with(
    substitutions: Vec<SubstitutionExpr>,
    suite_label: &str,
    suite_op: &str,
) -> ComboMetrics {
    run_substitution_tests_with_mode(
        substitutions,
        suite_label,
        suite_op,
        MetamorphicProofFlavor::Curated,
        true,
        MetatestShortcutMode::SmokeClosure,
    )
}

fn run_substitution_tests_with_strict_mode(
    substitutions: Vec<SubstitutionExpr>,
    suite_label: &str,
    suite_op: &str,
) -> ComboMetrics {
    run_substitution_tests_with_mode(
        substitutions,
        suite_label,
        suite_op,
        MetamorphicProofFlavor::Curated,
        true,
        MetatestShortcutMode::StrictPressure,
    )
}

fn run_substitution_tests_with_nf_first_mode(
    substitutions: Vec<SubstitutionExpr>,
    suite_label: &str,
    suite_op: &str,
) -> ComboMetrics {
    run_substitution_tests_with_mode(
        substitutions,
        suite_label,
        suite_op,
        MetamorphicProofFlavor::Curated,
        true,
        MetatestShortcutMode::NfFirstPressure,
    )
}

fn run_substitution_tests_with_mode(
    substitutions: Vec<SubstitutionExpr>,
    suite_label: &str,
    suite_op: &str,
    proof_flavor: MetamorphicProofFlavor,
    use_declared_filters: bool,
    shortcut_mode: MetatestShortcutMode,
) -> ComboMetrics {
    let identities = load_substitution_identities();
    let config = metatest_config();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let show_table = std::env::var("METATEST_TABLE").is_ok();
    let trace_sub = std::env::var("METATEST_TRACE_SUB").is_ok();
    let progress_every = std::env::var("METATEST_PROGRESS_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_METATEST_PROGRESS_EVERY);
    let requested_combo_cap = std::env::var("METATEST_MAX_COMBOS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .or_else(|| {
            if !shortcut_mode.allows_curated_shortcuts() && cfg!(debug_assertions) {
                Some(20)
            } else {
                None
            }
        });
    let requested_combo_start = std::env::var("METATEST_COMBO_START")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    // Filter out Assume-only identities (we run in Generic mode)
    let identities: Vec<_> = identities
        .into_iter()
        .filter(|p| p.mode != DomainRequirement::Assume)
        .collect();
    let total_combos = identities.len() * substitutions.len();
    let (combo_start_offset, effective_total_combos) =
        effective_combo_window(total_combos, requested_combo_start, requested_combo_cap);
    let mut touched_identity_indices = vec![false; identities.len()];
    for combo_idx in combo_start_offset..combo_start_offset.saturating_add(effective_total_combos) {
        let identity_idx = combo_idx / substitutions.len();
        if identity_idx < touched_identity_indices.len() {
            touched_identity_indices[identity_idx] = true;
        }
    }
    let identity_symbolic_ok: Vec<bool> = if (shortcut_mode.allows_composed_promotion()
        || matches!(shortcut_mode, MetatestShortcutMode::NfFirstPressure))
        && matches!(proof_flavor, MetamorphicProofFlavor::Curated)
    {
        identities
            .iter()
            .enumerate()
            .map(|(idx, pair)| {
                if !touched_identity_indices[idx] {
                    return false;
                }
                if shortcut_mode.allows_curated_shortcuts() {
                    pair_is_symbolically_proved(pair)
                } else {
                    pair_is_raw_pressure_proved(pair)
                }
            })
            .collect()
    } else {
        vec![false; identities.len()]
    };
    eprintln!(
        "📊 Running {} metamorphic tests: {} identities × {} substitutions = {} combos (seed {})",
        suite_label,
        identities.len(),
        substitutions.len(),
        effective_total_combos,
        config.seed
    );
    if combo_start_offset > 0 || effective_total_combos < total_combos {
        eprintln!(
            "🔬 Applying combo window [{}]: start {} size {} / {} planned substitution combinations",
            suite_op,
            combo_start_offset,
            effective_total_combos,
            total_combos
        );
    }

    // Global counters
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut nf_convergent = 0usize;
    let mut proved_symbolic = 0usize;
    let mut numeric_only = 0usize;
    let mut inconclusive = 0usize;
    let mut inconclusive_causes: HashMap<String, usize> = HashMap::new();
    let mut timeouts = 0usize;
    let mut cycle_events_total: usize = 0;
    let mut parse_errors = 0usize;
    let mut numeric_only_causes: HashMap<String, usize> = HashMap::new();
    let mut numeric_only_by_label: HashMap<String, usize> = HashMap::new();
    let mut numeric_only_by_expr: HashMap<String, usize> = HashMap::new();
    let mut numeric_only_examples: Vec<(String, String, String, String, String)> = Vec::new();
    let mut domain_frontier = 0usize;
    let mut domain_frontier_examples: Vec<(String, String, String)> = Vec::new();
    let mut inconclusive_examples: Vec<(String, String, String)> = Vec::new();
    let mut symbolic_tracker_count = 0usize;
    let mut symbolic_tracker_examples: Vec<(String, String, String)> = Vec::new();
    let mut timeout_examples: Vec<(String, String, String, String)> = Vec::new();

    // Cross-product table data: (family, sub_label) → (nf, proved, numeric, failed)
    let mut cell_data: HashMap<(String, String), (usize, usize, usize, usize)> = HashMap::new();

    let combo_timeout = std::time::Duration::from_secs(5);
    let mut processed_combos = 0usize;
    let mut visited_combos = 0usize;

    for (identity_idx, identity) in identities.iter().enumerate() {
        let id_var = &identity.vars[0]; // Variable to substitute (typically "x")

        for sub in &substitutions {
            if processed_combos >= effective_total_combos {
                break;
            }
            if visited_combos < combo_start_offset {
                visited_combos += 1;
                continue;
            }
            // Build LHS and RHS by substituting x → sub.expr
            let lhs_str = text_substitute(&identity.exp, id_var, &sub.expr);
            let rhs_str = text_substitute(&identity.simp, id_var, &sub.expr);
            let free_var = sub.var.clone();
            let filters = substitution_filters_for_mode(sub, use_declared_filters);
            let cell_key = (identity.family.clone(), sub.label.clone());
            if trace_sub {
                eprintln!(
                    "🔎 Sub [{}] #{} / {} :: [{}] {}  with [{}] {}",
                    suite_op,
                    processed_combos + 1,
                    effective_total_combos,
                    identity.family,
                    identity.exp,
                    sub.label,
                    sub.expr
                );
            }

            if matches!(proof_flavor, MetamorphicProofFlavor::Curated)
                && identity_symbolic_ok[identity_idx]
            {
                proved_symbolic += 1;
                passed += 1;
                cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).1 += 1;
                processed_combos += 1;
                visited_combos += 1;
                if should_report_combo_progress(
                    verbose,
                    effective_total_combos,
                    processed_combos,
                    progress_every,
                ) {
                    eprintln!(
                        "⏳ Progress [{}]: {}/{} ({:.1}%) | NF {} | Proved {} | Numeric {} | Inconcl {} | T/O {} | Failed {}",
                        suite_op,
                        processed_combos,
                        effective_total_combos,
                        100.0 * (processed_combos as f64) / (effective_total_combos as f64),
                        nf_convergent,
                        proved_symbolic,
                        numeric_only,
                        inconclusive,
                        timeouts,
                        failed,
                    );
                }
                continue;
            }

            if shortcut_mode.allows_curated_shortcuts()
                && matches!(proof_flavor, MetamorphicProofFlavor::Curated)
                && prove_zero_from_residual_pair_corpus_text(&lhs_str, &rhs_str)
            {
                proved_symbolic += 1;
                passed += 1;
                cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).1 += 1;
                processed_combos += 1;
                visited_combos += 1;
                if should_report_combo_progress(
                    verbose,
                    effective_total_combos,
                    processed_combos,
                    progress_every,
                ) {
                    eprintln!(
                        "⏳ Progress [{}]: {}/{} ({:.1}%) | NF {} | Proved {} | Numeric {} | Inconcl {} | T/O {} | Failed {}",
                        suite_op,
                        processed_combos,
                        effective_total_combos,
                        100.0 * (processed_combos as f64) / (effective_total_combos as f64),
                        nf_convergent,
                        proved_symbolic,
                        numeric_only,
                        inconclusive,
                        timeouts,
                        failed,
                    );
                }
                continue;
            }

            if shortcut_mode.allows_curated_shortcuts()
                && matches!(proof_flavor, MetamorphicProofFlavor::Curated)
                && atan_double_angle_identity_matches(&lhs_str, &rhs_str)
            {
                proved_symbolic += 1;
                passed += 1;
                cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).1 += 1;
                processed_combos += 1;
                visited_combos += 1;
                if should_report_combo_progress(
                    verbose,
                    effective_total_combos,
                    processed_combos,
                    progress_every,
                ) {
                    eprintln!(
                        "⏳ Progress [{}]: {}/{} ({:.1}%) | NF {} | Proved {} | Numeric {} | Inconcl {} | T/O {} | Failed {}",
                        suite_op,
                        processed_combos,
                        effective_total_combos,
                        100.0 * (processed_combos as f64) / (effective_total_combos as f64),
                        nf_convergent,
                        proved_symbolic,
                        numeric_only,
                        inconclusive,
                        timeouts,
                        failed,
                    );
                }
                continue;
            }

            if shortcut_mode.allows_curated_shortcuts()
                && matches!(proof_flavor, MetamorphicProofFlavor::Curated)
                && half_angle_identity_matches(&lhs_str, &rhs_str)
            {
                proved_symbolic += 1;
                passed += 1;
                cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).1 += 1;
                processed_combos += 1;
                visited_combos += 1;
                if should_report_combo_progress(
                    verbose,
                    effective_total_combos,
                    processed_combos,
                    progress_every,
                ) {
                    eprintln!(
                        "⏳ Progress [{}]: {}/{} ({:.1}%) | NF {} | Proved {} | Numeric {} | Inconcl {} | T/O {} | Failed {}",
                        suite_op,
                        processed_combos,
                        effective_total_combos,
                        100.0 * (processed_combos as f64) / (effective_total_combos as f64),
                        nf_convergent,
                        proved_symbolic,
                        numeric_only,
                        inconclusive,
                        timeouts,
                        failed,
                    );
                }
                continue;
            }

            if matches!(shortcut_mode, MetatestShortcutMode::NfFirstPressure)
                && nf_converges_in_child_process(&lhs_str, &rhs_str)
            {
                nf_convergent += 1;
                passed += 1;
                cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).0 += 1;
                processed_combos += 1;
                visited_combos += 1;
                if should_report_combo_progress(
                    verbose,
                    effective_total_combos,
                    processed_combos,
                    progress_every,
                ) {
                    eprintln!(
                        "⏳ Progress [{}]: {}/{} ({:.1}%) | NF {} | Proved {} | Numeric {} | Inconcl {} | T/O {} | Failed {}",
                        suite_op,
                        processed_combos,
                        effective_total_combos,
                        100.0 * (processed_combos as f64) / (effective_total_combos as f64),
                        nf_convergent,
                        proved_symbolic,
                        numeric_only,
                        inconclusive,
                        timeouts,
                        failed,
                    );
                }
                continue;
            }

            let outcome = if shortcut_mode.requires_deep_combo_worker() {
                classify_substitution_combo_in_child_process(
                    &lhs_str,
                    &rhs_str,
                    &free_var,
                    &filters,
                    proof_flavor,
                    shortcut_mode,
                    combo_timeout,
                )
            } else {
                classify_substitution_combo_in_thread(
                    &lhs_str,
                    &rhs_str,
                    &free_var,
                    &filters,
                    &config,
                    proof_flavor,
                    shortcut_mode,
                    combo_timeout,
                )
            };

            match outcome {
                Some(SubstitutionComboOutcome {
                    kind,
                    residual,
                    cause,
                    cycles,
                }) => match kind.as_str() {
                    "nf" => {
                        nf_convergent += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).0 += 1;
                    }
                    "proved" => {
                        proved_symbolic += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).1 += 1;
                    }
                    "numeric" => {
                        numeric_only += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        *numeric_only_causes.entry(cause.clone()).or_default() += 1;
                        *numeric_only_by_label.entry(sub.label.clone()).or_default() += 1;
                        *numeric_only_by_expr.entry(sub.expr.clone()).or_default() += 1;
                        if let Some(reason) = known_symbolic_residual_reason(&lhs_str, &rhs_str) {
                            symbolic_tracker_count += 1;
                            if verbose && symbolic_tracker_examples.len() < 32 {
                                symbolic_tracker_examples.push((
                                    lhs_str.clone(),
                                    rhs_str.clone(),
                                    reason.to_string(),
                                ));
                            }
                        }
                        cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).2 += 1;
                        if verbose && numeric_only_examples.len() < 200 {
                            numeric_only_examples.push((
                                lhs_str.clone(),
                                rhs_str.clone(),
                                identity.family.clone(),
                                residual,
                                cause,
                            ));
                        }
                    }
                    "inconclusive" => {
                        inconclusive += 1;
                        cycle_events_total += cycles;
                        record_inconclusive_reason(
                            &mut inconclusive_causes,
                            "inconclusive",
                            &residual,
                        );
                        if verbose && inconclusive_examples.len() < 32 {
                            inconclusive_examples.push((
                                lhs_str.clone(),
                                rhs_str.clone(),
                                residual,
                            ));
                        }
                    }
                    "domain_frontier" => {
                        inconclusive += 1;
                        domain_frontier += 1;
                        cycle_events_total += cycles;
                        record_inconclusive_reason(
                            &mut inconclusive_causes,
                            "domain_frontier",
                            &residual,
                        );
                        if verbose && domain_frontier_examples.len() < 32 {
                            domain_frontier_examples.push((
                                lhs_str.clone(),
                                rhs_str.clone(),
                                residual,
                            ));
                        }
                    }
                    "parse_error" => {
                        parse_errors += 1;
                        passed += 1; // Don't count as failure
                    }
                    "failed" => {
                        failed += 1;
                        cycle_events_total += cycles;
                        cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).3 += 1;
                        if verbose {
                            eprintln!(
                                "  ❌ FAIL [{} → {}]: {} vs {}",
                                identity.family, sub.label, lhs_str, rhs_str
                            );
                        }
                    }
                    _ => {
                        failed += 1;
                        cycle_events_total += cycles;
                    }
                },
                None => {
                    timeouts += 1;
                    if verbose && timeout_examples.len() < 32 {
                        timeout_examples.push((
                            lhs_str.clone(),
                            rhs_str.clone(),
                            identity.family.clone(),
                            sub.label.clone(),
                        ));
                    }
                }
            }
            processed_combos += 1;
            visited_combos += 1;
            if should_report_combo_progress(
                verbose,
                effective_total_combos,
                processed_combos,
                progress_every,
            ) {
                eprintln!(
                    "⏳ Progress [{}]: {}/{} ({:.1}%) | NF {} | Proved {} | Numeric {} | Inconcl {} | T/O {} | Failed {}",
                    suite_op,
                    processed_combos,
                    effective_total_combos,
                    100.0 * (processed_combos as f64) / (effective_total_combos as f64),
                    nf_convergent,
                    proved_symbolic,
                    numeric_only,
                    inconclusive,
                    timeouts,
                    failed,
                );
            }
        }
        if processed_combos >= effective_total_combos {
            break;
        }
    }

    // Report: flat summary (always shown)
    eprintln!(
        "✅ {} tests: {} passed, {} failed, {} timed out, {} parse errors, {} inconclusive",
        suite_label, passed, failed, timeouts, parse_errors, inconclusive
    );
    eprintln!(
        "   📐 NF-convergent: {} | 🔢 Proved-symbolic: {} | 🌡️ Numeric-only: {} | ◐ Inconclusive: {}",
        nf_convergent, proved_symbolic, numeric_only, inconclusive
    );
    if domain_frontier > 0 {
        eprintln!(
            "   🛡️ Known domain-frontier: {} (counted inside inconclusive)",
            domain_frontier
        );
    }
    if verbose && inconclusive > 0 {
        print_inconclusive_breakdown(&inconclusive_causes);
    }
    if symbolic_tracker_count > 0 {
        eprintln!(
            "   📌 Known symbolic-residual tracker: {} (still counted inside numeric-only)",
            symbolic_tracker_count
        );
    }
    if verbose && numeric_only > 0 {
        print_numeric_only_cause_breakdown(&numeric_only_causes);
        if !numeric_only_by_label.is_empty() {
            eprintln!("   🧪 Numeric-only by substitution label:");
            let mut sorted: Vec<_> = numeric_only_by_label.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            for (label, count) in sorted {
                eprintln!("      - {}: {}", label, count);
            }
        }
        if !numeric_only_by_expr.is_empty() {
            eprintln!("   🧬 Numeric-only by substitution expr:");
            let mut sorted: Vec<_> = numeric_only_by_expr.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            for (expr, count) in sorted.into_iter().take(12) {
                eprintln!("      - {}: {}", expr, count);
            }
        }
    }

    // Cross-product table (METATEST_TABLE=1)
    if show_table {
        // Collect unique families and sub-labels in order
        let mut families: Vec<String> = identities
            .iter()
            .map(|i| i.family.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        families.sort();

        let mut sub_labels: Vec<String> = substitutions
            .iter()
            .map(|s| s.label.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        sub_labels.sort();

        // Abbreviate long sub-labels for columns
        let col_width = 11;
        let family_width = 25;

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  Substitution × Identity Cross-Product (NF/Proved/Numeric)                         ║");
        eprintln!(
            "╠═════════════════════════╤{}╣",
            sub_labels
                .iter()
                .map(|_l| format!("═{:═<width$}═", "", width = col_width))
                .collect::<Vec<_>>()
                .join("╤")
        );

        // Header row
        eprint!(
            "║ {:family_width$}│",
            "Identity Family",
            family_width = family_width
        );
        for label in &sub_labels {
            let short = if label.len() > col_width {
                format!("{}…", &label[..col_width - 1])
            } else {
                label.clone()
            };
            eprint!(" {:^width$}│", short, width = col_width);
        }
        eprintln!();

        // Separator
        eprint!(
            "╠═{:═<family_width$}═╪",
            "",
            family_width = family_width - 2
        );
        for (i, _) in sub_labels.iter().enumerate() {
            if i < sub_labels.len() - 1 {
                eprint!("═{:═<width$}═╪", "", width = col_width);
            } else {
                eprint!("═{:═<width$}═╣", "", width = col_width);
            }
        }
        eprintln!();

        // Data rows
        for family in &families {
            eprint!("║ {:family_width$}│", family, family_width = family_width);
            for label in &sub_labels {
                let key = (family.clone(), label.clone());
                let (nf, prov, num, fail) = cell_data.get(&key).copied().unwrap_or((0, 0, 0, 0));
                let cell = if fail > 0 {
                    format!("{}/{}/{}❌{}", nf, prov, num, fail)
                } else {
                    format!("{}/{}/{}", nf, prov, num)
                };
                eprint!(" {:^width$}│", cell, width = col_width);
            }
            eprintln!();
        }

        // Bottom border
        eprint!(
            "╚═{:═<family_width$}═╧",
            "",
            family_width = family_width - 2
        );
        for (i, _) in sub_labels.iter().enumerate() {
            if i < sub_labels.len() - 1 {
                eprint!("═{:═<width$}═╧", "", width = col_width);
            } else {
                eprint!("═{:═<width$}═╝", "", width = col_width);
            }
        }
        eprintln!();
        eprintln!("Legend: NF/Proved/Numeric (❌N = N failures)");
    }

    // Verbose: show numeric-only cases grouped by family
    if verbose && !numeric_only_examples.is_empty() {
        eprintln!("\n── numeric-only examples ──");
        // Group by family
        let mut family_groups: HashMap<String, Vec<(String, String, String, String)>> =
            HashMap::new();
        for (lhs, rhs, family, residual, cause) in &numeric_only_examples {
            family_groups.entry(family.clone()).or_default().push((
                lhs.clone(),
                rhs.clone(),
                residual.clone(),
                cause.clone(),
            ));
        }
        let mut families: Vec<_> = family_groups.keys().cloned().collect();
        families.sort();
        for family in &families {
            let examples = &family_groups[family];
            eprintln!("── {} ({} cases) ──", family, examples.len());
            for (lhs, rhs, residual, cause) in examples.iter().take(10) {
                eprintln!("  LHS: {}", lhs);
                eprintln!("  RHS: {}", rhs);
                eprintln!("  Cause: {}", cause);
                if !residual.is_empty() {
                    eprintln!("  Residual: {}", residual);
                }
                eprintln!();
            }
        }
    }

    if verbose && !domain_frontier_examples.is_empty() {
        eprintln!("\n── domain-frontier examples ──");
        for (lhs, rhs, reason) in domain_frontier_examples.iter().take(10) {
            eprintln!("  LHS: {}", lhs);
            eprintln!("  RHS: {}", rhs);
            eprintln!("  Reason: {}", reason);
            eprintln!();
        }
    }

    if verbose && !inconclusive_examples.is_empty() {
        eprintln!("\n── inconclusive examples ──");
        for (lhs, rhs, reason) in inconclusive_examples.iter().take(10) {
            eprintln!("  LHS: {}", lhs);
            eprintln!("  RHS: {}", rhs);
            eprintln!("  Reason: {}", reason);
            eprintln!();
        }
    }

    if verbose && !symbolic_tracker_examples.is_empty() {
        eprintln!("\n── symbolic-residual tracker examples ──");
        for (lhs, rhs, reason) in symbolic_tracker_examples.iter().take(10) {
            eprintln!("  LHS: {}", lhs);
            eprintln!("  RHS: {}", rhs);
            eprintln!("  Reason: {}", reason);
            eprintln!();
        }
    }

    if verbose && !timeout_examples.is_empty() {
        eprintln!("\n── timeout examples ──");
        for (lhs, rhs, family, label) in timeout_examples.iter().take(10) {
            eprintln!("  Family: {}", family);
            eprintln!("  Substitution: {}", label);
            eprintln!("  LHS: {}", lhs);
            eprintln!("  RHS: {}", rhs);
            eprintln!();
        }
    }

    // Count unique identity families used
    let num_families = identities
        .iter()
        .map(|i| &i.family)
        .collect::<std::collections::HashSet<_>>()
        .len();

    ComboMetrics {
        op: suite_op.to_string(),
        pairs: identities.len(),
        families: num_families,
        combos: effective_total_combos,
        nf_convergent,
        proved_quotient: proved_symbolic,
        proved_difference: 0,
        proved_composed: 0,
        numeric_only,
        inconclusive,
        failed,
        skipped: parse_errors,
        timeouts,
        cycle_events_total,
        known_symbolic_residuals: symbolic_tracker_count,
        numeric_only_causes,
        inconclusive_causes,
        domain_frontier_examples,
    }
}

pub(super) fn run_substitution_tests() -> ComboMetrics {
    run_substitution_tests_with(load_substitution_expressions(), "Substitution", "⇄sub")
}

pub(super) fn run_substitution_tests_strict() -> ComboMetrics {
    run_substitution_tests_with_strict_mode(load_substitution_expressions(), "Substitution", "⇄sub")
}

pub(super) fn run_substitution_tests_nf_first() -> ComboMetrics {
    run_substitution_tests_with_nf_first_mode(
        load_substitution_expressions(),
        "Substitution",
        "⇄sub",
    )
}

pub(super) fn run_structural_substitution_tests() -> ComboMetrics {
    run_substitution_tests_with(
        load_structural_substitution_expressions(),
        "Structural substitution",
        "⇄sub+",
    )
}

pub(super) fn run_structural_substitution_tests_strict() -> ComboMetrics {
    run_substitution_tests_with_strict_mode(
        load_structural_substitution_expressions(),
        "Structural substitution",
        "⇄sub+",
    )
}

pub(super) fn run_structural_substitution_tests_nf_first() -> ComboMetrics {
    run_substitution_tests_with_nf_first_mode(
        load_structural_substitution_expressions(),
        "Structural substitution",
        "⇄sub+",
    )
}

pub(super) fn run_structural_substitution_tests_raw() -> ComboMetrics {
    run_substitution_tests_with_mode(
        load_structural_substitution_expressions(),
        "Structural substitution (raw pressure)",
        "⇄sub+raw",
        MetamorphicProofFlavor::RawPressure,
        false,
        MetatestShortcutMode::StrictPressure,
    )
}

pub(super) fn run_structural_phase_substitution_tests() -> ComboMetrics {
    run_substitution_tests_with(
        filter_substitutions_by_labels(load_structural_substitution_expressions(), &["phase"]),
        "Structural substitution (phase)",
        "⇄sub+.phase",
    )
}

pub(super) fn run_structural_radical_substitution_tests() -> ComboMetrics {
    run_substitution_tests_with(
        filter_substitutions_by_labels(
            load_structural_substitution_expressions(),
            &["composed", "root_ctx"],
        ),
        "Structural substitution (radical)",
        "⇄sub+.rad",
    )
}

pub(super) fn run_structural_poly_high_substitution_tests() -> ComboMetrics {
    run_substitution_tests_with(
        filter_substitutions_by_labels(load_structural_substitution_expressions(), &["poly_high"]),
        "Structural substitution (poly-high)",
        "⇄sub+.poly",
    )
}

pub(super) fn run_structural_rational_ctx_substitution_tests() -> ComboMetrics {
    run_substitution_tests_with(
        filter_substitutions_by_labels(
            load_structural_substitution_expressions(),
            &["rational_ctx"],
        ),
        "Structural substitution (rational-ctx)",
        "⇄sub+.ratctx",
    )
}

pub(super) fn run_structural_composed_substitution_tests() -> ComboMetrics {
    run_substitution_tests_with(
        filter_substitutions_by_labels(load_structural_substitution_expressions(), &["composed"]),
        "Structural substitution (composed)",
        "⇄sub+.cmp",
    )
}

pub(super) fn run_structural_root_ctx_substitution_tests() -> ComboMetrics {
    run_substitution_tests_with(
        filter_substitutions_by_labels(load_structural_substitution_expressions(), &["root_ctx"]),
        "Structural substitution (root-ctx)",
        "⇄sub+.root",
    )
}

pub(super) fn run_structural_absolute_substitution_tests() -> ComboMetrics {
    run_substitution_tests_with(
        filter_substitutions_by_labels(load_structural_substitution_expressions(), &["absolute"]),
        "Structural substitution (absolute)",
        "⇄sub+.abs",
    )
}

pub(super) fn run_structural_rational_substitution_tests() -> ComboMetrics {
    run_substitution_tests_with(
        filter_substitutions_by_labels(load_structural_substitution_expressions(), &["rational"]),
        "Structural substitution (rational)",
        "⇄sub+.rat",
    )
}

pub(super) fn run_structural_inv_trig_substitution_tests() -> ComboMetrics {
    run_substitution_tests_with(
        filter_substitutions_by_labels(load_structural_substitution_expressions(), &["inv_trig"]),
        "Structural substitution (inv-trig)",
        "⇄sub+.inv",
    )
}

/// Run curated contextual metamorphic tests.
pub(super) fn run_contextual_pair_tests() -> ComboMetrics {
    let pairs = load_contextual_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual metamorphic tests",
        "Contextual tests",
        MetatestShortcutMode::SmokeClosure,
    )
}

pub(super) fn run_contextual_rational_pair_tests() -> ComboMetrics {
    let pairs = load_contextual_rational_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual rational metamorphic tests",
        "Contextual rational tests",
        MetatestShortcutMode::SmokeClosure,
    )
}

pub(super) fn run_contextual_trig_pair_tests() -> ComboMetrics {
    let pairs = load_contextual_trig_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual trig metamorphic tests",
        "Contextual trig tests",
        MetatestShortcutMode::SmokeClosure,
    )
}

pub(super) fn run_contextual_polynomial_pair_tests() -> ComboMetrics {
    let pairs = load_contextual_polynomial_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual polynomial metamorphic tests",
        "Contextual polynomial tests",
        MetatestShortcutMode::SmokeClosure,
    )
}

pub(super) fn run_contextual_radical_pair_tests() -> ComboMetrics {
    let pairs = load_contextual_radical_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual radical metamorphic tests",
        "Contextual radical tests",
        MetatestShortcutMode::SmokeClosure,
    )
}

pub(super) fn run_contextual_pair_tests_strict() -> ComboMetrics {
    let pairs = load_contextual_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual metamorphic tests",
        "Contextual tests",
        MetatestShortcutMode::StrictPressure,
    )
}

pub(super) fn run_contextual_pair_tests_nf_first() -> ComboMetrics {
    let pairs = load_contextual_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual metamorphic tests",
        "Contextual tests",
        MetatestShortcutMode::NfFirstPressure,
    )
}

pub(super) fn run_contextual_rational_pair_tests_strict() -> ComboMetrics {
    let pairs = load_contextual_rational_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual rational metamorphic tests",
        "Contextual rational tests",
        MetatestShortcutMode::StrictPressure,
    )
}

pub(super) fn run_contextual_rational_pair_tests_nf_first() -> ComboMetrics {
    let pairs = load_contextual_rational_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual rational metamorphic tests",
        "Contextual rational tests",
        MetatestShortcutMode::NfFirstPressure,
    )
}

pub(super) fn run_contextual_trig_pair_tests_strict() -> ComboMetrics {
    let pairs = load_contextual_trig_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual trig metamorphic tests",
        "Contextual trig tests",
        MetatestShortcutMode::StrictPressure,
    )
}

pub(super) fn run_contextual_trig_pair_tests_nf_first() -> ComboMetrics {
    let pairs = load_contextual_trig_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual trig metamorphic tests",
        "Contextual trig tests",
        MetatestShortcutMode::NfFirstPressure,
    )
}

pub(super) fn run_contextual_polynomial_pair_tests_strict() -> ComboMetrics {
    let pairs = load_contextual_polynomial_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual polynomial metamorphic tests",
        "Contextual polynomial tests",
        MetatestShortcutMode::StrictPressure,
    )
}

pub(super) fn run_contextual_polynomial_pair_tests_nf_first() -> ComboMetrics {
    let pairs = load_contextual_polynomial_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual polynomial metamorphic tests",
        "Contextual polynomial tests",
        MetatestShortcutMode::NfFirstPressure,
    )
}

pub(super) fn run_contextual_radical_pair_tests_strict() -> ComboMetrics {
    let pairs = load_contextual_radical_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual radical metamorphic tests",
        "Contextual radical tests",
        MetatestShortcutMode::StrictPressure,
    )
}

pub(super) fn run_contextual_radical_pair_tests_nf_first() -> ComboMetrics {
    let pairs = load_contextual_radical_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual radical metamorphic tests",
        "Contextual radical tests",
        MetatestShortcutMode::NfFirstPressure,
    )
}

#[test]
fn load_structural_substitution_expressions_parses_optional_filters() {
    let substitutions = load_structural_substitution_expressions();
    let root_ctx = substitutions
        .into_iter()
        .find(|sub| sub.label == "root_ctx")
        .expect("root_ctx substitution");

    assert_eq!(root_ctx.filters.len(), 1);
    assert_eq!(root_ctx.filters[0].as_str(), "gt(0.1)");
}

#[test]
fn substitution_filters_for_raw_mode_strip_declared_filters() {
    let substitutions = load_structural_substitution_expressions();
    let root_ctx = substitutions
        .iter()
        .find(|sub| sub.label == "root_ctx")
        .expect("root_ctx substitution");

    let raw_filters = substitution_filters_for_mode(root_ctx, false);
    assert_eq!(raw_filters.len(), 1);
    assert!(raw_filters[0].is_none());
}

#[test]
fn known_domain_frontier_detects_substitution_log_square_pair() {
    assert_eq!(
        known_domain_frontier_reason("ln((2*u)^2)", "2*ln((2*u))"),
        Some("log-square expansion changes domain")
    );
}

#[test]
fn known_symbolic_residual_clears_trig_square_cube_substitution_pair() {
    assert_eq!(
        known_symbolic_residual_reason(
            "((sin(u)^2)^3 - 1)/((sin(u)^2) - 1)",
            "(sin(u)^2)^2 + (sin(u)^2) + 1"
        ),
        None
    );
}
