//! `metamorphic_simplification_tests`: familia `domain_conditions`.
//!
//! Ver la cabecera de `metamorphic_simplification_tests.rs` para el contexto.

use super::*;

pub(super) fn known_raw_domain_frontier_reason(
    lhs_text: &str,
    rhs_text: &str,
) -> Option<&'static str> {
    known_domain_frontier_reason(lhs_text, rhs_text)
}

/// Parse domain mode from string
pub(super) fn parse_domain_mode(s: &str) -> DomainRequirement {
    match s.to_lowercase().as_str() {
        "a" | "assume" => DomainRequirement::Assume,
        _ => DomainRequirement::Generic,
    }
}

pub(super) fn safe_window_mirror_closes_all_domain_frontiers(
    total_domain_frontier: usize,
    safe_window_metrics: &ComboMetrics,
) -> bool {
    safe_window_metrics.proved_symbolic() == total_domain_frontier
        && safe_window_metrics.failed == 0
        && safe_window_metrics.inconclusive == 0
        && safe_window_metrics.numeric_only == 0
        && safe_window_metrics.timeouts == 0
}

#[test]
fn safe_window_mirror_closure_requires_exact_symbolic_cover_and_clean_metrics() {
    let good = ComboMetrics {
        op: "⇄ctx".to_string(),
        pairs: 8,
        families: 3,
        combos: 8,
        nf_convergent: 0,
        proved_quotient: 8,
        proved_difference: 0,
        proved_composed: 0,
        numeric_only: 0,
        inconclusive: 0,
        failed: 0,
        skipped: 0,
        timeouts: 0,
        cycle_events_total: 0,
        known_symbolic_residuals: 0,
        numeric_only_causes: HashMap::new(),
        inconclusive_causes: HashMap::new(),
        domain_frontier_examples: Vec::new(),
    };
    assert!(safe_window_mirror_closes_all_domain_frontiers(8, &good));

    let mut numeric_leak = good.clone();
    numeric_leak.numeric_only = 1;
    assert!(!safe_window_mirror_closes_all_domain_frontiers(
        8,
        &numeric_leak
    ));

    let mut missing_cover = good.clone();
    missing_cover.proved_quotient = 7;
    assert!(!safe_window_mirror_closes_all_domain_frontiers(
        8,
        &missing_cover
    ));
}

#[test]
fn combo_progress_reporting_requires_verbose_large_suite_and_interval_boundary() {
    assert!(should_report_combo_progress(true, 5000, 1000, 1000));
    assert!(!should_report_combo_progress(false, 5000, 1000, 1000));
    assert!(!should_report_combo_progress(true, 900, 900, 1000));
    assert!(!should_report_combo_progress(true, 5000, 999, 1000));
}

pub(super) fn load_known_domain_frontier_pairs() -> Vec<ContextualPair> {
    static KNOWN_DOMAIN_FRONTIER_PAIRS: OnceLock<Vec<ContextualPair>> = OnceLock::new();
    KNOWN_DOMAIN_FRONTIER_PAIRS
        .get_or_init(|| parse_direct_pairs("known_domain_frontier_pairs.csv"))
        .clone()
}

pub(super) fn load_known_domain_frontier_safe_pairs() -> Vec<ContextualPair> {
    static KNOWN_DOMAIN_FRONTIER_SAFE_PAIRS: OnceLock<Vec<ContextualPair>> = OnceLock::new();
    KNOWN_DOMAIN_FRONTIER_SAFE_PAIRS
        .get_or_init(|| parse_direct_pairs("known_domain_frontier_safe_pairs.csv"))
        .clone()
}

fn load_requires_contract_expressions() -> Vec<RequiresContractExpr> {
    let csv_path = find_test_data_file("requires_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read requires_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(2, ',').collect();
        if parts.len() != 2 {
            panic!(
                "requires_contract_expressions.csv line {}: expected expr,expect_requires. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[1].trim().to_string();
        let expect_requires = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "requires_contract_expressions.csv line {}: invalid expect_requires '{}'",
                line_num, other
            ),
        };

        exprs.push(RequiresContractExpr {
            expr,
            expect_requires,
            family: current_family.clone(),
        });
    }

    exprs
}

pub(super) fn parse_domain_mode_label(
    label: &str,
    csv_name: &str,
    line_num: usize,
) -> cas_solver::runtime::DomainMode {
    match label.trim().to_lowercase().as_str() {
        "generic" => cas_solver::runtime::DomainMode::Generic,
        "strict" => cas_solver::runtime::DomainMode::Strict,
        "assume" => cas_solver::runtime::DomainMode::Assume,
        other => panic!(
            "{} line {}: invalid domain mode '{}'",
            csv_name, line_num, other
        ),
    }
}

pub(super) fn domain_mode_label(mode: cas_solver::runtime::DomainMode) -> &'static str {
    match mode {
        cas_solver::runtime::DomainMode::Generic => "generic",
        cas_solver::runtime::DomainMode::Strict => "strict",
        cas_solver::runtime::DomainMode::Assume => "assume",
    }
}

pub(super) fn parse_value_domain_label(
    label: &str,
    csv_name: &str,
    line_num: usize,
) -> cas_solver::runtime::ValueDomain {
    match label.trim().to_lowercase().as_str() {
        "real" | "realonly" => cas_solver::runtime::ValueDomain::RealOnly,
        "complex" | "complexenabled" => cas_solver::runtime::ValueDomain::ComplexEnabled,
        other => panic!(
            "{} line {}: invalid value domain '{}'",
            csv_name, line_num, other
        ),
    }
}

pub(super) fn value_domain_label(value_domain: cas_solver::runtime::ValueDomain) -> &'static str {
    match value_domain {
        cas_solver::runtime::ValueDomain::RealOnly => "real",
        cas_solver::runtime::ValueDomain::ComplexEnabled => "complex",
    }
}

fn load_requires_mode_contract_expressions() -> Vec<RequiresModeContractExpr> {
    let csv_path = find_test_data_file("requires_mode_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read requires_mode_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(3, ',').collect();
        if parts.len() != 3 {
            panic!(
                "requires_mode_contract_expressions.csv line {}: expected expr,mode,expect_requires. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[2].trim().to_string();
        let mode =
            parse_domain_mode_label(parts[1], "requires_mode_contract_expressions.csv", line_num);
        let expect_requires = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "requires_mode_contract_expressions.csv line {}: invalid expect_requires '{}'",
                line_num, other
            ),
        };

        exprs.push(RequiresModeContractExpr {
            expr,
            mode,
            expect_requires,
            family: current_family.clone(),
        });
    }

    exprs
}

fn load_assumption_trace_contract_expressions() -> Vec<AssumptionTraceContractExpr> {
    let csv_path = find_test_data_file("assumption_trace_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read assumption_trace_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(4, ',').collect();
        if parts.len() != 4 {
            panic!(
                "assumption_trace_contract_expressions.csv line {}: expected expr,mode,inv_trig,expected_kind. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[3].trim().to_string();
        let mode = parse_domain_mode_label(
            parts[2],
            "assumption_trace_contract_expressions.csv",
            line_num,
        );
        let inv_trig = parse_inv_trig_policy_label(
            parts[1],
            "assumption_trace_contract_expressions.csv",
            line_num,
        );
        let expected_kind = match parts[0].trim().to_lowercase().as_str() {
            "none" | "" => None,
            other => Some(other.to_string()),
        };

        exprs.push(AssumptionTraceContractExpr {
            expr,
            mode,
            inv_trig,
            expected_kind,
            family: current_family.clone(),
        });
    }

    exprs
}

fn simplify_with_assumption_trace(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
    inv_trig: cas_solver::runtime::InverseTrigPolicy,
) -> Result<SimplifyTraceMetadata, String> {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().shared.semantics.domain_mode = mode;
    state.options_mut().shared.semantics.inv_trig = inv_trig;

    let parsed = parse(input, &mut engine.simplifier.context)
        .map_err(|e| format!("parse failed for '{}': {:?}", input, e))?;
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine
        .eval(&mut state, req)
        .map_err(|e| format!("eval failed for '{}': {:?}", input, e))?;

    let result = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        other => {
            return Err(format!(
                "unexpected eval result for '{}': {:?}",
                input, other
            ));
        }
    };

    let mut assumption_kinds: Vec<String> = output
        .steps
        .iter()
        .flat_map(|step| step.assumption_events().iter())
        .map(|event| event.key.kind().to_string())
        .collect();
    assumption_kinds.sort();
    assumption_kinds.dedup();

    Ok(SimplifyTraceMetadata {
        result,
        assumption_kinds,
    })
}

pub(super) fn simplify_with_metadata_in_domain(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
) -> Result<SimplifyMetadata, String> {
    simplify_with_metadata_on_axes(input, mode, cas_solver::runtime::ValueDomain::RealOnly)
}

pub(super) fn run_requires_contract_tests() -> RequiresContractMetrics {
    let cases = load_requires_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = RequiresContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running required_conditions contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_generic_with_metadata(&case.expr) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!("[{}] {} — {}", case.family, case.expr, err));
                continue;
            }
        };

        let second = match simplify_generic_with_metadata(&first.result) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}] {} -> '{}' reparsed failed: {}",
                    case.family, case.expr, first.result, err
                ));
                continue;
            }
        };

        if case.expect_requires && !first.required.is_empty() {
            metrics.expected_requires_present += 1;
        }

        let mut case_failed = false;
        if case.expect_requires && first.required.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}] {} — expected requires, got none",
                case.family, case.expr
            ));
            case_failed = true;
        }

        let first_required: std::collections::HashSet<_> = first.required.iter().cloned().collect();
        let second_required: std::collections::HashSet<_> =
            second.required.iter().cloned().collect();
        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();

        let introduced_requires: Vec<_> = second_required
            .difference(&first_required)
            .cloned()
            .collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        if !introduced_requires.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}] {} — introduced requires: {:?} (first={:?}, second={:?})",
                case.family, case.expr, introduced_requires, first.required, second.required
            ));
            case_failed = true;
        }

        if !introduced_warnings.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family, case.expr, introduced_warnings, first.warnings, second.warnings
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.required == second.required && first.warnings == second.warnings {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}] {} — requires {:?} -> {:?}, warnings {:?} -> {:?}",
                        case.family,
                        case.expr,
                        first.required,
                        second.required,
                        first.warnings,
                        second.warnings
                    ));
                }
            }
        } else if verbose {
            eprintln!("  ❌ [{}] {}", case.family, case.expr);
        }
    }

    eprintln!(
        "✅ Requires contracts: exact={} relaxed={} expected_requires_present={}/{} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_requires_present,
        cases.iter().filter(|c| c.expect_requires).count(),
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ requires contract relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 requires contract failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

pub(super) fn run_requires_mode_contract_tests() -> RequiresModeContractMetrics {
    let cases = load_requires_mode_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = RequiresModeContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running mode-aware required_conditions contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_metadata_in_domain(&case.expr, case.mode) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}] {} — {}",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second = match simplify_with_metadata_in_domain(&first.result, case.mode) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}] {} -> '{}' reparsed failed: {}",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr,
                    first.result,
                    err
                ));
                continue;
            }
        };

        let mut case_failed = false;
        if case.expect_requires {
            if first.required.is_empty() {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}] {} — expected requires, got none",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_requires_present += 1;
            }
        } else if first.required.is_empty() {
            metrics.expected_requires_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — unexpected requires: {:?}",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                first.required
            ));
            case_failed = true;
        }

        let first_required: std::collections::HashSet<_> = first.required.iter().cloned().collect();
        let second_required: std::collections::HashSet<_> =
            second.required.iter().cloned().collect();
        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();

        let introduced_requires: Vec<_> = second_required
            .difference(&first_required)
            .cloned()
            .collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        if !introduced_requires.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — introduced requires: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                introduced_requires,
                first.required,
                second.required
            ));
            case_failed = true;
        }

        if !introduced_warnings.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                introduced_warnings,
                first.warnings,
                second.warnings
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.required == second.required && first.warnings == second.warnings {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}] {} — requires {:?} -> {:?}, warnings {:?} -> {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        case.expr,
                        first.required,
                        second.required,
                        first.warnings,
                        second.warnings
                    ));
                }
            }
        } else if verbose {
            eprintln!(
                "  ❌ [{}|{}] {}",
                case.family,
                domain_mode_label(case.mode),
                case.expr
            );
        }
    }

    eprintln!(
        "✅ Mode-aware requires contracts: exact={} relaxed={} expected_requires_present={} expected_requires_absent={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_requires_present,
        metrics.expected_requires_absent,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ mode-aware requires relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 mode-aware requires failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

pub(super) fn run_assumption_trace_contract_tests() -> AssumptionTraceContractMetrics {
    let cases = load_assumption_trace_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = AssumptionTraceContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running assumption trace contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_assumption_trace(&case.expr, case.mode, case.inv_trig) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — {}",
                    case.family,
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second = match simplify_with_assumption_trace(&first.result, case.mode, case.inv_trig) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} -> '{}' reparsed failed: {}",
                    case.family,
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr,
                    first.result,
                    err
                ));
                continue;
            }
        };

        let mut case_failed = false;
        match &case.expected_kind {
            Some(kind) => {
                if !first.assumption_kinds.iter().any(|k| k == kind) {
                    metrics.failed += 1;
                    failures.push(format!(
                        "[{}|{}|{}] {} — expected assumption kind '{}', got {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        inv_trig_policy_label(case.inv_trig),
                        case.expr,
                        kind,
                        first.assumption_kinds
                    ));
                    case_failed = true;
                } else {
                    metrics.expected_present += 1;
                }
            }
            None => {
                if first.assumption_kinds.is_empty() {
                    metrics.expected_absent += 1;
                } else {
                    metrics.failed += 1;
                    failures.push(format!(
                        "[{}|{}|{}] {} — unexpected assumption kinds {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        inv_trig_policy_label(case.inv_trig),
                        case.expr,
                        first.assumption_kinds
                    ));
                    case_failed = true;
                }
            }
        }

        let first_kinds: std::collections::HashSet<_> =
            first.assumption_kinds.iter().cloned().collect();
        let second_kinds: std::collections::HashSet<_> =
            second.assumption_kinds.iter().cloned().collect();
        let introduced_kinds: Vec<_> = second_kinds.difference(&first_kinds).cloned().collect();

        if !introduced_kinds.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — introduced assumption kinds {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                introduced_kinds,
                first.assumption_kinds,
                second.assumption_kinds
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.assumption_kinds == second.assumption_kinds {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}|{}] {} — assumption kinds {:?} -> {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        inv_trig_policy_label(case.inv_trig),
                        case.expr,
                        first.assumption_kinds,
                        second.assumption_kinds
                    ));
                }
            }
        } else if verbose {
            eprintln!(
                "  ❌ [{}|{}|{}] {}",
                case.family,
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr
            );
        }
    }

    eprintln!(
        "✅ Assumption trace contracts: exact={} relaxed={} expected_present={} expected_absent={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_present,
        metrics.expected_absent,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ assumption trace relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 assumption trace failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

pub(super) fn run_known_domain_frontier_pair_tests() -> ComboMetrics {
    let pairs = load_known_domain_frontier_pairs();
    run_direct_pair_tests(
        pairs,
        "known domain-frontier metamorphic tests",
        "Known domain-frontier tests",
        MetatestShortcutMode::SmokeClosure,
    )
}

pub(super) fn run_known_domain_frontier_safe_pair_tests() -> ComboMetrics {
    let pairs = load_known_domain_frontier_safe_pairs();
    run_direct_pair_tests_with_frontier_policy(
        pairs,
        "known domain-frontier safe-window metamorphic tests",
        "Known domain-frontier safe-window tests",
        false,
        true,
        MetatestShortcutMode::SmokeClosure,
    )
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_requires_contracts -- --ignored --nocapture
fn metatest_simplify_requires_contracts() {
    let m = run_requires_contract_tests();
    assert_eq!(m.failed, 0, "{} requires contracts failed", m.failed);
    assert_eq!(
        m.parse_errors, 0,
        "{} requires contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_requires_mode_contracts -- --ignored --nocapture
fn metatest_simplify_requires_mode_contracts() {
    let m = run_requires_mode_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} mode-aware requires contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} mode-aware requires contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_assumption_trace_contracts -- --ignored --nocapture
fn metatest_simplify_assumption_trace_contracts() {
    let m = run_assumption_trace_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} assumption trace contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} assumption trace contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
fn known_raw_domain_frontier_detects_rational_ctx_log_square_pair() {
    assert_eq!(
        known_raw_domain_frontier_reason(
            "ln((1/(u - 1) + 1/(u + 1))^2)",
            "2*ln((1/(u - 1) + 1/(u + 1)))"
        ),
        Some("log-square expansion changes domain")
    );
}

#[test]
fn known_domain_frontier_detects_mul_inverse_trig_pair() {
    assert_eq!(
        known_domain_frontier_reason(
            "((exp(x)-exp(-x))/2)*(sin(2*arcsin(u)))",
            "(sinh(x))*(2*u*sqrt(1-u^2))"
        ),
        Some("inverse-trig branch introduces domain/branch sensitivity")
    );
}

#[test]
fn known_domain_frontier_detects_mul_sqrt_product_pair() {
    assert_eq!(
        known_domain_frontier_reason(
            "(cos(3*pi/8))*(sqrt(u)*sqrt(4*u))",
            "(sqrt(2-sqrt(2))/2)*(2*u)"
        ),
        Some("sqrt product contraction changes sign/domain behavior")
    );
}

#[test]
fn known_domain_frontier_primary_and_safe_window_metrics_stay_complementary() {
    let primary = run_known_domain_frontier_pair_tests();
    let safe = run_known_domain_frontier_safe_pair_tests();
    let pair_count = load_known_domain_frontier_pairs().len();

    assert_eq!(primary.failed, 0);
    assert_eq!(primary.timeouts, 0);
    assert_eq!(primary.numeric_only, 0);
    assert_eq!(
        primary.nf_convergent + primary.known_domain_frontier_count(),
        pair_count
    );
    assert_eq!(primary.inconclusive, primary.known_domain_frontier_count());
    assert_eq!(primary.proved_symbolic(), 0);

    assert_eq!(safe.failed, 0);
    assert_eq!(safe.timeouts, 0);
    assert_eq!(safe.inconclusive, 0);
    assert_eq!(safe.numeric_only, 0);
    assert_eq!(safe.nf_convergent + safe.proved_symbolic(), pair_count);

    assert_eq!(
        primary.known_domain_frontier_count(),
        safe.proved_symbolic(),
        "safe-window should symbolically close exactly the frontier cases reported by the primary suite"
    );
}

#[test]
fn combo_metrics_known_domain_frontier_count_sums_domain_frontier_causes_only() {
    let mut causes = HashMap::new();
    causes.insert(
        "domain-frontier: inverse-trig branch introduces domain/branch sensitivity".to_string(),
        3,
    );
    causes.insert(
        "domain-frontier: log-square expansion changes domain".to_string(),
        2,
    );
    causes.insert("too few valid samples".to_string(), 1);

    let metrics = ComboMetrics {
        op: "test".to_string(),
        pairs: 0,
        families: 0,
        combos: 0,
        nf_convergent: 0,
        proved_quotient: 0,
        proved_difference: 0,
        proved_composed: 0,
        numeric_only: 0,
        inconclusive: 6,
        failed: 0,
        skipped: 0,
        timeouts: 0,
        cycle_events_total: 0,
        known_symbolic_residuals: 0,
        numeric_only_causes: HashMap::new(),
        inconclusive_causes: causes,
        domain_frontier_examples: Vec::new(),
    };

    assert_eq!(metrics.known_domain_frontier_count(), 5);
}

#[test]
fn rational_ctx_log_square_rule_is_domain_sensitive_without_filter() {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse("ln((1/(u - 1) + 1/(u + 1))^2)", &mut simplifier.context).expect("lhs");
    let rhs = parse("2*ln((1/(u - 1) + 1/(u + 1)))", &mut simplifier.context).expect("rhs");

    let (lhs_simp, _) = simplifier.simplify(lhs);
    let (rhs_simp, _) = simplifier.simplify(rhs);
    let diff = simplifier
        .context
        .add(cas_ast::Expr::Sub(lhs_simp, rhs_simp));
    let (diff_simp, _) = simplifier.simplify(diff);
    let residual_shape = expr_shape_signature(&simplifier.context, diff_simp);

    let cause = numeric_only_cause_for_vars(
        &simplifier.context,
        lhs_simp,
        rhs_simp,
        &[String::from("u")],
        &[FilterSpec::None],
        &metatest_config(),
        &residual_shape,
    );

    assert!(matches!(cause, NumericOnlyCause::DomainSensitive));

    let outcome = classify_numeric_equiv_for_vars(
        &simplifier.context,
        lhs_simp,
        rhs_simp,
        &[String::from("u")],
        &[FilterSpec::Range { min: 1.1, max: 3.0 }],
        &metatest_config(),
    );
    assert!(matches!(outcome, NumericCheckOutcome::Pass));
}

#[test]
fn build_nvar_slice_anchors_prefers_positive_domain_when_needed() {
    let mut ctx = Context::new();
    let lhs = parse("exp(ln(x)+ln(y))+z", &mut ctx).expect("parse lhs");
    let rhs = parse("x*y+z", &mut ctx).expect("parse rhs");
    let vars = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    let filters = vec![FilterSpec::None, FilterSpec::None, FilterSpec::None];
    let anchors = build_nvar_slice_anchors(
        &ctx,
        lhs,
        rhs,
        &vars,
        &filters,
        &metatest_config(),
        0.173_205_080_756_887_73,
    );

    let map = anchors.into_iter().collect::<HashMap<String, f64>>();
    assert!(
        map["x"] > 0.0,
        "expected x anchor to be positive, got {}",
        map["x"]
    );
    assert!(
        map["y"] > 0.0,
        "expected y anchor to be positive, got {}",
        map["y"]
    );
}
