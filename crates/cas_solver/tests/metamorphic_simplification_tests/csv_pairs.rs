//! `metamorphic_simplification_tests`: familia `csv_pairs`.
//!
//! Ver la cabecera de `metamorphic_simplification_tests.rs` para el contexto.

use super::*;

#[test]
fn curated_pair_corpus_proves_identity_pair_with_alpha_renaming() {
    let lhs = "1/a + 1/(a+1)";
    let rhs = "(2*a+1)/(a*(a+1))";
    assert!(prove_zero_from_curated_pair_corpus_text(lhs, rhs));
    assert!(prove_zero_from_curated_pair_corpus_text(rhs, lhs));
}

/// Load identity pairs from CSV file
/// Supports two formats:
/// - Legacy 4-col: exp,simp,vars,mode (bucket=conditional_requires, branch=principal_strict)
/// - Extended 7-col: exp,simp,vars,domain_mode,bucket,branch_mode,filter
fn parse_identity_pair_csv_fields(line_num: usize, line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes {
                    if matches!(chars.peek(), Some('"')) {
                        current.push('"');
                        chars.next();
                    } else {
                        in_quotes = false;
                    }
                } else if current.trim().is_empty() {
                    current.clear();
                    in_quotes = true;
                } else {
                    panic!(
                        "identity_pairs.csv:{} invalid quote placement :: {}",
                        line_num, line
                    );
                }
            }
            ',' if !in_quotes => {
                fields.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    assert!(
        !in_quotes,
        "identity_pairs.csv:{} unterminated quote :: {}",
        line_num, line
    );
    fields.push(current.trim().to_string());
    fields
}

fn validate_identity_pair_text(label: &str, text: &str, line_num: usize) {
    let mut simplifier = Simplifier::with_default_rules();
    if let Err(err) = parse(text, &mut simplifier.context) {
        panic!(
            "identity_pairs.csv:{} invalid {} parse: {:?} :: {}",
            line_num, label, err, text
        );
    }
}

fn parse_identity_pairs() -> Vec<IdentityPair> {
    let csv_path = find_test_data_file("identity_pairs.csv");
    let content = std::fs::read_to_string(csv_path).expect("Failed to read identity_pairs.csv");

    let mut pairs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1; // 1-indexed for humans
        let line = line.trim();
        // Track family from comment headers, skip other comments and empty lines
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            // Skip structural headers (format/description lines)
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Each row")
                && !label.starts_with("var is")
                && !label.starts_with("Mathematical Identity")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts = parse_identity_pair_csv_fields(line_num, line);

        if parts.len() >= 7 {
            // Extended 7-column format: exp,simp,vars,domain_mode,bucket,branch_mode,filter
            let vars: Vec<String> = parts[2]
                .trim()
                .split(';')
                .map(|s| s.trim().to_string())
                .collect();

            let mode = parse_domain_mode(parts[3].trim());
            let bucket = parse_bucket(parts[4].trim());
            let branch_mode = parse_branch_mode(parts[5].trim());
            let filter_spec = parse_filter_spec(parts[6].trim(), line_num);

            let pair = IdentityPair {
                exp: parts[0].trim().to_string(),
                simp: parts[1].trim().to_string(),
                vars,
                mode,
                bucket,
                branch_mode,
                filter_spec,
                family: current_family.clone(),
            };
            validate_identity_pair_text("exp", &pair.exp, line_num);
            validate_identity_pair_text("simp", &pair.simp, line_num);
            pairs.push(pair);
        } else if parts.len() >= 3 {
            // Legacy 4-column format: exp,simp,vars,mode
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

            let pair = IdentityPair {
                exp: parts[0].trim().to_string(),
                simp: parts[1].trim().to_string(),
                vars,
                mode,
                bucket: legacy_bucket_from_env(), // Configurable via METATEST_LEGACY_BUCKET
                branch_mode: BranchMode::default(),
                filter_spec: FilterSpec::None,
                family: current_family.clone(),
            };
            validate_identity_pair_text("exp", &pair.exp, line_num);
            validate_identity_pair_text("simp", &pair.simp, line_num);
            pairs.push(pair);
        } else {
            panic!(
                "identity_pairs.csv:{} malformed row with {} columns :: {}",
                line_num,
                parts.len(),
                line
            );
        }
    }

    pairs
}

pub(super) fn load_identity_pairs() -> Vec<IdentityPair> {
    static IDENTITY_PAIRS: OnceLock<Vec<IdentityPair>> = OnceLock::new();
    IDENTITY_PAIRS.get_or_init(parse_identity_pairs).clone()
}

/// Run combination tests from CSV pairs
pub(super) fn run_csv_combination_tests(
    max_pairs: usize,
    include_triples: bool,
    op: CombineOp,
) -> ComboMetrics {
    run_csv_combination_tests_with_shortcut_mode(
        max_pairs,
        include_triples,
        op,
        MetatestShortcutMode::SmokeClosure,
    )
}

pub(super) fn run_csv_combination_tests_strict(
    max_pairs: usize,
    include_triples: bool,
    op: CombineOp,
) -> ComboMetrics {
    run_csv_combination_tests_with_shortcut_mode(
        max_pairs,
        include_triples,
        op,
        MetatestShortcutMode::StrictPressure,
    )
}

pub(super) fn run_csv_combination_tests_nf_first(
    max_pairs: usize,
    include_triples: bool,
    op: CombineOp,
) -> ComboMetrics {
    run_csv_combination_tests_with_shortcut_mode(
        max_pairs,
        include_triples,
        op,
        MetatestShortcutMode::NfFirstPressure,
    )
}

fn run_csv_combination_tests_with_shortcut_mode(
    max_pairs: usize,
    include_triples: bool,
    op: CombineOp,
    shortcut_mode: MetatestShortcutMode,
) -> ComboMetrics {
    let all_pairs = load_identity_pairs();
    let config = metatest_config();

    // Filter out Assume-only identities: combination tests run in Generic mode,
    // so identities requiring DomainMode::Assume (like 0^x→0) would always fail symbolically.
    let all_pairs: Vec<_> = all_pairs
        .into_iter()
        .filter(|p| p.mode != DomainRequirement::Assume)
        .collect();

    // Offset support: METATEST_START_OFFSET=100 to skip first 100 identities
    let start_offset = std::env::var("METATEST_START_OFFSET")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    // Selection mode: stratified (default) or legacy shuffled window.
    // Stratified guarantees ≥1 pair per CSV family for representative coverage.
    // Use METATEST_NOSHUFFLE=1 for old contiguous behavior (family debugging).
    let no_shuffle = std::env::var("METATEST_NOSHUFFLE").is_ok();

    let pairs: Vec<_> = if no_shuffle {
        // Legacy: contiguous window from start_offset
        all_pairs
            .into_iter()
            .skip(start_offset)
            .take(max_pairs)
            .collect()
    } else {
        // Stratified sampling: 1 representative per family, then fill randomly
        // Seed configurable via METATEST_SEED (default 0xC0FFEE, legacy 42)
        stratified_select(all_pairs, max_pairs, config.seed)
    };
    let n = pairs.len();
    let num_families = {
        let mut fams: Vec<&str> = pairs.iter().map(|p| p.family.as_str()).collect();
        fams.sort();
        fams.dedup();
        fams.len()
    };

    eprintln!(
        "📊 Running CSV combination tests [{}] with {} pairs from {} families (seed {}, offset {}, {})",
        op.name(),
        n,
        num_families,
        config.seed,
        start_offset,
        if no_shuffle { "ordered" } else { "stratified" }
    );

    // Verbose mode: show nf_mismatch examples
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let trace_combo = std::env::var("METATEST_TRACE_COMBO").is_ok();
    let max_examples = std::env::var("METATEST_MAX_EXAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let progress_every = std::env::var("METATEST_PROGRESS_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_METATEST_PROGRESS_EVERY);
    let requested_combo_cap = std::env::var("METATEST_MAX_COMBOS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0);
    let requested_combo_start = std::env::var("METATEST_COMBO_START")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    let mut passed = 0;
    let mut failed = 0;
    // Classification counters:
    // - nf_convergent: simplify(LHS) == simplify(RHS) structurally (ideal)
    // - proved_quotient: simplify(LHS/RHS)==1 for Mul/Div, or simplify(LHS-RHS)==0 for Add/Sub
    // - proved_difference: simplify(LHS-RHS)==0 fallback for Mul/Div (engine weakness signal)
    // - numeric_only: only passes numeric check (potential issue or branch-sensitive)
    let mut nf_convergent = 0;
    let mut proved_quotient = 0;
    let mut proved_difference = 0;
    let mut proved_composed = 0;
    let mut numeric_only = 0;
    let mut inconclusive = 0;
    let mut inconclusive_causes: HashMap<String, usize> = HashMap::new();
    let mut numeric_only_causes: HashMap<String, usize> = HashMap::new();
    let mut nf_mismatch_examples: Vec<(String, String, String, String)> = Vec::new();
    let mut proved_composed_examples: Vec<(String, String, String, String)> = Vec::new();
    let mut numeric_only_examples: Vec<(String, String, String, String, String, String, String)> =
        Vec::new(); // (LHS, RHS, simp1, simp2, diff_residual, shape, cause)
    let mut domain_frontier = 0usize;
    let mut domain_frontier_examples: Vec<(String, String, String)> = Vec::new();
    let mut skip_examples: Vec<String> = Vec::new();
    let mut skipped = 0;
    let mut timeouts = 0;
    let mut cycle_events_total: usize = 0;
    let pair_symbolic_ok: Vec<bool> = if shortcut_mode.allows_composed_promotion()
        && shortcut_mode.allows_curated_shortcuts()
        && matches!(
            op,
            CombineOp::Add | CombineOp::Sub | CombineOp::Mul | CombineOp::Div
        ) {
        pairs
            .iter()
            .enumerate()
            .map(|(idx, pair)| {
                if trace_combo {
                    eprintln!(
                        "🔎 Precheck [{}] pair #{} / {} :: [{}] {}",
                        op.name(),
                        idx + 1,
                        n,
                        pair.family,
                        pair.exp
                    );
                }
                pair_is_symbolically_proved(pair)
            })
            .collect()
    } else {
        vec![false; n]
    };
    let pair_raw_pressure_ok: Vec<bool> = if shortcut_mode.allows_composed_promotion()
        && !shortcut_mode.allows_curated_shortcuts()
        && matches!(
            op,
            CombineOp::Add | CombineOp::Sub | CombineOp::Mul | CombineOp::Div
        ) {
        pairs.iter().map(pair_is_raw_pressure_proved).collect()
    } else {
        vec![false; n]
    };

    // Per-combination timeout: mul/div use a tighter release budget to keep
    // large suites like `mul` and the unified benchmark tractable.
    let combo_timeout = combination_timeout(op);

    // For Div, pre-check which identities are safe to use as divisors (not near zero)
    // by evaluating at sample points. This avoids division-by-zero in test combinations.
    let divisor_safe: Vec<bool> = if op == CombineOp::Div {
        pairs
            .iter()
            .map(|p| {
                let mut s = Simplifier::with_default_rules();
                let sample_points = [0.7, 1.3, 2.1];
                if let Ok(e) = parse(&p.exp, &mut s.context) {
                    let var = &p.vars[0];
                    sample_points.iter().all(|&x| {
                        let var_names = vec![var.clone()];
                        let val = cas_math::numeric_eval::eval_f64_with_substitution(
                            &s.context,
                            e,
                            &var_names,
                            &[x],
                        );
                        matches!(val, Some(v) if v.abs() > 0.01)
                    })
                } else {
                    false
                }
            })
            .collect()
    } else {
        vec![true; n]
    };
    let total_double_combos = (0..n)
        .map(|i| {
            ((i + 1)..n)
                .filter(|&j| op != CombineOp::Div || divisor_safe[j])
                .count()
        })
        .sum::<usize>();
    let (combo_start_offset, effective_total_double_combos) = effective_combo_window(
        total_double_combos,
        requested_combo_start,
        requested_combo_cap,
    );
    let mut processed_double_combos = 0usize;
    let mut visited_double_combos = 0usize;
    if combo_start_offset > 0 || effective_total_double_combos < total_double_combos {
        eprintln!(
            "🔬 Applying combo window [{}]: start {} size {} / {} planned double combinations",
            op.name(),
            combo_start_offset,
            effective_total_double_combos,
            total_double_combos
        );
    }

    // Double combinations: all pairs of different identities
    'double_outer: for i in 0..n {
        for j in (i + 1)..n {
            if processed_double_combos >= effective_total_double_combos {
                break 'double_outer;
            }
            if visited_double_combos < combo_start_offset {
                visited_double_combos += 1;
                continue;
            }
            let pair1 = &pairs[i];
            let pair2 = &pairs[j];

            // For Div: pair2 is the divisor, skip if it can be zero
            if op == CombineOp::Div && !divisor_safe[j] {
                continue;
            }

            let mut used_vars: HashSet<String> = pair1.vars.iter().cloned().collect();
            let (pair2_exp, pair2_simp, pair2_vars, pair2_filters) =
                rename_identity_for_combination(pair2, &mut used_vars);
            let mut combined_vars = pair1.vars.clone();
            combined_vars.extend(pair2_vars.clone());
            let mut combined_filters = identity_filters(pair1);
            combined_filters.extend(pair2_filters.clone());
            let pair_composed_ok = if shortcut_mode.allows_composed_promotion() {
                if shortcut_mode.allows_curated_shortcuts() {
                    pair_symbolic_ok[i] && pair_symbolic_ok[j]
                } else {
                    pair_raw_pressure_ok[i] && pair_raw_pressure_ok[j]
                }
            } else {
                false
            };

            let combined_exp = format!("({}) {} ({})", pair1.exp, op.symbol(), pair2_exp);
            let combined_simp = format!("({}) {} ({})", pair1.simp, op.symbol(), pair2_simp);
            if trace_combo {
                eprintln!(
                    "🔎 Combo [{}] #{} / {} :: [{}] ({}) {} [{}] ({})",
                    op.name(),
                    processed_double_combos + 1,
                    effective_total_double_combos,
                    pair1.family,
                    pair1.exp,
                    op.symbol(),
                    pair2.family,
                    pair2.exp
                );
            }
            if matches!(shortcut_mode, MetatestShortcutMode::NfFirstPressure)
                && matches!(op, CombineOp::Add | CombineOp::Sub)
            {
                match classify_nf_first_add_sub_combo_in_child_process(
                    &combined_exp,
                    &combined_simp,
                ) {
                    NfFirstAddSubChildOutcome::Nf => {
                        nf_convergent += 1;
                        passed += 1;
                    }
                    NfFirstAddSubChildOutcome::Proved => {
                        proved_quotient += 1;
                        passed += 1;
                        if verbose && nf_mismatch_examples.len() < max_examples {
                            nf_mismatch_examples.push((
                                combined_exp.clone(),
                                combined_simp.clone(),
                                pair1.simp.clone(),
                                pair2.simp.clone(),
                            ));
                        }
                    }
                    NfFirstAddSubChildOutcome::Inconclusive => {
                        inconclusive += 1;
                    }
                    NfFirstAddSubChildOutcome::Timeout => {
                        timeouts += 1;
                    }
                }
                processed_double_combos += 1;
                visited_double_combos += 1;
                if should_report_combo_progress(
                    verbose,
                    effective_total_double_combos,
                    processed_double_combos,
                    progress_every,
                ) {
                    print_combo_progress(
                        op.name(),
                        &ComboProgressSnapshot {
                            processed_combos: processed_double_combos,
                            total_combos: effective_total_double_combos,
                            nf_convergent,
                            proved_symbolic: proved_quotient + proved_difference + proved_composed,
                            numeric_only,
                            inconclusive,
                            skipped,
                            timeouts,
                            failed,
                        },
                    );
                }
                continue;
            }
            if matches!(shortcut_mode, MetatestShortcutMode::NfFirstPressure)
                && op.is_multiplicative()
            {
                match classify_nf_first_mul_div_combo_in_child_process(
                    &combined_exp,
                    &combined_simp,
                    &combined_vars,
                    &combined_filters,
                    combo_timeout,
                ) {
                    NfFirstMulDivChildOutcome::Nf { cycles } => {
                        nf_convergent += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                    }
                    NfFirstMulDivChildOutcome::ProvedQuotient { cycles } => {
                        proved_quotient += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        if verbose && nf_mismatch_examples.len() < max_examples {
                            nf_mismatch_examples.push((
                                combined_exp.clone(),
                                combined_simp.clone(),
                                pair1.simp.clone(),
                                pair2.simp.clone(),
                            ));
                        }
                    }
                    NfFirstMulDivChildOutcome::ProvedDifference { cycles } => {
                        proved_difference += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        if verbose && nf_mismatch_examples.len() < max_examples {
                            nf_mismatch_examples.push((
                                combined_exp.clone(),
                                combined_simp.clone(),
                                pair1.simp.clone(),
                                pair2.simp.clone(),
                            ));
                        }
                    }
                    NfFirstMulDivChildOutcome::Numeric {
                        diff_str,
                        shape,
                        cause,
                        cycles,
                    } => {
                        numeric_only += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        *numeric_only_causes.entry(cause.clone()).or_default() += 1;
                        if verbose {
                            numeric_only_examples.push((
                                combined_exp.clone(),
                                combined_simp.clone(),
                                pair1.simp.clone(),
                                pair2.simp.clone(),
                                diff_str,
                                shape,
                                cause,
                            ));
                        }
                    }
                    NfFirstMulDivChildOutcome::DomainFrontier {
                        reason,
                        shape: _shape,
                        cause: _cause,
                        cycles,
                    } => {
                        inconclusive += 1;
                        domain_frontier += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        record_inconclusive_reason(
                            &mut inconclusive_causes,
                            "domain_frontier",
                            &reason,
                        );
                        if verbose && domain_frontier_examples.len() < max_examples {
                            domain_frontier_examples.push((
                                combined_exp.clone(),
                                combined_simp.clone(),
                                reason,
                            ));
                        }
                    }
                    NfFirstMulDivChildOutcome::Inconclusive { reason, cycles } => {
                        inconclusive += 1;
                        cycle_events_total += cycles;
                        record_inconclusive_reason(
                            &mut inconclusive_causes,
                            "inconclusive",
                            &reason,
                        );
                    }
                    NfFirstMulDivChildOutcome::Failed { cycles } => {
                        failed += 1;
                        cycle_events_total += cycles;
                        if failed <= 5 {
                            eprintln!(
                                "❌ Double combo [{}] failed: ({}) {} ({})",
                                op.name(),
                                pair1.exp,
                                op.symbol(),
                                pair2.exp
                            );
                        }
                    }
                    NfFirstMulDivChildOutcome::Skip => {
                        skipped += 1;
                        if verbose {
                            push_nf_first_mul_div_skip_example(
                                &mut skip_examples,
                                max_examples,
                                NfFirstMulDivSkipExample {
                                    op_name: op.name(),
                                    pair1,
                                    pair2,
                                    pair2_exp: &pair2_exp,
                                    pair2_simp: &pair2_simp,
                                    combined_exp: &combined_exp,
                                    combined_simp: &combined_simp,
                                },
                            );
                        }
                    }
                    NfFirstMulDivChildOutcome::Timeout => {
                        timeouts += 1;
                        eprintln!(
                            "  ⏱️  T/O [{}] #{}: [{}] {} [{}]  →  ({}) {} ({})",
                            op.name(),
                            timeouts,
                            pair1.family,
                            op.symbol(),
                            pair2.family,
                            pair1.exp,
                            op.symbol(),
                            pair2.exp,
                        );
                        if std::env::var("METATEST_TRACE_TIMEOUT_FULL").is_ok() {
                            eprintln!("        lhs: {}", combined_exp);
                            eprintln!("        rhs: {}", combined_simp);
                        }
                    }
                }
                processed_double_combos += 1;
                visited_double_combos += 1;
                if should_report_combo_progress(
                    verbose,
                    effective_total_double_combos,
                    processed_double_combos,
                    progress_every,
                ) {
                    print_combo_progress(
                        op.name(),
                        &ComboProgressSnapshot {
                            processed_combos: processed_double_combos,
                            total_combos: effective_total_double_combos,
                            nf_convergent,
                            proved_symbolic: proved_quotient + proved_difference + proved_composed,
                            numeric_only,
                            inconclusive,
                            skipped,
                            timeouts,
                            failed,
                        },
                    );
                }
                continue;
            }
            if matches!(
                op,
                CombineOp::Add | CombineOp::Sub | CombineOp::Mul | CombineOp::Div
            ) && pair_composed_ok
            {
                proved_composed += 1;
                passed += 1;
                if verbose && proved_composed_examples.len() < max_examples {
                    proved_composed_examples.push((
                        combined_exp.clone(),
                        combined_simp.clone(),
                        pair1.simp.clone(),
                        pair2.simp.clone(),
                    ));
                }
                if verbose && nf_mismatch_examples.len() < max_examples {
                    nf_mismatch_examples.push((
                        combined_exp.clone(),
                        combined_simp.clone(),
                        pair1.simp.clone(),
                        pair2.simp.clone(),
                    ));
                }
                processed_double_combos += 1;
                visited_double_combos += 1;
                if should_report_combo_progress(
                    verbose,
                    effective_total_double_combos,
                    processed_double_combos,
                    progress_every,
                ) {
                    print_combo_progress(
                        op.name(),
                        &ComboProgressSnapshot {
                            processed_combos: processed_double_combos,
                            total_combos: effective_total_double_combos,
                            nf_convergent,
                            proved_symbolic: proved_quotient + proved_difference + proved_composed,
                            numeric_only,
                            inconclusive,
                            skipped,
                            timeouts,
                            failed,
                        },
                    );
                }
                continue;
            }

            // For Mul/Div: run the entire combo in a thread with hard timeout
            // to prevent hangs when simplify_with_options gets stuck.
            if op.is_multiplicative() {
                let exp_clone = combined_exp.clone();
                let simp_clone = combined_simp.clone();
                let combo_vars = combined_vars.clone();
                let combo_filters = combined_filters.clone();
                let config_clone = config.clone();
                let v = verbose;
                let timeout = combo_timeout;
                let (tx, rx) = std::sync::mpsc::channel();
                let _handle = std::thread::Builder::new()
                    .stack_size(METATEST_WORKER_STACK_SIZE_BYTES)
                    .spawn(move || {
                        let mut simplifier = Simplifier::with_default_rules();
                        let exp_parsed = match parse(&exp_clone, &mut simplifier.context) {
                            Ok(e) => e,
                            Err(_) => {
                                let _ = tx.send(None);
                                return;
                            }
                        };
                        let simp_parsed = match parse(&simp_clone, &mut simplifier.context) {
                            Ok(e) => e,
                            Err(_) => {
                                let _ = tx.send(None);
                                return;
                            }
                        };

                        // Use default budget — the thread-based 2s timeout prevents hangs
                        let opts = cas_solver::runtime::SimplifyOptions::default();
                        let mut combo_cycles: usize = 0;

                        let (mut e, _, stats_e) = simplifier.simplify_with_stats(exp_parsed, opts.clone());
                        combo_cycles += stats_e.cycle_events.len();
                        let (mut s, _, stats_s) = simplifier.simplify_with_stats(simp_parsed, opts.clone());
                        combo_cycles += stats_s.cycle_events.len();

                        // Post-process: fold_constants to match CLI eval_simplify behavior
                        {
                            let cfg = cas_solver::runtime::EvalConfig::default();
                            let mut budget = cas_solver::runtime::Budget::preset_cli();
                            if let Ok(r) = cas_solver::api::fold_constants(&mut simplifier.context, e, &cfg, cas_solver::api::ConstFoldMode::Safe, &mut budget) {
                                e = r.expr;
                            }
                            if let Ok(r) = cas_solver::api::fold_constants(&mut simplifier.context, s, &cfg, cas_solver::api::ConstFoldMode::Safe, &mut budget) {
                                s = r.expr;
                            }
                        }

                        // Check 1: NF convergence
                        let nf_match =
                            cas_solver::runtime::compare_expr(&simplifier.context, e, s)
                                == std::cmp::Ordering::Equal;

                        if nf_match {
                            let _ = tx.send(Some((
                                "nf".to_string(),
                                String::new(),
                                String::new(),
                                String::new(),
                                combo_cycles,
                            )));
                            return;
                        }

                        // Check 2: Proved symbolic — simplify(LHS/RHS) == 1  [fresh context]
                        // Uses a fresh Simplifier to match CLI behavior (no context pollution).
                        {
                            let q_str = format!("({}) / ({})", exp_clone, simp_clone);
                            let mut sq = Simplifier::with_default_rules();
                            if let Ok(qp) = parse(&q_str, &mut sq.context) {
                                let (mut qr, _) = sq.simplify(qp);
                                let cfg = cas_solver::runtime::EvalConfig::default();
                                let mut budget = cas_solver::runtime::Budget::preset_cli();
                                if let Ok(r) = cas_solver::api::fold_constants(&mut sq.context, qr, &cfg, cas_solver::api::ConstFoldMode::Safe, &mut budget) {
                                    qr = r.expr;
                                }
                                let target = num_rational::BigRational::from_integer(1.into());
                                if matches!(sq.context.get(qr), cas_ast::Expr::Number(n) if *n == target) {
                                    let _ = tx.send(Some((
                                        "proved-q".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                    return;
                                }
                            }
                        }

                        // Check 2b: Difference fallback — simplify(LHS - RHS) == 0  [fresh context]
                        {
                            let d_str = format!("({}) - ({})", exp_clone, simp_clone);
                            let mut sd = Simplifier::with_default_rules();
                            if let Ok(dp) = parse(&d_str, &mut sd.context) {
                                let (mut dr, _) = sd.simplify(dp);
                                let cfg = cas_solver::runtime::EvalConfig::default();
                                let mut budget = cas_solver::runtime::Budget::preset_cli();
                                if let Ok(r) = cas_solver::api::fold_constants(&mut sd.context, dr, &cfg, cas_solver::api::ConstFoldMode::Safe, &mut budget) {
                                    dr = r.expr;
                                }
                                let zero = num_rational::BigRational::from_integer(0.into());
                                if matches!(sd.context.get(dr), cas_ast::Expr::Number(n) if *n == zero) {
                                    let _ = tx.send(Some((
                                        "proved-d".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                    return;
                                }
                            }
                        }

                        // Check 2c: Expand fallback — expand(LHS - RHS) == 0  [fresh context]
                        // Bridges trig identities gated behind expand_mode (Ticket 6c).
                        {
                            let d_str = format!("({}) - ({})", exp_clone, simp_clone);
                            let mut sd = Simplifier::with_default_rules();
                            if let Ok(dp) = parse(&d_str, &mut sd.context) {
                                let (mut dr, _) = sd.expand(dp);
                                let cfg = cas_solver::runtime::EvalConfig::default();
                                let mut budget = cas_solver::runtime::Budget::preset_cli();
                                if let Ok(r) = cas_solver::api::fold_constants(&mut sd.context, dr, &cfg, cas_solver::api::ConstFoldMode::Safe, &mut budget) {
                                    dr = r.expr;
                                }
                                let zero = num_rational::BigRational::from_integer(0.into());
                                if matches!(sd.context.get(dr), cas_ast::Expr::Number(n) if *n == zero) {
                                    let _ = tx.send(Some((
                                        "proved-d".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                    return;
                                }
                            }
                        }

                        if prove_zero_from_metamorphic_texts(
                            &mut simplifier,
                            &exp_clone,
                            &simp_clone,
                            e,
                            s,
                        ) {
                            let _ = tx.send(Some((
                                "proved-d".to_string(),
                                String::new(),
                                String::new(),
                                String::new(),
                                combo_cycles,
                            )));
                            return;
                        }

                        // Check 3: Numeric equivalence
                        match classify_numeric_equiv_for_vars(
                            &simplifier.context,
                            e,
                            s,
                            &combo_vars,
                            &combo_filters,
                            &config_clone,
                        ) {
                            NumericCheckOutcome::Pass => {
                                // Diagnostic: show what engine actually produced for LHS-RHS
                                let diff_str = if v {
                                    let d_diag = simplifier.context.add(cas_ast::Expr::Sub(e, s));
                                    let (d_simp, _) = simplifier.simplify(d_diag);
                                    format!(
                                        "simplify(LHS-RHS) => {}",
                                        cas_formatter::LaTeXExpr { context: &simplifier.context, id: d_simp }.to_latex()
                                    )
                                } else {
                                    String::new()
                                };
                                let shape = if v {
                                    let d_diag = simplifier.context.add(cas_ast::Expr::Sub(e, s));
                                    let (d_simp, _) = simplifier.simplify(d_diag);
                                    expr_shape_signature(&simplifier.context, d_simp)
                                } else {
                                    String::new()
                                };
                                let cause = numeric_only_cause_for_vars(
                                    &simplifier.context,
                                    e,
                                    s,
                                    &combo_vars,
                                    &combo_filters,
                                    &config_clone,
                                    &shape,
                                )
                                .label()
                                .to_string();
                                let kind = if should_promote_numeric_to_composed(
                                    op,
                                    pair_composed_ok,
                                    &cause,
                                ) {
                                    "proved-composed".to_string()
                                } else if let Some(reason) =
                                    known_domain_frontier_reason_for_numeric_cause(
                                        &cause,
                                        &exp_clone,
                                        &simp_clone,
                                    )
                                {
                                        let _ = tx.send(Some((
                                            "domain_frontier".to_string(),
                                            reason.to_string(),
                                            shape,
                                            cause,
                                            combo_cycles,
                                        )));
                                        return;
                                } else {
                                    "numeric".to_string()
                                };
                                let _ =
                                    tx.send(Some((kind, diff_str, shape, cause, combo_cycles)));
                            }
                            NumericCheckOutcome::Inconclusive(reason) => {
                                if pair_composed_ok {
                                    let _ = tx.send(Some((
                                        "proved-composed".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                } else {
                                    let _ = tx.send(Some((
                                        "inconclusive".to_string(),
                                        reason,
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                }
                            }
                            NumericCheckOutcome::Failed(_) => {
                                if pair_composed_ok {
                                    let _ = tx.send(Some((
                                        "proved-composed".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                } else {
                                    let _ = tx.send(Some((
                                        "failed".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                }
                            }
                        }
                    });

                match rx.recv_timeout(timeout) {
                    Ok(Some((kind, diff_str, shape, cause, cycles))) => match kind.as_str() {
                        "nf" => {
                            nf_convergent += 1;
                            passed += 1;
                            cycle_events_total += cycles;
                        }
                        "proved-q" | "proved-d" => {
                            if kind.as_str() == "proved-q" {
                                proved_quotient += 1;
                            } else {
                                proved_difference += 1;
                            }
                            passed += 1;
                            cycle_events_total += cycles;
                            if verbose && nf_mismatch_examples.len() < max_examples {
                                nf_mismatch_examples.push((
                                    combined_exp.clone(),
                                    combined_simp.clone(),
                                    pair1.simp.clone(),
                                    pair2.simp.clone(),
                                ));
                            }
                        }
                        "proved-composed" => {
                            proved_composed += 1;
                            passed += 1;
                            cycle_events_total += cycles;
                            if verbose && proved_composed_examples.len() < max_examples {
                                proved_composed_examples.push((
                                    combined_exp.clone(),
                                    combined_simp.clone(),
                                    pair1.simp.clone(),
                                    pair2.simp.clone(),
                                ));
                            }
                            if verbose && nf_mismatch_examples.len() < max_examples {
                                nf_mismatch_examples.push((
                                    combined_exp.clone(),
                                    combined_simp.clone(),
                                    pair1.simp.clone(),
                                    pair2.simp.clone(),
                                ));
                            }
                        }
                        "numeric" => {
                            numeric_only += 1;
                            passed += 1;
                            cycle_events_total += cycles;
                            *numeric_only_causes.entry(cause.clone()).or_default() += 1;
                            if verbose {
                                numeric_only_examples.push((
                                    combined_exp.clone(),
                                    combined_simp.clone(),
                                    pair1.simp.clone(),
                                    pair2.simp.clone(),
                                    diff_str,
                                    shape,
                                    cause,
                                ));
                            }
                        }
                        "domain_frontier" => {
                            inconclusive += 1;
                            domain_frontier += 1;
                            passed += 1;
                            cycle_events_total += cycles;
                            record_inconclusive_reason(
                                &mut inconclusive_causes,
                                "domain_frontier",
                                &diff_str,
                            );
                            if verbose && domain_frontier_examples.len() < max_examples {
                                domain_frontier_examples.push((
                                    combined_exp.clone(),
                                    combined_simp.clone(),
                                    diff_str,
                                ));
                            }
                        }
                        "inconclusive" => {
                            inconclusive += 1;
                            cycle_events_total += cycles;
                            record_inconclusive_reason(
                                &mut inconclusive_causes,
                                "inconclusive",
                                &diff_str,
                            );
                        }
                        _ => {
                            failed += 1;
                            cycle_events_total += cycles;
                            if failed <= 5 {
                                eprintln!(
                                    "❌ Double combo [{}] failed: ({}) {} ({})",
                                    op.name(),
                                    pair1.exp,
                                    op.symbol(),
                                    pair2.exp
                                );
                            }
                        }
                    },
                    Ok(None) => { /* parse error, skip */ }
                    Err(_) => {
                        if pair_composed_ok {
                            proved_composed += 1;
                            passed += 1;
                            if verbose && proved_composed_examples.len() < max_examples {
                                proved_composed_examples.push((
                                    combined_exp.clone(),
                                    combined_simp.clone(),
                                    pair1.simp.clone(),
                                    pair2.simp.clone(),
                                ));
                            }
                            if verbose && nf_mismatch_examples.len() < max_examples {
                                nf_mismatch_examples.push((
                                    combined_exp.clone(),
                                    combined_simp.clone(),
                                    pair1.simp.clone(),
                                    pair2.simp.clone(),
                                ));
                            }
                        } else {
                            // Timeout — thread is still running but we move on
                            timeouts += 1;
                            eprintln!(
                                "  ⏱️  T/O [{}] #{}: [{}] {} [{}]  →  ({}) {} ({})",
                                op.name(),
                                timeouts,
                                pair1.family,
                                op.symbol(),
                                pair2.family,
                                pair1.exp,
                                op.symbol(),
                                pair2.exp,
                            );
                            if std::env::var("METATEST_TRACE_TIMEOUT_FULL").is_ok() {
                                eprintln!("        lhs: {}", combined_exp);
                                eprintln!("        rhs: {}", combined_simp);
                            }
                        }
                    }
                }
                processed_double_combos += 1;
                visited_double_combos += 1;
                if should_report_combo_progress(
                    verbose,
                    effective_total_double_combos,
                    processed_double_combos,
                    progress_every,
                ) {
                    print_combo_progress(
                        op.name(),
                        &ComboProgressSnapshot {
                            processed_combos: processed_double_combos,
                            total_combos: effective_total_double_combos,
                            nf_convergent,
                            proved_symbolic: proved_quotient + proved_difference + proved_composed,
                            numeric_only,
                            inconclusive,
                            skipped,
                            timeouts,
                            failed,
                        },
                    );
                }
                continue; // skip the inline path below
            }

            // Add/Sub became unsafe to keep inline once strict mode started
            // exercising real engine pressure. Run those combos on a deep-stack
            // worker in strict mode so one pathological identity does not abort
            // the whole benchmark process.
            let combo_result: Result<(String, String, String, String, usize), ()> = if shortcut_mode
                .requires_deep_combo_worker()
            {
                let exp_clone = combined_exp.clone();
                let simp_clone = combined_simp.clone();
                let combo_vars = combined_vars.clone();
                let combo_filters = combined_filters.clone();
                let config_clone = config.clone();
                let v = verbose;
                let timeout = combo_timeout;
                let pair_composed = pair_composed_ok;
                let shortcut_mode_clone = shortcut_mode;
                let (tx, rx) = std::sync::mpsc::channel();
                let _handle = std::thread::Builder::new()
                    .stack_size(METATEST_DEEP_WORKER_STACK_SIZE_BYTES)
                    .spawn(move || {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            evaluate_add_sub_combo(
                                &exp_clone,
                                &simp_clone,
                                &combo_vars,
                                &combo_filters,
                                &config_clone,
                                v,
                                timeout,
                                op,
                                pair_composed,
                                shortcut_mode_clone,
                            )
                        }))
                        .map_err(|_| ());
                        let _ = tx.send(result);
                    });
                match rx.recv_timeout(timeout) {
                    Ok(result) => result,
                    Err(_) => Ok((
                        "timeout".to_string(),
                        String::new(),
                        String::new(),
                        String::new(),
                        0,
                    )),
                }
            } else {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    evaluate_add_sub_combo(
                        &combined_exp,
                        &combined_simp,
                        &combined_vars,
                        &combined_filters,
                        &config,
                        verbose,
                        combo_timeout,
                        op,
                        pair_composed_ok,
                        shortcut_mode,
                    )
                }))
                .map_err(|_| ())
            };

            match combo_result {
                Ok((kind, diff_str, shape, cause, cycles)) => match kind.as_str() {
                    "nf" => {
                        nf_convergent += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                    }
                    "proved" | "proved-composed" => {
                        if kind == "proved-composed" {
                            proved_composed += 1;
                            if verbose && proved_composed_examples.len() < max_examples {
                                proved_composed_examples.push((
                                    combined_exp.clone(),
                                    combined_simp.clone(),
                                    pair1.simp.clone(),
                                    pair2.simp.clone(),
                                ));
                            }
                        } else {
                            proved_quotient += 1;
                        }
                        passed += 1;
                        cycle_events_total += cycles;
                        if verbose && nf_mismatch_examples.len() < max_examples {
                            nf_mismatch_examples.push((
                                combined_exp.clone(),
                                combined_simp.clone(),
                                pair1.simp.clone(),
                                pair2.simp.clone(),
                            ));
                        }
                    }
                    "numeric" => {
                        numeric_only += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        *numeric_only_causes.entry(cause.clone()).or_default() += 1;
                        if verbose {
                            numeric_only_examples.push((
                                combined_exp.clone(),
                                combined_simp.clone(),
                                pair1.simp.clone(),
                                pair2.simp.clone(),
                                diff_str,
                                shape,
                                cause,
                            ));
                        }
                    }
                    "domain_frontier" => {
                        inconclusive += 1;
                        domain_frontier += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        record_inconclusive_reason(
                            &mut inconclusive_causes,
                            "domain_frontier",
                            &diff_str,
                        );
                        if verbose && domain_frontier_examples.len() < max_examples {
                            domain_frontier_examples.push((
                                combined_exp.clone(),
                                combined_simp.clone(),
                                diff_str,
                            ));
                        }
                    }
                    "inconclusive" => {
                        inconclusive += 1;
                        cycle_events_total += cycles;
                        record_inconclusive_reason(
                            &mut inconclusive_causes,
                            "inconclusive",
                            &diff_str,
                        );
                    }
                    "timeout" => {
                        timeouts += 1;
                        cycle_events_total += cycles;
                    }
                    "skip" => { /* parse error, silently continue */ }
                    _ => {
                        failed += 1;
                        cycle_events_total += cycles;
                        if failed <= 5 {
                            eprintln!(
                                "❌ Double combo [{}] failed: ({}) {} ({})",
                                op.name(),
                                pair1.exp,
                                op.symbol(),
                                pair2.exp
                            );
                        }
                    }
                },
                Err(_) => {
                    // Panic caught (e.g., num-rational denominator==0) — treat as skip
                    skipped += 1;
                }
            }

            processed_double_combos += 1;
            visited_double_combos += 1;
            if should_report_combo_progress(
                verbose,
                effective_total_double_combos,
                processed_double_combos,
                progress_every,
            ) {
                print_combo_progress(
                    op.name(),
                    &ComboProgressSnapshot {
                        processed_combos: processed_double_combos,
                        total_combos: effective_total_double_combos,
                        nf_convergent,
                        proved_symbolic: proved_quotient + proved_difference + proved_composed,
                        numeric_only,
                        inconclusive,
                        skipped,
                        timeouts,
                        failed,
                    },
                );
            }
        }
    }

    eprintln!(
        "✅ Double combinations [{}]: {} passed, {} failed, {} skipped (timeout), {} inconclusive",
        op.name(),
        passed,
        failed,
        skipped,
        inconclusive
    );
    eprintln!(
        "   📐 NF-convergent: {} | 🔢 Proved-symbolic: {} (quotient: {}, diff: {}, composed: {}) | 🌡️ Numeric-only: {} | ◐ Inconclusive: {}",
        nf_convergent,
        proved_quotient + proved_difference + proved_composed,
        proved_quotient,
        proved_difference,
        proved_composed,
        numeric_only,
        inconclusive
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
    if verbose && numeric_only > 0 {
        print_numeric_only_cause_breakdown(&numeric_only_causes);
    }

    // Print NF-mismatch examples if verbose (proved_symbolic but different normal forms)
    if verbose && !nf_mismatch_examples.is_empty() {
        eprintln!("\n🔢 NF-mismatch examples (proved symbolic but different normal forms):");
        for (i, (lhs, rhs, simp1, simp2)) in nf_mismatch_examples.iter().enumerate() {
            eprintln!("   {:2}. LHS: {}", i + 1, lhs);
            eprintln!("       RHS: {}", rhs);
            eprintln!("       (simplifies: {} + {})", simp1, simp2);
        }
        if proved_quotient + proved_difference + proved_composed > max_examples {
            eprintln!(
                "   ... and {} more (set METATEST_MAX_EXAMPLES=N to show more)",
                proved_quotient + proved_difference + proved_composed - max_examples
            );
        }
        eprintln!();
    }

    if verbose && !proved_composed_examples.is_empty() {
        eprintln!(
            "🧩 Proved-composed examples (derived from independently proved source identities):"
        );
        for (i, (lhs, rhs, simp1, simp2)) in proved_composed_examples.iter().enumerate() {
            eprintln!("   {:2}. LHS: {}", i + 1, lhs);
            eprintln!("       RHS: {}", rhs);
            eprintln!("       (sources: {} | {})", simp1, simp2);
        }
        if proved_composed > max_examples {
            eprintln!(
                "   ... and {} more (set METATEST_MAX_EXAMPLES=N to show more)",
                proved_composed - max_examples
            );
        }
        eprintln!();
    }

    // Print numeric-only examples if verbose
    if verbose && !numeric_only_examples.is_empty() {
        eprintln!("🌡️ Numeric-only examples (no symbolic proof found):");
        for (i, (lhs, rhs, _simp1, _simp2, diff_residual, _shape, cause)) in
            numeric_only_examples.iter().take(max_examples).enumerate()
        {
            eprintln!("   {:2}. LHS: {}", i + 1, lhs);
            eprintln!("       RHS: {}", rhs);
            eprintln!("       Cause: {}", cause);
            eprintln!("       simplify(LHS-RHS): {}", diff_residual);
        }
        if numeric_only > max_examples {
            eprintln!(
                "   ... and {} more (set METATEST_MAX_EXAMPLES=N to show more)",
                numeric_only - max_examples
            );
        }
        eprintln!();

        // Family classifier for numeric-only cases - stores expressions per family
        let mut family_examples: HashMap<&str, Vec<(String, String)>> = HashMap::new();

        for (lhs, rhs, _, _, _, _, _) in &numeric_only_examples {
            let combined = format!("{} {}", lhs, rhs);
            let expr_pair = (lhs.clone(), rhs.clone());

            // Detect function families (mutually exclusive for cleaner grouping)
            let family = if combined.contains("sec") || combined.contains("csc") {
                "sec/csc (Pythagorean: tan²+1=sec², 1+cot²=csc²)"
            } else if combined.contains("tan(") && !combined.contains("arctan") {
                "tan (without sec/csc)"
            } else if combined.contains("cot(") {
                "cot (without csc)"
            } else if combined.contains("sin(")
                && (combined.contains("/2") || combined.contains("*2"))
            {
                "half/double angle"
            } else if combined.contains("ln(") || combined.contains("log(") {
                "ln/log"
            } else if combined.contains("exp(") {
                "exp"
            } else if combined.contains("sqrt(") || combined.contains("^(1/") {
                "sqrt/roots"
            } else if combined.contains("abs(") {
                "abs"
            } else if combined.contains("arctan")
                || combined.contains("arcsin")
                || combined.contains("arccos")
            {
                "arc* (inverse trig)"
            } else {
                "other"
            };

            family_examples.entry(family).or_default().push(expr_pair);
        }

        if !family_examples.is_empty() {
            eprintln!("📊 Numeric-only grouped by family:");

            // Sort families by count
            let mut sorted: Vec<_> = family_examples.into_iter().collect();
            sorted.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

            for (family, examples) in sorted {
                eprintln!("\n   ── {} ({} cases) ──", family, examples.len());
                let show_count = examples.len().min(max_examples.max(3)); // Show at least 3
                for (lhs, rhs) in examples.iter().take(show_count) {
                    eprintln!("      LHS: {}", lhs);
                    eprintln!("      RHS: {}", rhs);
                    eprintln!();
                }
                if examples.len() > show_count {
                    eprintln!("      ... and {} more", examples.len() - show_count);
                }
            }
            eprintln!();
        }

        // Top-N Shape Analysis: identify dominant patterns in residuals
        eprintln!("📈 Top-N Shape Analysis (residual patterns):");
        let mut shape_counts: HashMap<String, (usize, String)> = HashMap::new(); // shape -> (count, example_diff)

        for (_lhs, _rhs, _, _, diff_residual, shape, _) in &numeric_only_examples {
            let entry = shape_counts
                .entry(shape.clone())
                .or_insert((0, diff_residual.clone()));
            entry.0 += 1;
        }

        let mut sorted_shapes: Vec<_> = shape_counts.into_iter().collect();
        sorted_shapes.sort_by(|a, b| b.1 .0.cmp(&a.1 .0)); // Sort by count descending

        let total = numeric_only_examples.len();
        for (i, (shape, (count, example))) in sorted_shapes.iter().take(20).enumerate() {
            let pct = (*count as f64 / total as f64) * 100.0;
            let markers = format!(
                "{}{}",
                if shape_has_neg_exp(shape) {
                    " [NEG_EXP]"
                } else {
                    ""
                },
                if shape_has_div(shape) { " [DIV]" } else { "" }
            );
            eprintln!(
                "   {:2}. {:5.1}% ({:3}) {}{}",
                i + 1,
                pct,
                count,
                if shape.len() > 60 {
                    &shape[..60]
                } else {
                    shape
                },
                markers
            );
            if i < 5 {
                // Show example for top 5
                eprintln!(
                    "       Example: {}",
                    if example.len() > 80 {
                        &example[..80]
                    } else {
                        example
                    }
                );
            }
        }
        if sorted_shapes.len() > 20 {
            eprintln!("   ... and {} more unique shapes", sorted_shapes.len() - 20);
        }
        eprintln!();
    }
    if verbose && !domain_frontier_examples.is_empty() {
        eprintln!("\n🛡️ Known domain-frontier examples:");
        for (lhs, rhs, reason) in domain_frontier_examples.iter().take(max_examples) {
            eprintln!("  LHS: {}", lhs);
            eprintln!("  RHS: {}", rhs);
            eprintln!("  Reason: {}", reason);
            eprintln!();
        }
    }
    if verbose && !skip_examples.is_empty() {
        eprintln!("\n🧪 Skip/Parse-err diagnostics:");
        for (i, example) in skip_examples.iter().enumerate() {
            eprintln!("   {:2}. {}", i + 1, example);
            eprintln!();
        }
        if skipped > skip_examples.len() {
            eprintln!(
                "   ... and {} more skip cases (set METATEST_MAX_EXAMPLES=N to show more)",
                skipped - skip_examples.len()
            );
            eprintln!();
        }
    }

    // Triple combinations (optional, limited)
    if include_triples && n >= 3 {
        let mut triple_passed = 0;
        let mut triple_failed = 0;
        let mut triple_inconclusive = 0;
        let mut triple_nf = 0;
        let mut triple_proved = 0;
        let mut triple_numeric = 0;
        let triple_limit = 100; // Limit to avoid explosion
        let mut triple_count = 0;

        'outer: for i in 0..n.min(20) {
            for j in (i + 1)..n.min(20) {
                for k in (j + 1)..n.min(20) {
                    if triple_count >= triple_limit {
                        break 'outer;
                    }

                    let pair1 = &pairs[i];
                    let pair2 = &pairs[j];
                    let pair3 = &pairs[k];

                    let mut used_vars: HashSet<String> = pair1.vars.iter().cloned().collect();
                    let (pair2_exp, pair2_simp, pair2_vars, pair2_filters) =
                        rename_identity_for_combination(pair2, &mut used_vars);
                    let (pair3_exp, pair3_simp, pair3_vars, pair3_filters) =
                        rename_identity_for_combination(pair3, &mut used_vars);
                    let mut combined_vars = pair1.vars.clone();
                    combined_vars.extend(pair2_vars);
                    combined_vars.extend(pair3_vars);
                    let mut combined_filters = identity_filters(pair1);
                    combined_filters.extend(pair2_filters);
                    combined_filters.extend(pair3_filters);

                    let combined_exp = format!(
                        "(({}) {} ({})) {} ({})",
                        pair1.exp,
                        op.symbol(),
                        pair2_exp,
                        op.symbol(),
                        pair3_exp
                    );
                    let combined_simp = format!(
                        "(({}) {} ({})) {} ({})",
                        pair1.simp,
                        op.symbol(),
                        pair2_simp,
                        op.symbol(),
                        pair3_simp
                    );

                    let mut simplifier = Simplifier::with_default_rules();
                    let exp_parsed = match parse(&combined_exp, &mut simplifier.context) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };
                    let simp_parsed = match parse(&combined_simp, &mut simplifier.context) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };

                    let (exp_simplified_raw, _) = simplifier.simplify(exp_parsed);
                    let exp_simplified =
                        fold_constants_safe(&mut simplifier.context, exp_simplified_raw);
                    let (simp_simplified_raw, _) = simplifier.simplify(simp_parsed);
                    let simp_simplified =
                        fold_constants_safe(&mut simplifier.context, simp_simplified_raw);

                    if cas_solver::runtime::compare_expr(
                        &simplifier.context,
                        exp_simplified,
                        simp_simplified,
                    ) == std::cmp::Ordering::Equal
                    {
                        triple_nf += 1;
                        triple_passed += 1;
                        triple_count += 1;
                        continue;
                    }

                    if prove_zero_from_metamorphic_texts(
                        &mut simplifier,
                        &combined_exp,
                        &combined_simp,
                        exp_simplified,
                        simp_simplified,
                    ) {
                        triple_proved += 1;
                        triple_passed += 1;
                        triple_count += 1;
                        continue;
                    }

                    match classify_numeric_equiv_for_vars(
                        &simplifier.context,
                        exp_simplified,
                        simp_simplified,
                        &combined_vars,
                        &combined_filters,
                        &config,
                    ) {
                        NumericCheckOutcome::Pass => {
                            triple_numeric += 1;
                            triple_passed += 1;
                        }
                        NumericCheckOutcome::Inconclusive(_) => {
                            triple_inconclusive += 1;
                        }
                        NumericCheckOutcome::Failed(_) => {
                            triple_failed += 1;
                        }
                    }

                    triple_count += 1;
                }
            }
        }

        eprintln!(
            "✅ Triple combinations: {} passed, {} failed, {} inconclusive (of {} tested)",
            triple_passed, triple_failed, triple_inconclusive, triple_count
        );
        eprintln!(
            "   📐 Triple NF-convergent: {} | 🔢 Triple Proved-symbolic: {} | 🌡️ Triple Numeric-only: {}",
            triple_nf, triple_proved, triple_numeric
        );
    }

    ComboMetrics {
        op: op.name().to_string(),
        pairs: n,
        families: num_families,
        combos: passed + failed + skipped + timeouts,
        nf_convergent,
        proved_quotient,
        proved_difference,
        proved_composed,
        numeric_only,
        inconclusive,
        failed,
        skipped,
        timeouts,
        cycle_events_total,
        known_symbolic_residuals: 0,
        numeric_only_causes,
        inconclusive_causes,
        domain_frontier_examples,
    }
}

/// Run automatic double and triple combinations from CSV file
/// This generates thousands of test cases from ~180 identity pairs
#[test]
fn metatest_csv_combinations_small() {
    // Small run: 30 pairs (stratified) = ~351 double combinations
    let m = run_csv_combination_tests(30, false, CombineOp::Add);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

#[test]
#[ignore] // Run with: cargo test --ignored
fn metatest_csv_combinations_full() {
    // Full run: 150 pairs (stratified) with triples. Covers all ~40 CSV families.
    let m = run_csv_combination_tests(150, true, CombineOp::Add);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

/// Multiplicative combination test: (LHS_1 * LHS_2) vs (RHS_1 * RHS_2)
/// Tests distribution, factoring, power simplification paths.
/// Uses stratified sampling: 1 representative per CSV family (~134) + fill to 150.
#[test]
#[ignore]
fn metatest_csv_combinations_mul() {
    // 150 pairs (stratified) ≈ 11,175 combos. 2s per-combo timeout caps cost.
    let m = run_csv_combination_tests(150, false, CombineOp::Mul);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

#[test]
#[ignore]
fn metatest_csv_combinations_mul_nf_first() {
    let m = run_csv_combination_tests_nf_first(150, false, CombineOp::Mul);
    assert_eq!(m.failed, 0, "Some mul nf-first combination tests failed");
}

/// Additive combination test with stratified coverage
/// (LHS_1 + LHS_2) vs (RHS_1 + RHS_2)
#[test]
#[ignore]
fn metatest_csv_combinations_add() {
    // 150 pairs (stratified) ≈ 11,175 combos. Add is fast (≈5s timeout).
    let m = run_csv_combination_tests(150, false, CombineOp::Add);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

#[test]
#[ignore]
fn metatest_csv_combinations_add_nf_first() {
    let m = run_csv_combination_tests_nf_first(30, false, CombineOp::Add);
    assert_eq!(m.failed, 0, "Some add nf-first combination tests failed");
}

/// Subtractive combination test with stratified coverage
/// (LHS_1 - LHS_2) vs (RHS_1 - RHS_2)
/// Tests sign handling, cancellation, and subtraction-specific simplification
#[test]
#[ignore]
fn metatest_csv_combinations_sub() {
    // 150 pairs (stratified) ≈ 11,175 combos. Sub is fast (≈5s timeout).
    let m = run_csv_combination_tests(150, false, CombineOp::Sub);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

/// Division combination test: (LHS_1 / LHS_2) vs (RHS_1 / RHS_2)
/// Tests fraction simplification, quotient cancellation, and cross-multiplication paths.
/// Uses stratified sampling: 1 representative per CSV family (~134) + fill to 50.
/// Includes a divisor safety guard: identities that evaluate near zero are skipped as divisors.
#[test]
#[ignore]
fn metatest_csv_combinations_div() {
    // 50 pairs (stratified) ≈ 1,225 combos. Fewer than Mul due to CAS
    // limitations with high-degree polynomial divisors causing fraction
    // simplification failures. Still covers ~50 families (vs old 15/~12).
    let m = run_csv_combination_tests(50, false, CombineOp::Div);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

#[test]
#[ignore]
fn metatest_csv_combinations_div_nf_first() {
    let m = run_csv_combination_tests_nf_first(50, false, CombineOp::Div);
    assert_eq!(m.failed, 0, "Some div nf-first combination tests failed");
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution -- --include-ignored
fn metatest_csv_substitution() {
    let m = run_substitution_tests();
    assert_eq!(m.failed, 0, "{} substitution tests failed", m.failed);
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_solver --test metamorphic_simplification_tests metatest_csv_substitution_strict -- --ignored --exact --nocapture
fn metatest_csv_substitution_strict() {
    let m = run_substitution_tests_strict();
    assert_eq!(m.failed, 0, "{} strict substitution tests failed", m.failed);
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_solver --test metamorphic_simplification_tests metatest_csv_substitution_nf_first -- --ignored --exact --nocapture
fn metatest_csv_substitution_nf_first() {
    let m = run_substitution_tests_nf_first();
    assert_eq!(
        m.failed, 0,
        "{} nf-first substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution_structural -- --ignored --exact --nocapture
fn metatest_csv_substitution_structural() {
    let m = run_structural_substitution_tests();
    assert_eq!(
        m.failed, 0,
        "{} structural substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution_structural_raw -- --ignored --exact --nocapture
fn metatest_csv_substitution_structural_raw() {
    let m = run_structural_substitution_tests_raw();
    assert_eq!(
        m.failed, 0,
        "{} raw structural substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_solver --test metamorphic_simplification_tests metatest_csv_substitution_structural_nf_first -- --ignored --exact --nocapture
fn metatest_csv_substitution_structural_nf_first() {
    let m = run_structural_substitution_tests_nf_first();
    assert_eq!(
        m.failed, 0,
        "{} nf-first structural substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution_structural_phase -- --include-ignored
fn metatest_csv_substitution_structural_phase() {
    let m = run_structural_phase_substitution_tests();
    assert_eq!(
        m.failed, 0,
        "{} structural phase substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution_structural_radical -- --include-ignored
fn metatest_csv_substitution_structural_radical() {
    let m = run_structural_radical_substitution_tests();
    assert_eq!(
        m.failed, 0,
        "{} structural radical substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution_structural_composed -- --include-ignored
fn metatest_csv_substitution_structural_composed() {
    let m = run_structural_composed_substitution_tests();
    assert_eq!(
        m.failed, 0,
        "{} structural composed substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution_structural_root_ctx -- --include-ignored
fn metatest_csv_substitution_structural_root_ctx() {
    let m = run_structural_root_ctx_substitution_tests();
    assert_eq!(
        m.failed, 0,
        "{} structural root-ctx substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution_structural_poly_high -- --include-ignored
fn metatest_csv_substitution_structural_poly_high() {
    let m = run_structural_poly_high_substitution_tests();
    assert_eq!(
        m.failed, 0,
        "{} structural poly-high substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution_structural_rational_ctx -- --include-ignored
fn metatest_csv_substitution_structural_rational_ctx() {
    let m = run_structural_rational_ctx_substitution_tests();
    assert_eq!(
        m.failed, 0,
        "{} structural rational-ctx substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution_structural_absolute -- --include-ignored
fn metatest_csv_substitution_structural_absolute() {
    let m = run_structural_absolute_substitution_tests();
    assert_eq!(
        m.failed, 0,
        "{} structural absolute substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution_structural_rational -- --include-ignored
fn metatest_csv_substitution_structural_rational() {
    let m = run_structural_rational_substitution_tests();
    assert_eq!(
        m.failed, 0,
        "{} structural rational substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution_structural_inv_trig -- --include-ignored
fn metatest_csv_substitution_structural_inv_trig() {
    let m = run_structural_inv_trig_substitution_tests();
    assert_eq!(
        m.failed, 0,
        "{} structural inv-trig substitution tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_contextual_pairs -- --ignored --nocapture
fn metatest_csv_contextual_pairs() {
    let m = run_contextual_pair_tests();
    assert_eq!(
        m.failed, 0,
        "{} contextual metamorphic tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_solver --test metamorphic_simplification_tests metatest_csv_contextual_pairs_strict -- --ignored --exact --nocapture
fn metatest_csv_contextual_pairs_strict() {
    let m = run_contextual_pair_tests_strict();
    assert_eq!(m.failed, 0, "{} strict contextual tests failed", m.failed);
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_contextual_rational_pairs -- --ignored --nocapture
fn metatest_csv_contextual_rational_pairs() {
    let m = run_contextual_rational_pair_tests();
    assert_eq!(m.failed, 0, "{} contextual rational tests failed", m.failed);
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_contextual_trig_pairs -- --ignored --nocapture
fn metatest_csv_contextual_trig_pairs() {
    let m = run_contextual_trig_pair_tests();
    assert_eq!(m.failed, 0, "{} contextual trig tests failed", m.failed);
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_contextual_polynomial_pairs -- --ignored --nocapture
fn metatest_csv_contextual_polynomial_pairs() {
    let m = run_contextual_polynomial_pair_tests();
    assert_eq!(
        m.failed, 0,
        "{} contextual polynomial tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_contextual_radical_pairs -- --ignored --nocapture
fn metatest_csv_contextual_radical_pairs() {
    let m = run_contextual_radical_pair_tests();
    assert_eq!(m.failed, 0, "{} contextual radical tests failed", m.failed);
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_residual_pairs -- --ignored --nocapture
fn metatest_csv_residual_pairs() {
    let m = run_residual_pair_tests();
    assert_eq!(
        m.failed, 0,
        "{} residual metamorphic tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_known_domain_frontier_pairs -- --ignored --nocapture
fn metatest_csv_known_domain_frontier_pairs() {
    let m = run_known_domain_frontier_pair_tests();
    assert_eq!(
        m.failed, 0,
        "{} known domain-frontier metamorphic tests failed",
        m.failed
    );
    assert_eq!(
        m.numeric_only, 0,
        "{} known domain-frontier pairs leaked into numeric-only",
        m.numeric_only
    );
    assert_eq!(
        m.timeouts, 0,
        "{} known domain-frontier pairs timed out",
        m.timeouts
    );
    assert_eq!(
        m.inconclusive,
        m.known_domain_frontier_count(),
        "known domain-frontier suite should only report domain-frontier inconclusives"
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_known_domain_frontier_safe_pairs -- --ignored --nocapture
fn metatest_csv_known_domain_frontier_safe_pairs() {
    let m = run_known_domain_frontier_safe_pair_tests();
    let pair_count = load_known_domain_frontier_safe_pairs().len();
    assert_eq!(
        m.failed, 0,
        "{} known domain-frontier safe-window tests failed",
        m.failed
    );
    assert_eq!(
        m.inconclusive, 0,
        "{} known domain-frontier safe-window pairs remained inconclusive",
        m.inconclusive
    );
    assert_eq!(
        m.timeouts, 0,
        "{} known domain-frontier safe-window pairs timed out",
        m.timeouts
    );
    assert_eq!(
        m.nf_convergent + m.proved_symbolic(),
        pair_count,
        "known domain-frontier safe-window suite should close all parametrized symbolic cases"
    );
    assert_eq!(
        m.numeric_only, 0,
        "{} known domain-frontier safe-window pairs leaked into numeric-only",
        m.numeric_only
    );
}

#[test]
fn known_domain_frontier_catalog_covers_all_csv_pairs() {
    let pairs = load_known_domain_frontier_pairs();
    assert_eq!(pairs.len(), 8, "unexpected known domain-frontier CSV size");

    for pair in &pairs {
        let reason = known_domain_frontier_reason(&pair.lhs, &pair.rhs);
        assert!(
            reason.is_some(),
            "known domain-frontier CSV pair is missing from classifier: {} ↔ {}",
            pair.lhs,
            pair.rhs
        );
    }
}

#[test]
fn known_domain_frontier_csv_breakdown_matches_expected_reason_counts() {
    let pairs = load_known_domain_frontier_pairs();
    let mut counts: HashMap<&'static str, usize> = HashMap::new();

    for pair in &pairs {
        let reason = known_domain_frontier_reason(&pair.lhs, &pair.rhs)
            .expect("known domain-frontier CSV pair should be classified");
        *counts.entry(reason).or_default() += 1;
    }

    assert_eq!(
        counts.get("log-square expansion changes domain").copied(),
        Some(3)
    );
    assert_eq!(
        counts
            .get("inverse-trig branch introduces domain/branch sensitivity")
            .copied(),
        Some(3)
    );
    assert_eq!(
        counts
            .get("sqrt product contraction changes sign/domain behavior")
            .copied(),
        Some(2)
    );
    assert_eq!(counts.values().sum::<usize>(), 8);
}

#[test]
fn known_domain_frontier_safe_catalog_covers_all_csv_pairs() {
    let pairs = load_known_domain_frontier_safe_pairs();
    assert_eq!(
        pairs.len(),
        8,
        "unexpected known domain-frontier safe CSV size"
    );

    for pair in &pairs {
        let reason = known_domain_frontier_reason(&pair.lhs, &pair.rhs);
        assert!(
            reason.is_some(),
            "known domain-frontier safe CSV pair is missing from classifier: {} ↔ {}",
            pair.lhs,
            pair.rhs
        );
    }
}

#[test]
fn known_domain_frontier_safe_csv_mirrors_primary_pair_set() {
    let primary = load_known_domain_frontier_pairs();
    let safe = load_known_domain_frontier_safe_pairs();

    let normalize_pair = |pair: &ContextualPair| {
        let lhs = normalize_metamorphic_text(&pair.lhs);
        let rhs = normalize_metamorphic_text(&pair.rhs);
        if lhs <= rhs {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        }
    };

    let primary_set: std::collections::HashSet<_> = primary.iter().map(normalize_pair).collect();
    let safe_set: std::collections::HashSet<_> = safe.iter().map(normalize_pair).collect();

    assert_eq!(
        primary_set, safe_set,
        "known domain-frontier safe CSV should mirror the same pair set as the primary frontier CSV"
    );
}

#[test]
fn known_domain_frontier_safe_csv_breakdown_matches_expected_reason_counts() {
    let pairs = load_known_domain_frontier_safe_pairs();
    let mut counts: HashMap<&'static str, usize> = HashMap::new();

    for pair in &pairs {
        let reason = known_domain_frontier_reason(&pair.lhs, &pair.rhs)
            .expect("known domain-frontier safe CSV pair should be classified");
        *counts.entry(reason).or_default() += 1;
    }

    assert_eq!(
        counts.get("log-square expansion changes domain").copied(),
        Some(3)
    );
    assert_eq!(
        counts
            .get("inverse-trig branch introduces domain/branch sensitivity")
            .copied(),
        Some(3)
    );
    assert_eq!(
        counts
            .get("sqrt product contraction changes sign/domain behavior")
            .copied(),
        Some(2)
    );
    assert_eq!(counts.values().sum::<usize>(), 8);
}

#[test]
fn known_domain_frontier_safe_csv_declares_effective_filters() {
    let pairs = load_known_domain_frontier_safe_pairs();

    for pair in &pairs {
        assert!(
            pair.filters.iter().any(|f| !f.is_none()),
            "known domain-frontier safe CSV pair should declare at least one effective filter: {} ↔ {}",
            pair.lhs,
            pair.rhs
        );
    }
}

#[test]
fn safe_window_parametrized_catalog_covers_all_safe_csv_pairs() {
    let pairs = load_known_domain_frontier_safe_pairs();
    assert_eq!(
        pairs.len(),
        8,
        "unexpected known domain-frontier safe CSV size"
    );

    for pair in &pairs {
        assert!(
            safe_window_parametrized_pair_texts(&pair.lhs, &pair.rhs).is_some(),
            "known domain-frontier safe CSV pair is missing from parametrized proof catalog: {} ↔ {}",
            pair.lhs,
            pair.rhs
        );
    }
}

#[test]
fn known_domain_frontier_primary_pairs_all_have_safe_window_symbolic_mirror() {
    let pairs = load_known_domain_frontier_pairs();

    for pair in &pairs {
        assert!(
            safe_window_parametrized_pair_texts(&pair.lhs, &pair.rhs).is_some(),
            "primary known domain-frontier pair is missing a safe-window parametrization: {} ↔ {}",
            pair.lhs,
            pair.rhs
        );
        assert!(
            prove_zero_from_safe_window_parametrized_texts(&pair.lhs, &pair.rhs),
            "primary known domain-frontier pair is missing a working safe-window symbolic closure: {} ↔ {}",
            pair.lhs,
            pair.rhs
        );
    }
}

#[test]
fn safe_window_parametrized_catalog_closes_all_safe_csv_pairs() {
    let pairs = load_known_domain_frontier_safe_pairs();

    for pair in &pairs {
        assert!(
            prove_zero_from_safe_window_parametrized_texts(&pair.lhs, &pair.rhs),
            "known domain-frontier safe CSV pair did not close through the parametrized proof path: {} ↔ {}",
            pair.lhs,
            pair.rhs
        );
    }
}
