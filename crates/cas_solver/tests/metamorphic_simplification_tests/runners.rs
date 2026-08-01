//! `metamorphic_simplification_tests`: familia `runners`.
//!
//! Ver la cabecera de `metamorphic_simplification_tests.rs` para el contexto.

use super::*;

/// Append a log entry to the metatest log file (JSON Lines format)
fn log_metatest_run(
    test_name: &str,
    config: &MetatestConfig,
    passed: usize,
    failed: usize,
    skipped: usize,
) {
    // Get timestamp
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let stress = env::var("METATEST_STRESS").ok().as_deref() == Some("1");

    // Build JSON entry
    let entry = format!(
        r#"{{"timestamp":{},"test":"{}","seed":{},"samples":{},"depth":{},"min_valid":{},"stress":{},"passed":{},"failed":{},"skipped":{}}}"#,
        timestamp,
        test_name,
        config.seed,
        config.samples,
        config.depth,
        config.min_valid,
        stress,
        passed,
        failed,
        skipped
    );

    // Append to log file
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file_path())
    {
        let _ = writeln!(file, "{}", entry);
    }
}

pub(super) fn shuffle_expr_seeded(ctx: &mut Context, expr: ExprId, seed: u64) -> ExprId {
    match ctx.nodes.get(expr.index()).cloned() {
        Some(cas_ast::Expr::Add(_, _)) => {
            // Flatten, shuffle, rebuild
            let mut terms = collect_addends(ctx, expr);
            if terms.len() > 1 {
                // Shuffle terms
                shuffle_vec(&mut terms, seed);
                // Recursively shuffle each term
                let shuffled_terms: Vec<_> = terms
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| shuffle_expr_seeded(ctx, t, seed.wrapping_add(i as u64)))
                    .collect();
                rebuild_add(ctx, &shuffled_terms)
            } else {
                expr
            }
        }
        Some(cas_ast::Expr::Mul(_, _)) => {
            // Flatten, shuffle, rebuild
            let mut factors = collect_factors(ctx, expr);
            if factors.len() > 1 {
                shuffle_vec(&mut factors, seed.wrapping_add(1000));
                let shuffled_factors: Vec<_> = factors
                    .iter()
                    .enumerate()
                    .map(|(i, &f)| shuffle_expr_seeded(ctx, f, seed.wrapping_add(2000 + i as u64)))
                    .collect();
                rebuild_mul(ctx, &shuffled_factors)
            } else {
                expr
            }
        }
        Some(cas_ast::Expr::Pow(base, exp)) => {
            // Don't shuffle base/exp order, just recurse
            let new_base = shuffle_expr_seeded(ctx, base, seed.wrapping_add(100));
            let new_exp = shuffle_expr_seeded(ctx, exp, seed.wrapping_add(200));
            ctx.add_raw(cas_ast::Expr::Pow(new_base, new_exp))
        }
        Some(cas_ast::Expr::Function(name, args)) => {
            // Recurse into args (don't reorder - function args aren't commutative)
            let new_args: Vec<_> = args
                .iter()
                .enumerate()
                .map(|(i, &a)| shuffle_expr_seeded(ctx, a, seed.wrapping_add(300 + i as u64)))
                .collect();
            ctx.add_raw(cas_ast::Expr::Function(name, new_args))
        }
        Some(cas_ast::Expr::Neg(inner)) => {
            let new_inner = shuffle_expr_seeded(ctx, inner, seed.wrapping_add(400));
            ctx.add_raw(cas_ast::Expr::Neg(new_inner))
        }
        Some(cas_ast::Expr::Sub(a, b)) => {
            // Sub is not commutative - just recurse
            let new_a = shuffle_expr_seeded(ctx, a, seed.wrapping_add(500));
            let new_b = shuffle_expr_seeded(ctx, b, seed.wrapping_add(600));
            ctx.add_raw(cas_ast::Expr::Sub(new_a, new_b))
        }
        Some(cas_ast::Expr::Div(a, b)) => {
            // Div is not commutative - just recurse
            let new_a = shuffle_expr_seeded(ctx, a, seed.wrapping_add(700));
            let new_b = shuffle_expr_seeded(ctx, b, seed.wrapping_add(800));
            ctx.add_raw(cas_ast::Expr::Div(new_a, new_b))
        }
        // Leaf nodes - no change
        _ => expr,
    }
}

/// Filter for [`metatest_child_raw_pressure_proof`], captured where the test
/// lives so moving the test moves the filter. See `ChildTest` in `main.rs`.
pub(super) const CHILD_RAW_PRESSURE_PROOF: ChildTest =
    ChildTest::here(module_path!(), "metatest_child_raw_pressure_proof");

#[test]
#[ignore]
pub(super) fn metatest_child_raw_pressure_proof() {
    let lhs =
        std::env::var(METATEST_CHILD_RAW_PROOF_LHS_ENV).expect("missing child raw-proof lhs env");
    let rhs =
        std::env::var(METATEST_CHILD_RAW_PROOF_RHS_ENV).expect("missing child raw-proof rhs env");
    let handle = std::thread::Builder::new()
        .stack_size(METATEST_DEEP_WORKER_STACK_SIZE_BYTES)
        .spawn(move || {
            assert!(
                prove_zero_from_engine_texts_child_hint(&lhs, &rhs),
                "child raw-pressure proof failed"
            );
        })
        .expect("spawn raw pressure child worker");
    handle.join().expect("raw pressure child worker panicked");
}

/// Filter for [`metatest_child_nf_convergence`] — see [`CHILD_RAW_PRESSURE_PROOF`].
pub(super) const CHILD_NF_CONVERGENCE: ChildTest =
    ChildTest::here(module_path!(), "metatest_child_nf_convergence");

#[test]
#[ignore]
pub(super) fn metatest_child_nf_convergence() {
    let lhs = std::env::var(METATEST_CHILD_NF_LHS_ENV).expect("missing child nf lhs env");
    let rhs = std::env::var(METATEST_CHILD_NF_RHS_ENV).expect("missing child nf rhs env");

    let handle = std::thread::Builder::new()
        .stack_size(METATEST_DEEP_WORKER_STACK_SIZE_BYTES)
        .spawn(move || {
            let mut engine = Engine::new();
            let simplifier = &mut engine.simplifier;
            let lhs_parsed = parse(&lhs, &mut simplifier.context).expect("nf child lhs parse");
            let rhs_parsed = parse(&rhs, &mut simplifier.context).expect("nf child rhs parse");
            let opts = cas_solver::runtime::SimplifyOptions::default();

            let (mut lhs_simp, _, _) = simplifier.simplify_with_stats(lhs_parsed, opts.clone());
            let (mut rhs_simp, _, _) = simplifier.simplify_with_stats(rhs_parsed, opts);
            lhs_simp = fold_constants_safe(&mut simplifier.context, lhs_simp);
            rhs_simp = fold_constants_safe(&mut simplifier.context, rhs_simp);

            assert!(
                normal_forms_visibly_equal(&simplifier.context, lhs_simp, rhs_simp),
                "child nf convergence failed"
            );
        })
        .expect("spawn nf child worker");
    handle.join().expect("nf child worker panicked");
}

fn fresh_combination_vars(count: usize, used: &mut HashSet<String>) -> Vec<String> {
    const POOL: [&str; 12] = ["u", "v", "w", "p", "q", "r", "s", "t", "m", "n", "i", "j"];

    let mut vars = Vec::with_capacity(count);
    let mut next_suffix = 0usize;

    while vars.len() < count {
        let candidate = if next_suffix < POOL.len() {
            POOL[next_suffix].to_string()
        } else {
            format!("u{}", next_suffix - POOL.len())
        };
        next_suffix += 1;
        if used.insert(candidate.clone()) {
            vars.push(candidate);
        }
    }

    vars
}

pub(super) fn rename_identity_for_combination(
    pair: &IdentityPair,
    used: &mut HashSet<String>,
) -> (String, String, Vec<String>, Vec<FilterSpec>) {
    let renamed_vars = fresh_combination_vars(pair.vars.len(), used);
    let renames: Vec<(String, String)> = pair
        .vars
        .iter()
        .cloned()
        .zip(renamed_vars.iter().cloned())
        .collect();

    (
        alpha_rename_many(&pair.exp, &renames),
        alpha_rename_many(&pair.simp, &renames),
        renamed_vars,
        identity_filters(pair),
    )
}

fn default_combination_timeout(op: CombineOp, debug_build: bool) -> std::time::Duration {
    match op {
        CombineOp::Mul | CombineOp::Div if !debug_build => std::time::Duration::from_secs(2),
        _ => std::time::Duration::from_secs(5),
    }
}

#[test]
fn multiplicative_combination_timeout_policy_is_tighter_in_release() {
    assert_eq!(
        default_combination_timeout(CombineOp::Mul, false),
        std::time::Duration::from_secs(2)
    );
    assert_eq!(
        default_combination_timeout(CombineOp::Div, false),
        std::time::Duration::from_secs(2)
    );
    assert_eq!(
        default_combination_timeout(CombineOp::Add, false),
        std::time::Duration::from_secs(5)
    );
    assert_eq!(
        default_combination_timeout(CombineOp::Mul, true),
        std::time::Duration::from_secs(5)
    );
}

pub(super) fn combination_timeout(op: CombineOp) -> std::time::Duration {
    std::env::var("METATEST_COMBO_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        .map(std::time::Duration::from_millis)
        .unwrap_or_else(|| default_combination_timeout(op, cfg!(debug_assertions)))
}

/// UNIFIED BENCHMARK: run all 4 operations and print a regression/improvement table.
///
/// This test does NOT assert on failures — it prints metrics for comparison.
/// Use it as a diagnostic benchmark before/after rule changes:
///
/// ```text
/// cargo test --release -p cas_engine --test metamorphic_simplification_tests \
///     -- metatest_benchmark_all_ops --ignored --nocapture
/// ```
///
/// Key metrics:
/// - NF-convergent: simplify(LHS) ≡ simplify(RHS) structurally (ideal)
/// - Proved-symbolic: simplify(LHS - RHS) = 0 (correct but different NFs)
/// - Numeric-only: only passes numeric check (target for improvement)
/// - Failed: semantic mismatches (regressions)
#[test]
#[ignore]
fn metatest_benchmark_all_ops() {
    // Pair counts per operation (stratified sampling)
    let configs: Vec<(CombineOp, usize)> = vec![
        (CombineOp::Add, 150),
        (CombineOp::Sub, 150),
        (CombineOp::Mul, 150),
        (CombineOp::Div, 50),
    ];

    let mut all_metrics: Vec<ComboMetrics> = Vec::new();
    let mut total_failed = 0;

    for (op, pairs) in &configs {
        // Run without internal assert — collect metrics only
        let metrics = run_csv_combination_tests(*pairs, false, *op);
        total_failed += metrics.failed;
        all_metrics.push(metrics);
    }

    // Print unified benchmark table
    eprintln!();
    eprintln!(
        "╔═══════════════════════════════════════════════════════════════════════════════════╗"
    );
    eprintln!(
        "║                     METAMORPHIC BENCHMARK RESULTS                                ║"
    );
    eprintln!(
        "╠═════╤════════╤══════════╤══════════════╤════════════════╤══════════════╤══════════╣"
    );
    eprintln!(
        "║ Op  │ Pairs  │ Families │ NF-convergent│ Proved-sym (Q+D)│ Numeric-only │ Failed   ║"
    );
    eprintln!(
        "╠═════╪════════╪══════════╪══════════════╪═════════════════╪══════════════╪══════════╣"
    );

    let mut total_nf = 0;
    let mut total_proved = 0;
    let mut total_numeric = 0;
    let mut total_combos = 0;
    let mut total_skipped = 0;
    let mut total_f = 0;

    for m in &all_metrics {
        let effective = m.combos - m.skipped;
        let proved = m.proved_symbolic();
        eprintln!(
            "║ {:<3} │ {:>5}  │ {:>7}  │ {:>6} {:>5.1}% │{:>4}+{:>4}+{:>4}{:>5.1}% │ {:>6} {:>5.1}% │ {:>6}   ║",
            m.op, m.pairs, m.families,
            m.nf_convergent,
            if effective > 0 { m.nf_convergent as f64 / effective as f64 * 100.0 } else { 0.0 },
            m.proved_quotient,
            m.proved_difference,
            m.proved_composed,
            if effective > 0 { proved as f64 / effective as f64 * 100.0 } else { 0.0 },
            m.numeric_only,
            if effective > 0 { m.numeric_only as f64 / effective as f64 * 100.0 } else { 0.0 },
            m.failed,
        );
        total_nf += m.nf_convergent;
        total_proved += proved;
        total_numeric += m.numeric_only;
        total_combos += m.combos;
        total_skipped += m.skipped;
        total_f += m.failed;
    }

    let total_effective = total_combos - total_skipped;
    eprintln!(
        "╠═════╪════════╪══════════╪══════════════╪═════════════════╪══════════════╪══════════╣"
    );
    eprintln!(
        "║ ALL │        │          │ {:>6} {:>5.1}% │     {:>5}{:>5.1}% │ {:>6} {:>5.1}% │ {:>6}   ║",
        total_nf,
        if total_effective > 0 {
            total_nf as f64 / total_effective as f64 * 100.0
        } else {
            0.0
        },
        total_proved,
        if total_effective > 0 {
            total_proved as f64 / total_effective as f64 * 100.0
        } else {
            0.0
        },
        total_numeric,
        if total_effective > 0 {
            total_numeric as f64 / total_effective as f64 * 100.0
        } else {
            0.0
        },
        total_f,
    );
    eprintln!(
        "╚═════╧════════╧══════════╧══════════════╧═════════════════╧══════════════╧══════════╝"
    );
    eprintln!(
        "   Total combos: {} (skipped: {})",
        total_combos, total_skipped
    );
    eprintln!();

    if total_failed > 0 {
        eprintln!(
            "⚠️  {} semantic failures detected — investigate before merging.",
            total_failed
        );
    }
}

/// Test individual identity pairs (not combinations) to see which simplify symbolically
///
/// Environment variables:
/// - METATEST_MODE=assume : Use DomainMode::Assume (includes all identities)
/// - METATEST_MODE=generic (or unset) : Use DomainMode::Generic (skips assume-only identities)
#[test]
#[ignore = "Diagnostic test - run manually to check symbolic vs numeric equivalence"]
fn metatest_individual_identities() {
    // Run in a thread with larger stack to avoid overflow
    let handle = std::thread::Builder::new()
        .stack_size(METATEST_WORKER_STACK_SIZE_BYTES)
        .spawn(metatest_individual_identities_impl)
        .expect("Failed to spawn test thread");
    handle.join().expect("Test thread panicked");
}

fn metatest_individual_identities_impl() {
    let pairs = load_identity_pairs();
    let config = metatest_config();

    // Determine test mode from environment
    let use_assume_mode = env::var("METATEST_MODE").ok().as_deref() == Some("assume");
    let domain_mode = if use_assume_mode {
        cas_solver::runtime::DomainMode::Assume
    } else {
        cas_solver::runtime::DomainMode::Generic
    };

    eprintln!(
        "🔧 Running in {} mode",
        if use_assume_mode { "ASSUME" } else { "GENERIC" }
    );

    let mut symbolic_passed = 0;
    let mut numeric_only_passed = 0;
    let mut failed = 0;
    let mut skipped = 0;
    let mut numeric_only_examples: Vec<String> = Vec::new();
    #[allow(unused_mut, unused_variables)]
    let mut fragile_identities: Vec<String> = Vec::new(); // High near_pole/domain rate
    #[allow(unused_mut, unused_variables)]
    let mut asymmetric_count = 0; // Suspicious L=Ok/R=Err cases

    // Diagnostics: track per-identity fragility (infrastructure for future use)
    #[allow(dead_code)]
    struct IdentityDiag {
        idx: usize,
        exp: String,
        simp: String,
        bucket: Bucket,
        stats: NumericEquivStats,
        filter_str: String, // Original filter spec from CSV (empty if None)
    }

    // Only collect diagnostics if METATEST_DIAG=1
    let diag_enabled = env::var("METATEST_DIAG").is_ok();
    let mut diagnostics: Vec<IdentityDiag> = Vec::new();

    // Snapshot/baseline mode detection
    let snapshot_enabled = env::var("METATEST_SNAPSHOT").is_ok();
    let update_baseline = env::var("METATEST_UPDATE_BASELINE").is_ok();
    let mut snapshots: Vec<(IdentityPair, NumericEquivStats)> = Vec::new();

    for pair in &pairs {
        // Skip assume-only identities in generic mode
        if pair.mode == DomainRequirement::Assume && !use_assume_mode {
            skipped += 1;
            continue;
        }

        let mut simplifier = Simplifier::with_default_rules();

        // Parse both expressions
        let exp_parsed = match parse(&pair.exp, &mut simplifier.context) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let simp_parsed = match parse(&pair.simp, &mut simplifier.context) {
            Ok(e) => e,
            Err(_) => continue,
        };
        // Check symbolic equality using bucket-aware equivalence
        let sym_result = check_symbolic_equiv_bucket_aware(
            &mut simplifier,
            exp_parsed,
            simp_parsed,
            pair.bucket,
        );

        // Simplify for display and numeric fallback
        let opts = cas_solver::runtime::SimplifyOptions {
            shared: cas_solver::runtime::SharedSemanticConfig {
                semantics: cas_solver::runtime::EvalConfig {
                    domain_mode,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let (exp_simplified, _) = simplifier.simplify_with_options(exp_parsed, opts.clone());
        let (simp_simplified, _) = simplifier.simplify_with_options(simp_parsed, opts);

        let symbolic_match = matches!(
            sym_result,
            SymbolicResult::Pass | SymbolicResult::PassConditional(_)
        );

        if symbolic_match {
            symbolic_passed += 1;
        } else {
            // Get string representations of simplified forms using Debug
            let exp_simplified_str = format!("{:?}", simplifier.context.get(exp_simplified));
            let simp_simplified_str = format!("{:?}", simplifier.context.get(simp_simplified));

            // Check numeric equivalence - select function based on variable count
            let (result, stats_opt) = match pair.vars.len() {
                1 => {
                    if diag_enabled {
                        let stats = check_numeric_equiv_1var_stats(
                            &simplifier.context,
                            exp_simplified,
                            simp_simplified,
                            &pair.vars[0],
                            &config,
                            &pair.filter_spec,
                        );
                        let pass = stats.valid >= config.min_valid && stats.mismatches.is_empty();
                        (
                            if pass {
                                Ok(stats.valid)
                            } else {
                                Err("failed".to_string())
                            },
                            Some(stats),
                        )
                    } else {
                        (
                            check_numeric_equiv_1var(
                                &simplifier.context,
                                exp_simplified,
                                simp_simplified,
                                &pair.vars[0],
                                &config,
                            ),
                            None,
                        )
                    }
                }
                2 => (
                    check_numeric_equiv_2var(
                        &simplifier.context,
                        exp_simplified,
                        simp_simplified,
                        &pair.vars[0],
                        &pair.vars[1],
                        &config,
                        &pair.filter_spec,
                        &FilterSpec::None,
                    ),
                    None,
                ),
                _ => {
                    // 3+ variables: skip for now
                    skipped += 1;
                    continue;
                }
            };

            // Collect stats for diagnostics if enabled
            if let Some(stats) = stats_opt {
                diagnostics.push(IdentityDiag {
                    idx: diagnostics.len(),
                    exp: pair.exp.clone(),
                    simp: pair.simp.clone(),
                    bucket: pair.bucket,
                    stats: stats.clone(),
                    filter_str: pair.filter_spec.as_str(),
                });

                // Collect snapshot for baseline comparison
                if snapshot_enabled || update_baseline {
                    snapshots.push((pair.clone(), stats));
                }
            }

            if result.is_ok() {
                numeric_only_passed += 1;
                if numeric_only_examples.len() < 30 {
                    numeric_only_examples.push(format!(
                        "{} ≡ {}\n     → L: {}\n     → R: {}",
                        pair.exp, pair.simp, exp_simplified_str, simp_simplified_str
                    ));
                }
            } else {
                failed += 1;
                if failed <= 10 {
                    eprintln!("❌ Identity failed: {} ≡ {}", pair.exp, pair.simp);
                    eprintln!("   → L simplified: {}", exp_simplified_str);
                    eprintln!("   → R simplified: {}", simp_simplified_str);
                }
            }
        }
    }

    let total = symbolic_passed + numeric_only_passed + failed;
    let symbolic_pct = if total > 0 {
        (symbolic_passed as f64 / total as f64 * 100.0) as u32
    } else {
        0
    };

    eprintln!("\n📊 Individual Identity Results:");
    eprintln!("   Total tested: {}", total);
    eprintln!("   ✅ Symbolic: {} ({}%)", symbolic_passed, symbolic_pct);
    eprintln!("   🔢 Numeric-only: {}", numeric_only_passed);
    eprintln!("   ❌ Failed: {}", failed);
    eprintln!("   ⏭️  Skipped: {}", skipped);

    // Top-10 fragility ranking (only when METATEST_DIAG=1)
    if diag_enabled && !diagnostics.is_empty() {
        // Classify all diagnostics
        let classified: Vec<_> = diagnostics
            .iter()
            .map(|d| (classify_diagnostic(&d.stats), d))
            .collect();

        // Count by category
        let bug_count = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::BugSignal)
            .count();
        let config_count = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::ConfigError)
            .count();
        let filter_count = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::NeedsFilter)
            .count();
        let fragile_count = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::Fragile)
            .count();
        let ok_count = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::Ok)
            .count();

        eprintln!("\n📊 Diagnostic Classification (METATEST_DIAG=1):");
        eprintln!(
            "   Summary: ✅ Ok={} | 🐛 BugSignal={} | ⚙️ ConfigError={} | 🔧 NeedsFilter={} | ⚠️ Fragile={}\n",
            ok_count, bug_count, config_count, filter_count, fragile_count
        );

        // Filter Coverage Report
        let with_filter: Vec<_> = diagnostics
            .iter()
            .filter(|d| !d.filter_str.is_empty())
            .collect();
        let filtered_count = with_filter.len();
        let total_diag = diagnostics.len();

        if filtered_count > 0 {
            // Dual coverage: snapshot vs total loaded
            let total_loaded = pairs.len();
            eprintln!(
                "🔍 Filter Coverage: {}/{} snapshot ({:.1}%) | {}/{} total loaded ({:.1}%)",
                filtered_count,
                total_diag,
                filtered_count as f64 / total_diag as f64 * 100.0,
                filtered_count,
                total_loaded,
                filtered_count as f64 / total_loaded as f64 * 100.0
            );

            // Sort by filtered_rate DESC (potential "cheating" filters)
            let mut by_filtered: Vec<_> = diagnostics
                .iter()
                .filter(|d| !d.filter_str.is_empty())
                .map(|d| {
                    let total = d.stats.total_samples();
                    let filtered_rate = if total > 0 {
                        d.stats.filtered_out as f64 / total as f64
                    } else {
                        0.0
                    };
                    (filtered_rate, d)
                })
                .collect();
            by_filtered.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            eprintln!("   Top-5 by filtered_rate (potential 'cheating' filters):");
            for (i, (rate, d)) in by_filtered.iter().take(5).enumerate() {
                eprintln!(
                    "   {:2}. [{:4.0}%] valid={:3}/{:3} {} → {}",
                    i + 1,
                    rate * 100.0,
                    d.stats.valid,
                    d.stats.total_samples(),
                    d.filter_str,
                    truncate_identity(&d.exp, 35)
                );
            }
            eprintln!();
        }
        // Helper to print a ranking section
        let print_ranking = |category: DiagCategory, items: Vec<&IdentityDiag>, max_show: usize| {
            if items.is_empty() {
                return;
            }
            eprintln!(
                "   {} {} ({})",
                category.emoji(),
                category.name(),
                items.len()
            );
            for (i, d) in items.iter().take(max_show).enumerate() {
                let total = d.stats.total_samples();
                eprintln!(
                    "      {:2}. valid={:3}/{:3} pole={:.0}% domain={:.0}% eval={:.0}% asym={}",
                    i + 1,
                    d.stats.valid,
                    total,
                    d.stats.pole_rate() * 100.0,
                    d.stats.domain_rate() * 100.0,
                    d.stats.eval_failed_rate() * 100.0,
                    d.stats.asymmetric_invalid,
                );
                eprintln!("          {} ≡ {}", d.exp, d.simp);
            }
            if items.len() > max_show {
                eprintln!("          ... and {} more", items.len() - max_show);
            }
            eprintln!();
        };

        // 1. BugSignal ranking (sorted by asymmetric_invalid DESC)
        let mut bug_items: Vec<_> = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::BugSignal)
            .map(|(_, d)| *d)
            .collect();
        bug_items.sort_by(|a, b| {
            b.stats
                .asymmetric_invalid
                .cmp(&a.stats.asymmetric_invalid)
                .then_with(|| a.idx.cmp(&b.idx))
        });
        print_ranking(DiagCategory::BugSignal, bug_items, 10);

        // 2. ConfigError ranking (sorted by eval_failed_rate DESC)
        let mut config_items: Vec<_> = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::ConfigError)
            .map(|(_, d)| *d)
            .collect();
        config_items.sort_by(|a, b| {
            b.stats
                .eval_failed_rate()
                .partial_cmp(&a.stats.eval_failed_rate())
                .unwrap()
                .then_with(|| a.idx.cmp(&b.idx))
        });
        print_ranking(DiagCategory::ConfigError, config_items, 5);

        // 3. NeedsFilter ranking (sorted by domain_rate DESC)
        let mut filter_items: Vec<_> = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::NeedsFilter)
            .map(|(_, d)| *d)
            .collect();
        filter_items.sort_by(|a, b| {
            b.stats
                .domain_rate()
                .partial_cmp(&a.stats.domain_rate())
                .unwrap()
                .then_with(|| a.idx.cmp(&b.idx))
        });
        print_ranking(DiagCategory::NeedsFilter, filter_items, 10);

        // 4. Fragile ranking (sorted by pole_rate DESC)
        let mut fragile_items: Vec<_> = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::Fragile)
            .map(|(_, d)| *d)
            .collect();
        fragile_items.sort_by(|a, b| {
            b.stats
                .pole_rate()
                .partial_cmp(&a.stats.pole_rate())
                .unwrap()
                .then_with(|| a.idx.cmp(&b.idx))
        });
        print_ranking(DiagCategory::Fragile, fragile_items, 10);

        // Total asymmetric summary
        let total_asym: usize = diagnostics.iter().map(|d| d.stats.asymmetric_invalid).sum();
        if total_asym > 0 {
            eprintln!("   🚨 Total asymmetric_invalid across all: {}", total_asym);
        }
    }

    // Show fragile identities (high near_pole/domain rate)
    if !fragile_identities.is_empty() {
        eprintln!("\n⚠️  Fragile Identities (>30% near_pole/domain):");
        for (i, id) in fragile_identities.iter().take(10).enumerate() {
            eprintln!("   {}. {}", i + 1, id);
        }
        if fragile_identities.len() > 10 {
            eprintln!("   ... and {} more", fragile_identities.len() - 10);
        }
    }

    // Show asymmetric failures (suspicious - may indicate bugs)
    if asymmetric_count > 0 {
        eprintln!("\n🚨 Asymmetric Failures Detected: {}", asymmetric_count);
        eprintln!("   This may indicate engine bugs (L=Ok but R=Err or vice versa)");
    }

    if !numeric_only_examples.is_empty() {
        eprintln!("\n📝 Numeric-only identities (showing simplifications):");
        for ex in &numeric_only_examples {
            eprintln!("   • {}", ex);
        }
    }

    if failed > 0 {
        eprintln!(
            "\n⚠️  {} identities failed numeric equivalence - may need domain restrictions",
            failed
        );
    }

    // JSONL Baseline Processing
    if snapshot_enabled || update_baseline {
        // Generate current snapshots
        let current_snapshots: Vec<IdentitySnapshot> = snapshots
            .iter()
            .map(|(pair, stats)| {
                let category = classify_diagnostic(stats);
                IdentitySnapshot::from_pair_stats(pair, stats, category)
            })
            .collect();

        let baseline_path = baseline_file_path();

        if update_baseline {
            // Write new baseline
            if let Some(parent) = baseline_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let mut file = File::create(&baseline_path).expect("Failed to create baseline file");
            // Write config header as first line
            writeln!(file, "{}", config_header_json(&config))
                .expect("Failed to write config header");
            for snap in &current_snapshots {
                writeln!(file, "{}", snap.to_json()).expect("Failed to write baseline");
            }
            eprintln!(
                "\n✅ Baseline updated: {} identities + config written to {}",
                current_snapshots.len(),
                baseline_path.display()
            );
            eprintln!("   cfg_hash: {}", generate_config_hash(&config));
        } else if snapshot_enabled {
            // Compare against baseline
            if !baseline_path.exists() {
                eprintln!("\n⚠️  No baseline found at {}", baseline_path.display());
                eprintln!("   Run with METATEST_UPDATE_BASELINE=1 to create one.");
            } else {
                // Load baseline and validate config hash
                let file = File::open(&baseline_path).expect("Failed to open baseline file");
                let reader = BufReader::new(file);
                let lines: Vec<String> = reader.lines().map_while(Result::ok).collect();

                // Check config hash from first line
                let current_cfg_hash = generate_config_hash(&config);
                if let Some(first_line) = lines.first() {
                    if first_line.contains("\"_type\":\"config\"") {
                        // Extract cfg_hash from first line
                        if let Some(start) = first_line.find("\"cfg_hash\":\"") {
                            let hash_start = start + 12;
                            if let Some(end) = first_line[hash_start..].find('"') {
                                let baseline_hash = &first_line[hash_start..hash_start + end];
                                if baseline_hash != current_cfg_hash {
                                    eprintln!("\n⚠️  Config mismatch detected!");
                                    eprintln!("   Baseline cfg_hash: {}", baseline_hash);
                                    eprintln!("   Current cfg_hash:  {}", current_cfg_hash);
                                    eprintln!(
                                        "   Run with METATEST_UPDATE_BASELINE=1 to regenerate."
                                    );
                                    panic!(
                                        "Baseline/config mismatch - test parameters have changed"
                                    );
                                }
                            }
                        }
                    }
                }

                // Skip config line when loading identities
                let baseline: HashMap<String, IdentitySnapshot> = lines
                    .iter()
                    .filter(|l| !l.contains("\"_type\":\"config\""))
                    .filter_map(|l| IdentitySnapshot::from_json(l))
                    .map(|s| (s.id.clone(), s))
                    .collect();

                // Check for regressions
                let mut regressions: Vec<RegressionResult> = Vec::new();
                let mut new_ids: Vec<String> = Vec::new();
                let mut missing_ids: Vec<String> = Vec::new();

                for snap in &current_snapshots {
                    if let Some(base) = baseline.get(&snap.id) {
                        if let Some(reg) = check_regression(base, snap) {
                            regressions.push(reg);
                        }
                    } else {
                        new_ids.push(snap.id.clone());
                    }
                }

                let current_ids: std::collections::HashSet<_> =
                    current_snapshots.iter().map(|s| &s.id).collect();
                for id in baseline.keys() {
                    if !current_ids.contains(id) {
                        missing_ids.push(id.clone());
                    }
                }

                // Report results
                eprintln!("\n📊 Baseline Comparison (METATEST_SNAPSHOT=1):");
                eprintln!(
                    "   Current: {} | Baseline: {} | Regressions: {} | New: {} | Missing: {}",
                    current_snapshots.len(),
                    baseline.len(),
                    regressions.len(),
                    new_ids.len(),
                    missing_ids.len()
                );

                if !regressions.is_empty() {
                    eprintln!("\n🚨 REGRESSIONS DETECTED:");
                    for reg in &regressions {
                        eprintln!("   • {}: {}", reg.exp, reg.reasons.join(", "));
                    }
                }

                if !new_ids.is_empty() && new_ids.len() <= 5 {
                    eprintln!("\n➕ New identities (not in baseline):");
                    for id in &new_ids {
                        eprintln!("   • {}", id);
                    }
                } else if !new_ids.is_empty() {
                    eprintln!("\n➕ {} new identities not in baseline", new_ids.len());
                }

                if !missing_ids.is_empty() && missing_ids.len() <= 5 {
                    eprintln!("\n➖ Missing identities (in baseline, not in current):");
                    for id in &missing_ids {
                        eprintln!("   • {}", id);
                    }
                } else if !missing_ids.is_empty() {
                    eprintln!("\n➖ {} identities missing from current", missing_ids.len());
                }

                // Fail on regressions in CI
                if !regressions.is_empty() {
                    panic!(
                        "Baseline regression detected: {} identities worsened",
                        regressions.len()
                    );
                }
            }
        }
    }
}

/// Test shuffle canonicalization with dual checks:
/// 1. Semantic: simplify(E) ≡ simplify(shuffle(E)) numerically (must pass - bug if fails)
/// 2. Structural: simplify(E) == simplify(shuffle(E)) exactly (metric, optional strict mode)
#[test]
#[ignore]
fn metatest_shuffle_canonicalization() {
    let shuffle_enabled = env::var("METATEST_SHUFFLE").is_ok();
    if !shuffle_enabled {
        eprintln!("Shuffle test skipped. Set METATEST_SHUFFLE=1 to enable.");
        return;
    }

    let strict_canon = env::var("METATEST_STRICT_CANON").is_ok();
    let pairs = load_identity_pairs();
    if pairs.is_empty() {
        panic!("No identity pairs loaded!");
    }

    eprintln!("🔀 Shuffle Canonicalization Test");
    eprintln!(
        "   Mode: {}",
        if strict_canon {
            "STRICT (fail on structural diff)"
        } else {
            "METRIC (report only)"
        }
    );
    eprintln!("   Testing {} identity expressions...\n", pairs.len());

    let mut semantic_failures: Vec<String> = Vec::new();
    let mut structural_failures: Vec<String> = Vec::new();
    let mut tested = 0;

    for pair in &pairs {
        if pair.vars.len() != 1 {
            continue;
        }

        // Test LHS
        match test_shuffle_dual(&pair.exp, &pair.vars[0]) {
            ShuffleResult::Ok => {}
            ShuffleResult::ParseSkip => {} // Skip unsupported syntax
            ShuffleResult::StructuralDiff(msg) => {
                structural_failures.push(format!(
                    "{} (LHS): {}",
                    truncate_identity(&pair.exp, 30),
                    msg
                ));
            }
            ShuffleResult::SemanticFail(msg) => {
                semantic_failures.push(format!("{} (LHS): {}", pair.exp, msg));
            }
        }

        // Test RHS
        match test_shuffle_dual(&pair.simp, &pair.vars[0]) {
            ShuffleResult::Ok => {}
            ShuffleResult::ParseSkip => {} // Skip unsupported syntax
            ShuffleResult::StructuralDiff(msg) => {
                structural_failures.push(format!(
                    "{} (RHS): {}",
                    truncate_identity(&pair.simp, 30),
                    msg
                ));
            }
            ShuffleResult::SemanticFail(msg) => {
                semantic_failures.push(format!("{} (RHS): {}", pair.simp, msg));
            }
        }

        tested += 1;
    }

    // Report results
    eprintln!("📊 Shuffle Results:");
    eprintln!("   Tested: {} expressions", tested * 2);
    eprintln!(
        "   Semantic failures: {} (MUST be 0)",
        semantic_failures.len()
    );
    eprintln!(
        "   Structural diffs: {} (canonicalization gaps)",
        structural_failures.len()
    );

    // Semantic failures are always fatal (indicates a real bug)
    if !semantic_failures.is_empty() {
        eprintln!("\n🚨 SEMANTIC FAILURES (shuffle broke equivalence!):");
        for (i, fail) in semantic_failures.iter().take(5).enumerate() {
            eprintln!("   {}. {}", i + 1, fail);
        }
        panic!(
            "Shuffle caused {} semantic failures - this is a BUG!",
            semantic_failures.len()
        );
    }

    // Structural diffs are informative (or fatal in strict mode)
    if !structural_failures.is_empty() {
        eprintln!("\n⚠️  STRUCTURAL DIFFS (order-dependent canonicalization):");
        for (i, fail) in structural_failures.iter().take(5).enumerate() {
            eprintln!("   {}. {}", i + 1, fail);
        }
        if structural_failures.len() > 5 {
            eprintln!("   ... and {} more", structural_failures.len() - 5);
        }

        if strict_canon {
            panic!(
                "Strict canon mode: {} structural diffs - canonicalization not stable",
                structural_failures.len()
            );
        } else {
            eprintln!("\n💡 Run with METATEST_STRICT_CANON=1 to fail on structural diffs.");
        }
    }

    if semantic_failures.is_empty() && structural_failures.is_empty() {
        eprintln!("\n✅ All shuffle checks passed (semantic + structural)!");
    } else if semantic_failures.is_empty() {
        eprintln!(
            "\n✅ Semantic checks passed. {} structural diffs (non-blocking).",
            structural_failures.len()
        );
    }
}

/// Test metamorphic transforms: A(T(x)) ≡ B(T(x)) for transforms T.
/// Verifies that identities hold under substitution (scale, shift, square).
#[test]
#[ignore]
fn metatest_transform_identities() {
    let transform_enabled =
        env::var("METATEST_TRANSFORMS").is_ok() || env::var("METATEST_TRANSFORMS_DEFAULT").is_ok();

    if !transform_enabled {
        eprintln!("Transform test skipped. Set METATEST_TRANSFORMS=scale:2 or METATEST_TRANSFORMS_DEFAULT=1 to enable.");
        return;
    }

    let transforms = parse_meta_transforms_from_env();
    let pairs = load_identity_pairs();

    if pairs.is_empty() {
        panic!("No identity pairs loaded!");
    }

    // Parse min_valid factor from env
    let min_valid_factor: f64 = env::var("METATEST_TRANSFORM_MIN_VALID_FACTOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.6);

    eprintln!("🔄 MetaTransform Test");
    eprintln!(
        "   Transforms: {:?}",
        transforms.iter().map(|t| t.name()).collect::<Vec<_>>()
    );
    eprintln!("   Identities: {}", pairs.len());
    eprintln!("   min_valid_factor: {}\n", min_valid_factor);

    let mut total_tests = 0;
    let mut passed = 0;
    let mut semantic_failures: Vec<String> = Vec::new();
    let mut skipped_bucket = 0;

    for pair in &pairs {
        // Skip multi-variable identities
        if pair.vars.len() != 1 {
            continue;
        }

        // Gating by bucket: BranchSensitive only gets scale(2)
        for transform in &transforms {
            // Gate BranchSensitive - only allow scale(2)
            if pair.bucket == Bucket::BranchSensitive
                && !matches!(transform, MetaTransform::Scale(k) if (*k - 2.0).abs() < 1e-10)
            {
                skipped_bucket += 1;
                continue;
            }

            total_tests += 1;

            match test_transform_identity(pair, &pair.vars[0], transform, min_valid_factor) {
                TransformResult::Pass => passed += 1,
                TransformResult::Skip(_) => passed += 1, // Inconclusive is OK
                TransformResult::Fail(msg) => {
                    semantic_failures.push(format!(
                        "{} [{}]: {}",
                        truncate_identity(&pair.exp, 25),
                        transform.name(),
                        msg
                    ));
                }
            }
        }
    }

    // Report
    eprintln!("📊 Transform Results:");
    eprintln!("   Total tests: {}", total_tests);
    eprintln!("   Passed: {}", passed);
    eprintln!("   Skipped (bucket gate): {}", skipped_bucket);
    eprintln!("   Semantic failures: {}", semantic_failures.len());

    if !semantic_failures.is_empty() {
        eprintln!("\n🚨 TRANSFORM FAILURES:");
        for (i, fail) in semantic_failures.iter().take(10).enumerate() {
            eprintln!("   {}. {}", i + 1, fail);
        }
        if semantic_failures.len() > 10 {
            eprintln!("   ... and {} more", semantic_failures.len() - 10);
        }
        panic!(
            "Transform test failed with {} semantic failures",
            semantic_failures.len()
        );
    }

    eprintln!("\n✅ All transform tests passed!");
}

fn run_idempotence_contract_tests() -> IdempotenceMetrics {
    let cases = load_idempotence_expressions();
    let config = metatest_config();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let timeout = std::time::Duration::from_secs(5);

    let mut metrics = IdempotenceMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut numeric_examples: Vec<(String, String, String, String)> = Vec::new();
    let mut failed_examples: Vec<(String, String, String)> = Vec::new();

    eprintln!(
        "📊 Running simplify idempotence contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let expr_text = case.expr.clone();
        let vars = case.vars.clone();
        let filters = case.filters.clone();
        let family = case.family.clone();
        let config_clone = config.clone();

        let (tx, rx) = std::sync::mpsc::channel();
        let _handle = std::thread::Builder::new()
            .stack_size(METATEST_WORKER_STACK_SIZE_BYTES)
            .spawn(move || {
                let mut simplifier = Simplifier::with_default_rules();
                let parsed = match parse(&expr_text, &mut simplifier.context) {
                    Ok(e) => e,
                    Err(_) => {
                        let _ = tx.send(("parse_error".to_string(), String::new(), String::new()));
                        return;
                    }
                };

                let (simp1_raw, _) = simplifier.simplify(parsed);
                let simp1 = fold_constants_safe(&mut simplifier.context, simp1_raw);
                let (simp2_raw, _) = simplifier.simplify(simp1);
                let simp2 = fold_constants_safe(&mut simplifier.context, simp2_raw);

                let exact = cas_solver::runtime::compare_expr(&simplifier.context, simp1, simp2)
                    == std::cmp::Ordering::Equal;
                if exact {
                    let _ = tx.send(("exact".to_string(), String::new(), String::new()));
                    return;
                }

                let simp1_text = DisplayExpr {
                    context: &simplifier.context,
                    id: simp1,
                }
                .to_string();
                let simp2_text = DisplayExpr {
                    context: &simplifier.context,
                    id: simp2,
                }
                .to_string();

                if prove_zero_from_metamorphic_texts(
                    &mut simplifier,
                    &simp1_text,
                    &simp2_text,
                    simp1,
                    simp2,
                ) {
                    let _ = tx.send(("symbolic".to_string(), String::new(), String::new()));
                    return;
                }

                let diff_expr = simplifier.context.add(cas_ast::Expr::Sub(simp1, simp2));
                let (diff_simp_raw, _) = simplifier.simplify(diff_expr);
                let diff_simp = fold_constants_safe(&mut simplifier.context, diff_simp_raw);
                let diff_render = cas_formatter::LaTeXExpr {
                    context: &simplifier.context,
                    id: diff_simp,
                }
                .to_latex();
                let diff_shape = expr_shape_signature(&simplifier.context, diff_simp);

                match classify_numeric_equiv_for_vars(
                    &simplifier.context,
                    simp1,
                    simp2,
                    &vars,
                    &filters,
                    &config_clone,
                ) {
                    NumericCheckOutcome::Pass => {
                        let cause = numeric_only_cause_for_vars(
                            &simplifier.context,
                            simp1,
                            simp2,
                            &vars,
                            &filters,
                            &config_clone,
                            &diff_shape,
                        )
                        .label()
                        .to_string();
                        let _ = tx.send(("numeric".to_string(), diff_render, cause));
                    }
                    NumericCheckOutcome::Inconclusive(reason) => {
                        let _ = tx.send(("inconclusive".to_string(), reason, String::new()));
                    }
                    NumericCheckOutcome::Failed(reason) => {
                        let _ = tx.send(("failed".to_string(), reason, String::new()));
                    }
                }
            });

        match rx.recv_timeout(timeout) {
            Ok((kind, detail, cause)) => match kind.as_str() {
                "exact" => metrics.exact_stable += 1,
                "symbolic" => metrics.symbolic_stable += 1,
                "numeric" => {
                    metrics.numeric_stable += 1;
                    *metrics.numeric_causes.entry(cause.clone()).or_default() += 1;
                    if verbose && numeric_examples.len() < 20 {
                        numeric_examples.push((case.expr.clone(), family, detail, cause));
                    }
                }
                "inconclusive" => {
                    metrics.inconclusive += 1;
                    if verbose {
                        eprintln!(
                            "  ◐ INCONCLUSIVE [{}]: {} — {}",
                            case.family, case.expr, detail
                        );
                    }
                }
                "failed" => {
                    metrics.failed += 1;
                    failed_examples.push((case.expr.clone(), family, detail));
                }
                "parse_error" => {
                    metrics.parse_errors += 1;
                }
                _ => {
                    metrics.failed += 1;
                    failed_examples.push((
                        case.expr.clone(),
                        family,
                        format!("unexpected result kind: {}", kind),
                    ));
                }
            },
            Err(_) => {
                metrics.timeouts += 1;
            }
        }
    }

    eprintln!(
        "✅ Idempotence contracts: exact={} symbolic={} numeric={} inconclusive={} failed={} parse={} timeout={}",
        metrics.exact_stable,
        metrics.symbolic_stable,
        metrics.numeric_stable,
        metrics.inconclusive,
        metrics.failed,
        metrics.parse_errors,
        metrics.timeouts
    );

    if metrics.numeric_stable > 0 {
        print_numeric_only_cause_breakdown(&metrics.numeric_causes);
    }

    if verbose && !numeric_examples.is_empty() {
        eprintln!("\n── idempotence numeric-only examples ──");
        for (expr, family, residual, cause) in numeric_examples.iter().take(10) {
            eprintln!("  Expr [{}]: {}", family, expr);
            eprintln!("  Cause: {}", cause);
            if !residual.is_empty() {
                eprintln!("  Residual: {}", residual);
            }
            eprintln!();
        }
    }

    if !failed_examples.is_empty() {
        eprintln!("\n🚨 idempotence failures:");
        for (expr, family, detail) in failed_examples.iter().take(10) {
            eprintln!("  [{}] {} — {}", family, expr, detail);
        }
    }

    metrics
}

fn run_warnings_contract_tests() -> WarningsContractMetrics {
    let cases = load_warnings_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = WarningsContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running warnings contracts: {} expressions from {} families",
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
        if case.expect_warning {
            if first.warnings.is_empty() {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}] {} — expected warning, got none",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_warning_present += 1;
            }
        } else if first.warnings.is_empty() {
            metrics.expected_warning_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — unexpected warnings: {:?}",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                first.warnings
            ));
            case_failed = true;
        }

        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

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
            if first.warnings == second.warnings {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}] {} — warnings {:?} -> {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        case.expr,
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
        "✅ Warnings contracts: exact={} relaxed={} expected_warning_present={} expected_warning_absent={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_warning_present,
        metrics.expected_warning_absent,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ warnings contract relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 warnings contract failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_transparency_signal_contract_tests() -> TransparencySignalContractMetrics {
    let cases = load_transparency_signal_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = TransparencySignalContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running transparency-signal contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_transparency_metadata(&case.expr, case.mode) {
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

        let second = match simplify_with_transparency_metadata(&first.result, case.mode) {
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

        let has_signal = !first.warnings.is_empty() || !first.assumption_signals.is_empty();
        let mut case_failed = false;
        if case.expect_signal {
            if !has_signal {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}] {} — expected transparency signal, got none",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_signal_present += 1;
                if !first.warnings.is_empty() {
                    metrics.warning_channel_present += 1;
                }
                if !first.assumption_signals.is_empty() {
                    metrics.assumption_channel_present += 1;
                }
            }
        } else if !has_signal {
            metrics.expected_signal_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — unexpected transparency signals: warnings={:?}, assumptions={:?}",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                first.warnings,
                first.assumption_signals
            ));
            case_failed = true;
        }

        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        let first_assumptions: std::collections::HashSet<_> =
            first.assumption_signals.iter().cloned().collect();
        let second_assumptions: std::collections::HashSet<_> =
            second.assumption_signals.iter().cloned().collect();
        let introduced_assumptions: Vec<_> = second_assumptions
            .difference(&first_assumptions)
            .cloned()
            .collect();

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

        if !introduced_assumptions.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — introduced assumption-signals: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                introduced_assumptions,
                first.assumption_signals,
                second.assumption_signals
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.warnings == second.warnings
                && first.assumption_signals == second.assumption_signals
            {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}] {} — warnings {:?} -> {:?}, assumptions {:?} -> {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        case.expr,
                        first.warnings,
                        second.warnings,
                        first.assumption_signals,
                        second.assumption_signals
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
        "✅ Transparency-signal contracts: exact={} relaxed={} signal_present={} signal_absent={} warning_channel={} assumption_channel={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_signal_present,
        metrics.expected_signal_absent,
        metrics.warning_channel_present,
        metrics.assumption_channel_present,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ transparency-signal relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 transparency-signal failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_branch_transparency_contract_tests() -> BranchTransparencyContractMetrics {
    let cases = load_branch_transparency_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = BranchTransparencyContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running branch-transparency contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_transparency_metadata_with_inv_trig(
            &case.expr,
            case.mode,
            case.inv_trig,
        ) {
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

        let second = match simplify_with_transparency_metadata_with_inv_trig(
            &first.result,
            case.mode,
            case.inv_trig,
        ) {
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

        let first_branch_assumptions: Vec<String> = first
            .assumption_signals
            .iter()
            .filter(|signal| signal.starts_with("Branch|"))
            .cloned()
            .collect();
        let second_branch_assumptions: Vec<String> = second
            .assumption_signals
            .iter()
            .filter(|signal| signal.starts_with("Branch|"))
            .cloned()
            .collect();

        let has_signal = !first.warnings.is_empty() || !first_branch_assumptions.is_empty();
        let mut case_failed = false;
        if case.expect_signal {
            if !has_signal {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — expected branch transparency signal, got none",
                    case.family,
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_signal_present += 1;
                if !first.warnings.is_empty() {
                    metrics.warning_channel_present += 1;
                }
                if !first_branch_assumptions.is_empty() {
                    metrics.assumption_channel_present += 1;
                }
            }
        } else if !has_signal {
            metrics.expected_signal_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — unexpected branch transparency signals: warnings={:?}, assumptions={:?}",
                case.family,
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                first.warnings,
                first_branch_assumptions
            ));
            case_failed = true;
        }

        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        let first_assumptions: std::collections::HashSet<_> =
            first_branch_assumptions.iter().cloned().collect();
        let second_assumptions: std::collections::HashSet<_> =
            second_branch_assumptions.iter().cloned().collect();
        let introduced_assumptions: Vec<_> = second_assumptions
            .difference(&first_assumptions)
            .cloned()
            .collect();

        if !introduced_warnings.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                introduced_warnings,
                first.warnings,
                second.warnings
            ));
            case_failed = true;
        }

        if !introduced_assumptions.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — introduced branch assumption-signals: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                introduced_assumptions,
                first_branch_assumptions,
                second_branch_assumptions
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.warnings == second.warnings
                && first_branch_assumptions == second_branch_assumptions
            {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}|{}] {} — warnings {:?} -> {:?}, assumptions {:?} -> {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        inv_trig_policy_label(case.inv_trig),
                        case.expr,
                        first.warnings,
                        second.warnings,
                        first_branch_assumptions,
                        second_branch_assumptions
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
        "✅ Branch-transparency contracts: exact={} relaxed={} signal_present={} signal_absent={} warning_channel={} assumption_channel={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_signal_present,
        metrics.expected_signal_absent,
        metrics.warning_channel_present,
        metrics.assumption_channel_present,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ branch-transparency relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 branch-transparency failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_semantic_behavior_contract_tests() -> SemanticBehaviorContractMetrics {
    let cases = load_semantic_behavior_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = SemanticBehaviorContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running semantic-behavior contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_metadata_on_axes(&case.expr, case.mode, case.value_domain) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second =
            match simplify_with_metadata_on_axes(&first.result, case.mode, case.value_domain) {
                Ok(v) => v,
                Err(err) => {
                    metrics.parse_errors += 1;
                    failures.push(format!(
                        "[{}|{}|{}] {} -> '{}' reparsed failed: {}",
                        case.family,
                        value_domain_label(case.value_domain),
                        domain_mode_label(case.mode),
                        case.expr,
                        first.result,
                        err
                    ));
                    continue;
                }
            };

        let expected_ok = semantic_behavior_matches(&case.expectation, &first.result);
        let second_ok = semantic_behavior_matches(&case.expectation, &second.result);

        if !expected_ok {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — expected {}, got '{}'",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr,
                semantic_behavior_label(&case.expectation),
                first.result
            ));
            continue;
        }

        if !second_ok {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — second simplify broke behavior: first='{}', second='{}', expected {}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr,
                first.result,
                second.result,
                semantic_behavior_label(&case.expectation)
            ));
            continue;
        }

        if first.result == second.result {
            metrics.exact_preserved += 1;
        } else {
            metrics.relaxed_preserved += 1;
            if verbose && relaxed_examples.len() < 10 {
                relaxed_examples.push(format!(
                    "[{}|{}|{}] {} — result '{}' -> '{}'",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    case.expr,
                    first.result,
                    second.result
                ));
            }
        }
    }

    eprintln!(
        "✅ Semantic-behavior contracts: exact={} relaxed={} failed={} parse={}",
        metrics.exact_preserved, metrics.relaxed_preserved, metrics.failed, metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ semantic-behavior relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 semantic-behavior failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_complex_mode_behavior_contract_tests() -> ComplexModeBehaviorContractMetrics {
    let cases = load_complex_mode_behavior_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = ComplexModeBehaviorContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running complex-mode behavior contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_complex_mode_behavior(
            &case.expr,
            case.value_domain,
            case.complex_mode,
        ) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    complex_mode_label(case.complex_mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second =
            match simplify_with_complex_mode_behavior(&first, case.value_domain, case.complex_mode)
            {
                Ok(v) => v,
                Err(err) => {
                    metrics.parse_errors += 1;
                    failures.push(format!(
                        "[{}|{}|{}] {} -> '{}' reparsed failed: {}",
                        case.family,
                        value_domain_label(case.value_domain),
                        complex_mode_label(case.complex_mode),
                        case.expr,
                        first,
                        err
                    ));
                    continue;
                }
            };

        let expected_ok = semantic_behavior_matches(&case.expectation, &first);
        let second_ok = semantic_behavior_matches(&case.expectation, &second);

        if !expected_ok {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — expected {}, got '{}'",
                case.family,
                value_domain_label(case.value_domain),
                complex_mode_label(case.complex_mode),
                case.expr,
                semantic_behavior_label(&case.expectation),
                first
            ));
            continue;
        }

        if !second_ok {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — second simplify broke behavior: first='{}', second='{}', expected {}",
                case.family,
                value_domain_label(case.value_domain),
                complex_mode_label(case.complex_mode),
                case.expr,
                first,
                second,
                semantic_behavior_label(&case.expectation)
            ));
            continue;
        }

        if first == second {
            metrics.exact_preserved += 1;
        } else {
            metrics.relaxed_preserved += 1;
            if verbose && relaxed_examples.len() < 10 {
                relaxed_examples.push(format!(
                    "[{}|{}|{}] {} — result '{}' -> '{}'",
                    case.family,
                    value_domain_label(case.value_domain),
                    complex_mode_label(case.complex_mode),
                    case.expr,
                    first,
                    second
                ));
            }
        }
    }

    eprintln!(
        "✅ Complex-mode behavior contracts: exact={} relaxed={} failed={} parse={}",
        metrics.exact_preserved, metrics.relaxed_preserved, metrics.failed, metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ complex-mode behavior relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 complex-mode behavior failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_const_fold_behavior_contract_tests() -> ConstFoldBehaviorContractMetrics {
    let cases = load_const_fold_behavior_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = ConstFoldBehaviorContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running const-fold behavior contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match fold_with_const_fold_behavior(
            &case.expr,
            case.value_domain,
            case.const_fold_mode,
        ) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    const_fold_mode_label(case.const_fold_mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second =
            match fold_with_const_fold_behavior(&first, case.value_domain, case.const_fold_mode) {
                Ok(v) => v,
                Err(err) => {
                    metrics.parse_errors += 1;
                    failures.push(format!(
                        "[{}|{}|{}] {} -> '{}' reparsed failed: {}",
                        case.family,
                        value_domain_label(case.value_domain),
                        const_fold_mode_label(case.const_fold_mode),
                        case.expr,
                        first,
                        err
                    ));
                    continue;
                }
            };

        let expected_ok = semantic_behavior_matches(&case.expectation, &first);
        let second_ok = semantic_behavior_matches(&case.expectation, &second);

        if !expected_ok {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — expected {}, got '{}'",
                case.family,
                value_domain_label(case.value_domain),
                const_fold_mode_label(case.const_fold_mode),
                case.expr,
                semantic_behavior_label(&case.expectation),
                first
            ));
            continue;
        }

        if !second_ok {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — second const_fold broke behavior: first='{}', second='{}', expected {}",
                case.family,
                value_domain_label(case.value_domain),
                const_fold_mode_label(case.const_fold_mode),
                case.expr,
                first,
                second,
                semantic_behavior_label(&case.expectation)
            ));
            continue;
        }

        if first == second {
            metrics.exact_preserved += 1;
        } else {
            metrics.relaxed_preserved += 1;
            if verbose && relaxed_examples.len() < 10 {
                relaxed_examples.push(format!(
                    "[{}|{}|{}] {} — result '{}' -> '{}'",
                    case.family,
                    value_domain_label(case.value_domain),
                    const_fold_mode_label(case.const_fold_mode),
                    case.expr,
                    first,
                    second
                ));
            }
        }
    }

    eprintln!(
        "✅ Const-fold behavior contracts: exact={} relaxed={} failed={} parse={}",
        metrics.exact_preserved, metrics.relaxed_preserved, metrics.failed, metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ const-fold behavior relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 const-fold behavior failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_semantic_axes_contract_tests() -> SemanticAxesContractMetrics {
    let cases = load_semantic_axes_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = SemanticAxesContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running semantic-axes contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_metadata_on_axes(&case.expr, case.mode, case.value_domain) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second =
            match simplify_with_metadata_on_axes(&first.result, case.mode, case.value_domain) {
                Ok(v) => v,
                Err(err) => {
                    metrics.parse_errors += 1;
                    failures.push(format!(
                        "[{}|{}|{}] {} -> '{}' reparsed failed: {}",
                        case.family,
                        value_domain_label(case.value_domain),
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
                    "[{}|{}|{}] {} — expected requires, got none",
                    case.family,
                    value_domain_label(case.value_domain),
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
                "[{}|{}|{}] {} — unexpected requires: {:?}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr,
                first.required
            ));
            case_failed = true;
        }

        if case.expect_warning {
            if first.warnings.is_empty() {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — expected warning, got none",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_warning_present += 1;
            }
        } else if first.warnings.is_empty() {
            metrics.expected_warning_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — unexpected warnings: {:?}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr,
                first.warnings
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
                "[{}|{}|{}] {} — introduced requires: {:?} (first={:?}, second={:?})",
                case.family,
                value_domain_label(case.value_domain),
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
                "[{}|{}|{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family,
                value_domain_label(case.value_domain),
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
                        "[{}|{}|{}] {} — requires {:?} -> {:?}, warnings {:?} -> {:?}",
                        case.family,
                        value_domain_label(case.value_domain),
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
                "  ❌ [{}|{}|{}] {}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr
            );
        }
    }

    eprintln!(
        "✅ Semantic-axes contracts: exact={} relaxed={} requires_present={} requires_absent={} warning_present={} warning_absent={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_requires_present,
        metrics.expected_requires_absent,
        metrics.expected_warning_present,
        metrics.expected_warning_absent,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ semantic-axes relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 semantic-axes failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

pub(super) fn run_residual_pair_tests() -> ComboMetrics {
    let pairs = load_residual_pairs();
    run_direct_pair_tests(
        pairs,
        "residual metamorphic tests",
        "Residual tests",
        MetatestShortcutMode::SmokeClosure,
    )
}

pub(super) fn run_direct_pair_tests(
    pairs: Vec<ContextualPair>,
    suite_title: &str,
    suite_summary: &str,
    shortcut_mode: MetatestShortcutMode,
) -> ComboMetrics {
    run_direct_pair_tests_with_frontier_policy(
        pairs,
        suite_title,
        suite_summary,
        true,
        false,
        shortcut_mode,
    )
}

pub(super) fn run_direct_pair_tests_with_frontier_policy(
    pairs: Vec<ContextualPair>,
    suite_title: &str,
    suite_summary: &str,
    promote_known_domain_frontier: bool,
    enable_safe_window_shortcuts: bool,
    shortcut_mode: MetatestShortcutMode,
) -> ComboMetrics {
    let config = metatest_config();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let requested_pair_cap = std::env::var("METATEST_MAX_PAIRS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0);
    let requested_pair_start = std::env::var("METATEST_PAIR_START")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    let original_total_pairs = pairs.len();
    let (pair_start_offset, effective_total_pairs) = effective_combo_window(
        original_total_pairs,
        requested_pair_start,
        requested_pair_cap,
    );
    let pairs: Vec<_> = pairs
        .into_iter()
        .skip(pair_start_offset)
        .take(effective_total_pairs)
        .collect();

    let total_pairs = pairs.len();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut nf_convergent = 0usize;
    let mut proved_symbolic = 0usize;
    let mut numeric_only = 0usize;
    let mut inconclusive = 0usize;
    let mut domain_frontier = 0usize;
    let mut inconclusive_causes: HashMap<String, usize> = HashMap::new();
    let skipped = 0usize;
    let mut timeouts = 0usize;
    let mut cycle_events_total: usize = 0;
    let mut parse_errors = 0usize;
    let pair_timeout = if cfg!(debug_assertions) {
        std::time::Duration::from_secs(10)
    } else {
        std::time::Duration::from_secs(5)
    };
    let mut numeric_only_causes: HashMap<String, usize> = HashMap::new();
    let mut numeric_only_examples: Vec<(String, String, String, String, String)> = Vec::new();
    let mut domain_frontier_examples: Vec<(String, String, String)> = Vec::new();

    let num_families = pairs
        .iter()
        .map(|p| &p.family)
        .collect::<std::collections::HashSet<_>>()
        .len();

    eprintln!(
        "📊 Running {}: {} pairs from {} families (seed {})",
        suite_title, total_pairs, num_families, config.seed
    );
    if pair_start_offset > 0 || total_pairs < original_total_pairs {
        eprintln!(
            "🔬 Applying pair window [{}]: start {} size {} / {} planned contextual pairs",
            suite_summary, pair_start_offset, total_pairs, original_total_pairs
        );
    }

    for pair in &pairs {
        if shortcut_mode.allows_curated_shortcuts()
            && enable_safe_window_shortcuts
            && prove_zero_from_safe_window_parametrized_texts(&pair.lhs, &pair.rhs)
        {
            proved_symbolic += 1;
            passed += 1;
            continue;
        }
        if shortcut_mode.allows_curated_shortcuts()
            && prove_zero_from_curated_pair_corpus_text(&pair.lhs, &pair.rhs)
        {
            proved_symbolic += 1;
            passed += 1;
            continue;
        }
        if promote_known_domain_frontier {
            if let Some(frontier_reason) = known_domain_frontier_reason(&pair.lhs, &pair.rhs) {
                inconclusive += 1;
                domain_frontier += 1;
                passed += 1;
                record_inconclusive_reason(
                    &mut inconclusive_causes,
                    "domain_frontier",
                    frontier_reason,
                );
                if verbose && domain_frontier_examples.len() < 32 {
                    domain_frontier_examples.push((
                        pair.lhs.clone(),
                        pair.rhs.clone(),
                        frontier_reason.to_string(),
                    ));
                }
                continue;
            }
        }

        let lhs_str = pair.lhs.clone();
        let rhs_str = pair.rhs.clone();
        let free_vars = pair.vars.clone();
        let filters = pair.filters.clone();
        let family = pair.family.clone();
        let config_clone = config.clone();
        let promote_known_domain_frontier_clone = promote_known_domain_frontier;
        let enable_safe_window_shortcuts_clone = enable_safe_window_shortcuts;
        let pre_nf_engine_preproof = shortcut_mode.allows_pre_nf_proof_shortcuts();
        let shortcut_mode_clone = shortcut_mode;

        let (tx, rx) = std::sync::mpsc::channel();
        let _handle = std::thread::Builder::new()
            .stack_size(METATEST_WORKER_STACK_SIZE_BYTES)
            .spawn(move || {
                if pre_nf_engine_preproof
                    && prove_zero_from_engine_texts_in_child_process(&lhs_str, &rhs_str)
                {
                    let _ = tx.send(Some((
                        "proved".to_string(),
                        String::new(),
                        String::new(),
                        0,
                    )));
                    return;
                }

                let mut simplifier = Simplifier::with_default_rules();
                let lhs_parsed = match parse(&lhs_str, &mut simplifier.context) {
                    Ok(e) => e,
                    Err(_) => {
                        let _ = tx.send(Some((
                            "parse_error".to_string(),
                            String::new(),
                            String::new(),
                            0,
                        )));
                        return;
                    }
                };
                let rhs_parsed = match parse(&rhs_str, &mut simplifier.context) {
                    Ok(e) => e,
                    Err(_) => {
                        let _ = tx.send(Some((
                            "parse_error".to_string(),
                            String::new(),
                            String::new(),
                            0,
                        )));
                        return;
                    }
                };

                let opts = cas_solver::runtime::SimplifyOptions::default();
                let mut sub_cycles: usize = 0;
                let (mut lhs_simp, _, stats_lhs) =
                    simplifier.simplify_with_stats(lhs_parsed, opts.clone());
                sub_cycles += stats_lhs.cycle_events.len();
                let (mut rhs_simp, _, stats_rhs) =
                    simplifier.simplify_with_stats(rhs_parsed, opts.clone());
                sub_cycles += stats_rhs.cycle_events.len();

                {
                    let cfg = cas_solver::runtime::EvalConfig::default();
                    let mut budget = cas_solver::runtime::Budget::preset_cli();
                    if let Ok(r) = cas_solver::api::fold_constants(
                        &mut simplifier.context,
                        lhs_simp,
                        &cfg,
                        cas_solver::api::ConstFoldMode::Safe,
                        &mut budget,
                    ) {
                        lhs_simp = r.expr;
                    }
                    if let Ok(r) = cas_solver::api::fold_constants(
                        &mut simplifier.context,
                        rhs_simp,
                        &cfg,
                        cas_solver::api::ConstFoldMode::Safe,
                        &mut budget,
                    ) {
                        rhs_simp = r.expr;
                    }
                }

                let nf_match =
                    cas_solver::runtime::compare_expr(&simplifier.context, lhs_simp, rhs_simp)
                        == std::cmp::Ordering::Equal;
                if nf_match {
                    let _ = tx.send(Some((
                        "nf".to_string(),
                        String::new(),
                        String::new(),
                        sub_cycles,
                    )));
                    return;
                }

                if matches!(shortcut_mode_clone, MetatestShortcutMode::NfFirstPressure)
                    && prove_zero_from_engine_texts_in_child_process(&lhs_str, &rhs_str)
                {
                    let _ = tx.send(Some((
                        "proved".to_string(),
                        String::new(),
                        String::new(),
                        sub_cycles,
                    )));
                    return;
                }

                if prove_zero_from_metamorphic_texts(
                    &mut simplifier,
                    &lhs_str,
                    &rhs_str,
                    lhs_simp,
                    rhs_simp,
                ) {
                    let _ = tx.send(Some((
                        "proved".to_string(),
                        String::new(),
                        String::new(),
                        sub_cycles,
                    )));
                    return;
                }

                if enable_safe_window_shortcuts_clone
                    && prove_zero_from_safe_window_parametrized_texts(&lhs_str, &rhs_str)
                {
                    let _ = tx.send(Some((
                        "proved".to_string(),
                        String::new(),
                        String::new(),
                        sub_cycles,
                    )));
                    return;
                }

                let outcome = match free_vars.as_slice() {
                    [var] if filters.first().is_none_or(FilterSpec::is_none) => {
                        classify_numeric_equiv_1var_relaxed(
                            &simplifier.context,
                            lhs_simp,
                            rhs_simp,
                            var,
                            &config_clone,
                        )
                    }
                    [var] => {
                        let filter = filters.first().cloned().unwrap_or(FilterSpec::None);
                        let stats = check_numeric_equiv_1var_stats(
                            &simplifier.context,
                            lhs_simp,
                            rhs_simp,
                            var,
                            &config_clone,
                            &filter,
                        );
                        let result = finalize_numeric_equiv_1var(stats.clone(), &config_clone);
                        classify_numeric_check_with_stats(result, &stats)
                    }
                    [var1, var2] => classify_numeric_equiv_2var_relaxed(
                        &simplifier.context,
                        lhs_simp,
                        rhs_simp,
                        var1,
                        var2,
                        &config_clone,
                        filters.first().unwrap_or(&FilterSpec::None),
                        filters.get(1).unwrap_or(&FilterSpec::None),
                    ),
                    vars if vars.len() >= 3 => classify_numeric_equiv_nvar_relaxed(
                        &simplifier.context,
                        lhs_simp,
                        rhs_simp,
                        vars,
                        &filters,
                        &config_clone,
                    ),
                    _ => NumericCheckOutcome::Inconclusive(format!(
                        "Unsupported contextual numeric arity: {}",
                        free_vars.len()
                    )),
                };
                match outcome {
                    NumericCheckOutcome::Pass => {
                        let residual = {
                            let d = simplifier
                                .context
                                .add(cas_ast::Expr::Sub(lhs_simp, rhs_simp));
                            let (d_simp, _) = simplifier.simplify(d);
                            cas_formatter::LaTeXExpr {
                                context: &simplifier.context,
                                id: d_simp,
                            }
                            .to_latex()
                        };
                        let shape = {
                            let d = simplifier
                                .context
                                .add(cas_ast::Expr::Sub(lhs_simp, rhs_simp));
                            let (d_simp, _) = simplifier.simplify(d);
                            expr_shape_signature(&simplifier.context, d_simp)
                        };
                        let cause = match free_vars.as_slice() {
                            [var] => numeric_only_cause_for_1var(
                                &simplifier.context,
                                lhs_simp,
                                rhs_simp,
                                var,
                                &config_clone,
                                filters.first().unwrap_or(&FilterSpec::None),
                                &shape,
                            ),
                            [var1, var2] => numeric_only_cause_for_2var(
                                &simplifier.context,
                                lhs_simp,
                                rhs_simp,
                                var1,
                                var2,
                                &config_clone,
                                filters.first().unwrap_or(&FilterSpec::None),
                                filters.get(1).unwrap_or(&FilterSpec::None),
                                &shape,
                            ),
                            vars => classify_numeric_only_cause(None, vars.len(), &shape),
                        }
                        .label()
                        .to_string();
                        if promote_known_domain_frontier_clone {
                            if let Some(reason) = known_domain_frontier_reason_for_numeric_cause(
                                &cause, &lhs_str, &rhs_str,
                            ) {
                                let _ = tx.send(Some((
                                    "domain_frontier".to_string(),
                                    reason.to_string(),
                                    String::new(),
                                    sub_cycles,
                                )));
                                return;
                            }
                        }
                        let _ = tx.send(Some(("numeric".to_string(), residual, cause, sub_cycles)));
                    }
                    NumericCheckOutcome::Inconclusive(reason) => {
                        if promote_known_domain_frontier_clone {
                            if let Some(frontier_reason) =
                                known_domain_frontier_reason(&lhs_str, &rhs_str)
                            {
                                let _ = tx.send(Some((
                                    "domain_frontier".to_string(),
                                    frontier_reason.to_string(),
                                    String::new(),
                                    sub_cycles,
                                )));
                                return;
                            }
                        }
                        let _ = tx.send(Some((
                            "inconclusive".to_string(),
                            reason,
                            String::new(),
                            sub_cycles,
                        )));
                    }
                    NumericCheckOutcome::Failed(reason) => {
                        if promote_known_domain_frontier_clone {
                            if let Some(frontier_reason) =
                                known_domain_frontier_reason(&lhs_str, &rhs_str)
                            {
                                let _ = tx.send(Some((
                                    "domain_frontier".to_string(),
                                    frontier_reason.to_string(),
                                    String::new(),
                                    sub_cycles,
                                )));
                                return;
                            }
                        }
                        let _ = tx.send(Some((
                            "failed".to_string(),
                            reason,
                            String::new(),
                            sub_cycles,
                        )));
                    }
                }
            });

        match rx.recv_timeout(pair_timeout) {
            Ok(Some((kind, residual, cause, cycles))) => match kind.as_str() {
                "nf" => {
                    nf_convergent += 1;
                    passed += 1;
                    cycle_events_total += cycles;
                }
                "proved" => {
                    proved_symbolic += 1;
                    passed += 1;
                    cycle_events_total += cycles;
                }
                "numeric" => {
                    numeric_only += 1;
                    passed += 1;
                    cycle_events_total += cycles;
                    *numeric_only_causes.entry(cause.clone()).or_default() += 1;
                    if verbose && numeric_only_examples.len() < 200 {
                        numeric_only_examples.push((
                            pair.lhs.clone(),
                            pair.rhs.clone(),
                            family,
                            residual,
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
                        &residual,
                    );
                    if verbose && domain_frontier_examples.len() < 32 {
                        domain_frontier_examples.push((
                            pair.lhs.clone(),
                            pair.rhs.clone(),
                            residual,
                        ));
                    }
                }
                "inconclusive" => {
                    inconclusive += 1;
                    cycle_events_total += cycles;
                    record_inconclusive_reason(&mut inconclusive_causes, "inconclusive", &residual);
                }
                "parse_error" => {
                    parse_errors += 1;
                    passed += 1;
                }
                "failed" => {
                    failed += 1;
                    cycle_events_total += cycles;
                    if verbose {
                        eprintln!("  ❌ FAIL [{}]: {} vs {}", pair.family, pair.lhs, pair.rhs);
                        if !residual.is_empty() {
                            eprintln!("     Reason: {}", residual);
                        }
                    }
                }
                _ => {
                    failed += 1;
                    cycle_events_total += cycles;
                }
            },
            Ok(None) => {
                parse_errors += 1;
                passed += 1;
            }
            Err(_) => {
                timeouts += 1;
            }
        }
    }

    eprintln!(
        "✅ {}: {} passed, {} failed, {} timed out, {} parse errors, {} inconclusive",
        suite_summary, passed, failed, timeouts, parse_errors, inconclusive
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
    if verbose && numeric_only > 0 {
        print_numeric_only_cause_breakdown(&numeric_only_causes);
    }

    if verbose && !numeric_only_examples.is_empty() {
        eprintln!("\n── contextual numeric-only examples ──");
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

    ComboMetrics {
        op: "⇄ctx".to_string(),
        pairs: total_pairs,
        families: num_families,
        combos: total_pairs,
        nf_convergent,
        proved_quotient: proved_symbolic,
        proved_difference: 0,
        proved_composed: 0,
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

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_idempotence_contracts -- --ignored --nocapture
fn metatest_simplify_idempotence_contracts() {
    let m = run_idempotence_contract_tests();
    assert_eq!(m.failed, 0, "{} idempotence contracts failed", m.failed);
    assert_eq!(
        m.parse_errors, 0,
        "{} idempotence expressions failed to parse",
        m.parse_errors
    );
    assert_eq!(
        m.timeouts, 0,
        "{} idempotence contracts timed out",
        m.timeouts
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_warnings_contracts -- --ignored --nocapture
fn metatest_simplify_warnings_contracts() {
    let m = run_warnings_contract_tests();
    assert_eq!(m.failed, 0, "{} warnings contracts failed", m.failed);
    assert_eq!(
        m.parse_errors, 0,
        "{} warnings contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_transparency_signal_contracts -- --ignored --nocapture
fn metatest_simplify_transparency_signal_contracts() {
    let m = run_transparency_signal_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} transparency-signal contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} transparency-signal contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_branch_transparency_contracts -- --ignored --nocapture
fn metatest_simplify_branch_transparency_contracts() {
    let m = run_branch_transparency_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} branch-transparency contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} branch-transparency contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_semantic_behavior_contracts -- --ignored --nocapture
fn metatest_simplify_semantic_behavior_contracts() {
    let m = run_semantic_behavior_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} semantic-behavior contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} semantic-behavior contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_complex_mode_behavior_contracts -- --ignored --nocapture
fn metatest_simplify_complex_mode_behavior_contracts() {
    let m = run_complex_mode_behavior_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} complex-mode behavior contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} complex-mode behavior contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_const_fold_behavior_contracts -- --ignored --nocapture
fn metatest_simplify_const_fold_behavior_contracts() {
    let m = run_const_fold_behavior_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} const-fold behavior contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} const-fold behavior contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_semantic_axes_contracts -- --ignored --nocapture
fn metatest_simplify_semantic_axes_contracts() {
    let m = run_semantic_axes_contract_tests();
    assert_eq!(m.failed, 0, "{} semantic-axes contracts failed", m.failed);
    assert_eq!(
        m.parse_errors, 0,
        "{} semantic-axes contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_phase4_contract_suites -- --ignored --nocapture
fn metatest_simplify_phase4_contract_suites() {
    let idempotence = run_idempotence_contract_tests();
    let requires = run_requires_contract_tests();
    let warnings = run_warnings_contract_tests();
    let transparency = run_transparency_signal_contract_tests();
    let branch_transparency = run_branch_transparency_contract_tests();
    let semantic_behavior = run_semantic_behavior_contract_tests();
    let complex_mode_behavior = run_complex_mode_behavior_contract_tests();
    let const_fold_behavior = run_const_fold_behavior_contract_tests();
    let eval_path_behavior = run_eval_path_behavior_contract_tests();
    let eval_path_axes = run_eval_path_axes_contract_tests();
    let eval_path_inv_trig_axes = run_eval_path_inv_trig_axes_contract_tests();
    let requires_mode = run_requires_mode_contract_tests();
    let semantic_axes = run_semantic_axes_contract_tests();
    let assumption_trace = run_assumption_trace_contract_tests();

    eprintln!(
        "\n📦 Phase 4 contract summary: idempotence={} requires={} warnings={} transparency={} branch_transparency={} semantic_behavior={} complex_mode_behavior={} const_fold_behavior={} eval_path_behavior={} eval_path_axes={} eval_path_inv_trig_axes={} requires_mode={} semantic_axes={} assumption_trace={}",
        idempotence.total,
        requires.total,
        warnings.total,
        transparency.total,
        branch_transparency.total,
        semantic_behavior.total,
        complex_mode_behavior.total,
        const_fold_behavior.total,
        eval_path_behavior.total,
        eval_path_axes.total,
        eval_path_inv_trig_axes.total,
        requires_mode.total,
        semantic_axes.total,
        assumption_trace.total
    );

    assert_eq!(
        idempotence.failed, 0,
        "{} idempotence contracts failed",
        idempotence.failed
    );
    assert_eq!(
        idempotence.parse_errors, 0,
        "{} idempotence contract parse errors",
        idempotence.parse_errors
    );
    assert_eq!(
        idempotence.timeouts, 0,
        "{} idempotence contract timeouts",
        idempotence.timeouts
    );

    assert_eq!(
        requires.failed, 0,
        "{} requires contracts failed",
        requires.failed
    );
    assert_eq!(
        requires.parse_errors, 0,
        "{} requires contract parse errors",
        requires.parse_errors
    );

    assert_eq!(
        warnings.failed, 0,
        "{} warnings contracts failed",
        warnings.failed
    );
    assert_eq!(
        warnings.parse_errors, 0,
        "{} warnings contract parse errors",
        warnings.parse_errors
    );

    assert_eq!(
        transparency.failed, 0,
        "{} transparency contracts failed",
        transparency.failed
    );
    assert_eq!(
        transparency.parse_errors, 0,
        "{} transparency contract parse errors",
        transparency.parse_errors
    );

    assert_eq!(
        branch_transparency.failed, 0,
        "{} branch_transparency contracts failed",
        branch_transparency.failed
    );
    assert_eq!(
        branch_transparency.parse_errors, 0,
        "{} branch_transparency contract parse errors",
        branch_transparency.parse_errors
    );

    assert_eq!(
        semantic_behavior.failed, 0,
        "{} semantic_behavior contracts failed",
        semantic_behavior.failed
    );
    assert_eq!(
        semantic_behavior.parse_errors, 0,
        "{} semantic_behavior contract parse errors",
        semantic_behavior.parse_errors
    );

    assert_eq!(
        complex_mode_behavior.failed, 0,
        "{} complex_mode_behavior contracts failed",
        complex_mode_behavior.failed
    );
    assert_eq!(
        complex_mode_behavior.parse_errors, 0,
        "{} complex_mode_behavior contract parse errors",
        complex_mode_behavior.parse_errors
    );

    assert_eq!(
        const_fold_behavior.failed, 0,
        "{} const_fold_behavior contracts failed",
        const_fold_behavior.failed
    );
    assert_eq!(
        const_fold_behavior.parse_errors, 0,
        "{} const_fold_behavior contract parse errors",
        const_fold_behavior.parse_errors
    );

    assert_eq!(
        eval_path_behavior.failed, 0,
        "{} eval_path_behavior contracts failed",
        eval_path_behavior.failed
    );
    assert_eq!(
        eval_path_behavior.parse_errors, 0,
        "{} eval_path_behavior contract parse errors",
        eval_path_behavior.parse_errors
    );

    assert_eq!(
        eval_path_axes.failed, 0,
        "{} eval_path_axes contracts failed",
        eval_path_axes.failed
    );
    assert_eq!(
        eval_path_axes.parse_errors, 0,
        "{} eval_path_axes contract parse errors",
        eval_path_axes.parse_errors
    );

    assert_eq!(
        eval_path_inv_trig_axes.failed, 0,
        "{} eval_path_inv_trig_axes contracts failed",
        eval_path_inv_trig_axes.failed
    );
    assert_eq!(
        eval_path_inv_trig_axes.parse_errors, 0,
        "{} eval_path_inv_trig_axes contract parse errors",
        eval_path_inv_trig_axes.parse_errors
    );

    assert_eq!(
        requires_mode.failed, 0,
        "{} requires_mode contracts failed",
        requires_mode.failed
    );
    assert_eq!(
        requires_mode.parse_errors, 0,
        "{} requires_mode contract parse errors",
        requires_mode.parse_errors
    );

    assert_eq!(
        semantic_axes.failed, 0,
        "{} semantic_axes contracts failed",
        semantic_axes.failed
    );
    assert_eq!(
        semantic_axes.parse_errors, 0,
        "{} semantic_axes contract parse errors",
        semantic_axes.parse_errors
    );

    assert_eq!(
        assumption_trace.failed, 0,
        "{} assumption_trace contracts failed",
        assumption_trace.failed
    );
    assert_eq!(
        assumption_trace.parse_errors, 0,
        "{} assumption_trace contract parse errors",
        assumption_trace.parse_errors
    );
}

/// Unified regression benchmark combining combination tests (add, sub, mul, div)
/// and substitution tests into a single-run scorecard.
///
/// Run with:
/// ```text
/// cargo test --release -p cas_engine --test metamorphic_simplification_tests \
///     metatest_unified_benchmark -- --ignored --nocapture
/// ```
///
/// Key metrics per suite:
/// - NF-convergent: simplify(LHS) ≡ simplify(RHS) structurally (ideal)
/// - Proved-symbolic: simplify(LHS - RHS) = 0 (correct but different NFs)
/// - Numeric-only: only passes numeric check (target for improvement)
/// - Failed: semantic mismatches (regressions—must be 0)
/// - Timeout: combos that exceeded time limit (potential performance issues)
fn run_unified_benchmark(shortcut_mode: MetatestShortcutMode, enforce_clean: bool) {
    let seed = metatest_config().seed;

    // Phase 1: Combination tests (add, sub, mul, div)
    let combo_configs: Vec<(CombineOp, usize)> = vec![
        (CombineOp::Add, 30),
        (CombineOp::Sub, 30),
        (CombineOp::Mul, 150),
        (CombineOp::Div, 50),
    ];

    let mut all_metrics: Vec<ComboMetrics> = Vec::new();

    for (op, pairs) in &combo_configs {
        let metrics = match shortcut_mode {
            MetatestShortcutMode::SmokeClosure => run_csv_combination_tests(*pairs, false, *op),
            MetatestShortcutMode::StrictPressure => {
                run_csv_combination_tests_strict(*pairs, false, *op)
            }
            MetatestShortcutMode::NfFirstPressure => {
                run_csv_combination_tests_nf_first(*pairs, false, *op)
            }
        };
        all_metrics.push(metrics);
    }

    // Phase 2: Substitution tests
    let sub_metrics = match shortcut_mode {
        MetatestShortcutMode::SmokeClosure => run_substitution_tests(),
        MetatestShortcutMode::StrictPressure => run_substitution_tests_strict(),
        MetatestShortcutMode::NfFirstPressure => run_substitution_tests_nf_first(),
    };
    all_metrics.push(sub_metrics);
    let structural_sub_metrics = match shortcut_mode {
        MetatestShortcutMode::SmokeClosure => run_structural_substitution_tests(),
        MetatestShortcutMode::StrictPressure => run_structural_substitution_tests_strict(),
        MetatestShortcutMode::NfFirstPressure => run_structural_substitution_tests_nf_first(),
    };
    all_metrics.push(structural_sub_metrics);

    // Phase 3: Curated contextual tests
    let contextual_metrics = match shortcut_mode {
        MetatestShortcutMode::SmokeClosure => run_contextual_pair_tests(),
        MetatestShortcutMode::StrictPressure => run_contextual_pair_tests_strict(),
        MetatestShortcutMode::NfFirstPressure => run_contextual_pair_tests_nf_first(),
    };
    all_metrics.push(contextual_metrics);
    let contextual_rational_metrics = match shortcut_mode {
        MetatestShortcutMode::SmokeClosure => run_contextual_rational_pair_tests(),
        MetatestShortcutMode::StrictPressure => run_contextual_rational_pair_tests_strict(),
        MetatestShortcutMode::NfFirstPressure => run_contextual_rational_pair_tests_nf_first(),
    };
    all_metrics.push(contextual_rational_metrics);
    let contextual_trig_metrics = match shortcut_mode {
        MetatestShortcutMode::SmokeClosure => run_contextual_trig_pair_tests(),
        MetatestShortcutMode::StrictPressure => run_contextual_trig_pair_tests_strict(),
        MetatestShortcutMode::NfFirstPressure => run_contextual_trig_pair_tests_nf_first(),
    };
    all_metrics.push(contextual_trig_metrics);
    let contextual_polynomial_metrics = match shortcut_mode {
        MetatestShortcutMode::SmokeClosure => run_contextual_polynomial_pair_tests(),
        MetatestShortcutMode::StrictPressure => run_contextual_polynomial_pair_tests_strict(),
        MetatestShortcutMode::NfFirstPressure => run_contextual_polynomial_pair_tests_nf_first(),
    };
    all_metrics.push(contextual_polynomial_metrics);
    let contextual_radical_metrics = match shortcut_mode {
        MetatestShortcutMode::SmokeClosure => run_contextual_radical_pair_tests(),
        MetatestShortcutMode::StrictPressure => run_contextual_radical_pair_tests_strict(),
        MetatestShortcutMode::NfFirstPressure => run_contextual_radical_pair_tests_nf_first(),
    };
    all_metrics.push(contextual_radical_metrics);

    // Phase 4: Print unified table
    eprintln!();
    eprintln!("╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    eprintln!(
        "║            UNIFIED METAMORPHIC REGRESSION BENCHMARK [{}] (seed {:<10})                                  ║",
        shortcut_mode.benchmark_label(),
        seed
    );
    eprintln!("╠═══════╤════════╤══════════════╤══════════════╤══════════════╤══════════════╪════════╪═══════╪════════╪════════════════════╣");
    eprintln!("║ Suite │ Combos │ NF-convergent│ Proved-sym   │ Numeric-only │ Inconcl.     │ Failed │  T/O  │ Cycles │ Skip/Parse-err     ║");
    eprintln!("╠═══════╪════════╪══════════════╪══════════════╪══════════════╪══════════════╪════════╪═══════╪════════╪════════════════════╣");

    let mut total_combos = 0usize;
    let mut total_nf = 0usize;
    let mut total_proved = 0usize;
    let mut total_proved_quotient = 0usize;
    let mut total_proved_difference = 0usize;
    let mut total_proved_composed = 0usize;
    let mut total_numeric = 0usize;
    let mut total_inconclusive = 0usize;
    let mut total_failed = 0usize;
    let mut total_timeouts = 0usize;
    let mut total_cycles = 0usize;
    let mut total_skipped = 0usize;
    let mut total_symbolic_trackers = 0usize;
    let mut total_domain_frontier = 0usize;
    let mut total_inconclusive_causes: HashMap<String, usize> = HashMap::new();
    let mut total_domain_frontier_examples: Vec<(String, String, String, String)> = Vec::new();

    for m in &all_metrics {
        let effective = m
            .combos
            .saturating_sub(m.skipped)
            .saturating_sub(m.timeouts);
        let proved = m.proved_symbolic();
        let nf_pct = if effective > 0 {
            m.nf_convergent as f64 / effective as f64 * 100.0
        } else {
            0.0
        };
        let prov_pct = if effective > 0 {
            proved as f64 / effective as f64 * 100.0
        } else {
            0.0
        };
        let num_pct = if effective > 0 {
            m.numeric_only as f64 / effective as f64 * 100.0
        } else {
            0.0
        };
        let inc_pct = if effective > 0 {
            m.inconclusive as f64 / effective as f64 * 100.0
        } else {
            0.0
        };

        eprintln!(
            "║ {:5} │ {:>6} │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>6} │ {:>5} │ {:>6} │ {:>6}             ║",
            m.op, m.combos,
            m.nf_convergent, nf_pct,
            proved, prov_pct,
            m.numeric_only, num_pct,
            m.inconclusive, inc_pct,
            m.failed,
            m.timeouts,
            m.cycle_events_total,
            m.skipped,
        );

        total_combos += m.combos;
        total_nf += m.nf_convergent;
        total_proved += proved;
        total_proved_quotient += m.proved_quotient;
        total_proved_difference += m.proved_difference;
        total_proved_composed += m.proved_composed;
        total_numeric += m.numeric_only;
        total_inconclusive += m.inconclusive;
        total_failed += m.failed;
        total_timeouts += m.timeouts;
        total_cycles += m.cycle_events_total;
        total_skipped += m.skipped;
        total_symbolic_trackers += m.known_symbolic_residuals;
        total_domain_frontier += m.known_domain_frontier_count();
        for (lhs, rhs, reason) in &m.domain_frontier_examples {
            if total_domain_frontier_examples.len() >= 6 {
                break;
            }
            total_domain_frontier_examples.push((
                m.op.clone(),
                lhs.clone(),
                rhs.clone(),
                reason.clone(),
            ));
        }
        for (cause, count) in &m.inconclusive_causes {
            *total_inconclusive_causes.entry(cause.clone()).or_default() += *count;
        }
    }

    let total_effective = total_combos
        .saturating_sub(total_skipped)
        .saturating_sub(total_timeouts);
    let total_nf_pct = if total_effective > 0 {
        total_nf as f64 / total_effective as f64 * 100.0
    } else {
        0.0
    };
    let total_prov_pct = if total_effective > 0 {
        total_proved as f64 / total_effective as f64 * 100.0
    } else {
        0.0
    };
    let total_num_pct = if total_effective > 0 {
        total_numeric as f64 / total_effective as f64 * 100.0
    } else {
        0.0
    };
    let total_inc_pct = if total_effective > 0 {
        total_inconclusive as f64 / total_effective as f64 * 100.0
    } else {
        0.0
    };

    eprintln!("╠═══════╪════════╪══════════════╪══════════════╪══════════════╪══════════════╪════════╪═══════╪════════╪════════════════════╣");
    eprintln!(
        "║ TOTAL │ {:>6} │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>6} │ {:>5} │ {:>6} │ {:>6}             ║",
        total_combos,
        total_nf, total_nf_pct,
        total_proved, total_prov_pct,
        total_numeric, total_num_pct,
        total_inconclusive, total_inc_pct,
        total_failed,
        total_timeouts,
        total_cycles,
        total_skipped,
    );
    eprintln!("╚═══════╧════════╧══════════════╧══════════════╧══════════════╧══════════════╧════════╧═══════╧════════╧════════════════════╝");

    if matches!(shortcut_mode, MetatestShortcutMode::NfFirstPressure) {
        eprintln!(
            "ℹ️ NF-FIRST now runs true NF-first routing on add/sub, mul/div, substitution, and contextual suites."
        );
    }

    if total_proved > 0 {
        eprintln!();
        eprintln!(
            "🔢 Proved-symbolic breakdown: quotient {} | diff {} | composed {}",
            total_proved_quotient, total_proved_difference, total_proved_composed
        );
        let top_proved = top_proved_symbolic_contributors(&all_metrics, 5);
        if !top_proved.is_empty() {
            eprintln!("   Biggest proved contributors:");
            for (op, proved, quotient, diff, composed) in top_proved {
                eprintln!(
                    "   - {}: {} (quotient {}, diff {}, composed {})",
                    op, proved, quotient, diff, composed
                );
            }
        }
        let top_gap = top_normalization_gap_hotspots(&all_metrics, 5);
        if !top_gap.is_empty() {
            eprintln!("   Normalization-gap hotspots (diff + composed):");
            for (op, burden, diff, composed) in top_gap {
                eprintln!(
                    "   - {}: {} (diff {}, composed {})",
                    op, burden, diff, composed
                );
            }
        }
    }

    if total_failed > 0 {
        eprintln!(
            "⚠️  {} semantic failures detected — investigate before merging.",
            total_failed
        );
    }

    // Cycle events summary
    if total_cycles > 0 {
        eprintln!();
        eprintln!(
            "🔄 Cycle Events Summary: {} total across all suites",
            total_cycles
        );
        eprintln!("   The cycle detector successfully prevented oscillations.");
        eprintln!("   Run with METATEST_VERBOSE=1 for per-rule breakdown.");
    }

    if total_timeouts > 0 {
        eprintln!();
        eprintln!("⏱️  {} timeouts detected — consider increasing time budget or investigating slow combos.", total_timeouts);
    }

    let mut safe_window_metrics = None;

    if total_inconclusive > 0 {
        eprintln!();
        eprintln!(
            "◐ {} inconclusive numeric checks recorded — tracked separately from semantic failures.",
            total_inconclusive
        );
        if total_domain_frontier > 0 {
            let metrics =
                safe_window_metrics.get_or_insert_with(run_known_domain_frontier_safe_pair_tests);
            eprintln!(
                "🛡️  {} known domain-frontier case(s) counted inside inconclusive.",
                total_domain_frontier
            );
            if total_domain_frontier == total_inconclusive {
                eprintln!("   All remaining inconclusives are explicit domain-frontier cases.");
            }
            if safe_window_mirror_closes_all_domain_frontiers(total_domain_frontier, metrics) {
                eprintln!(
                    "   Safe-window mirror closes all {} domain-frontier cases symbolically.",
                    total_domain_frontier
                );
            }
            for m in &all_metrics {
                let domain_frontier = m.known_domain_frontier_count();
                if domain_frontier > 0 {
                    eprintln!("   - {}: {}", m.op, domain_frontier);
                }
            }
            if !total_domain_frontier_examples.is_empty() {
                eprintln!("   Examples:");
                for (op, lhs, rhs, reason) in total_domain_frontier_examples.iter().take(5) {
                    eprintln!("     [{}] {}  ↔  {}", op, lhs, rhs);
                    eprintln!("         reason: {}", reason);
                }
            }
        }
        print_inconclusive_breakdown(&total_inconclusive_causes);
    }

    if total_symbolic_trackers > 0 {
        eprintln!();
        eprintln!(
            "📌 {} known symbolic residual tracker(s) still counted inside numeric-only.",
            total_symbolic_trackers
        );
        for m in &all_metrics {
            if m.known_symbolic_residuals > 0 {
                eprintln!("   - {}: {}", m.op, m.known_symbolic_residuals);
            }
        }
    }

    if enforce_clean {
        assert_eq!(
            total_failed, 0,
            "unified benchmark detected {} semantic failure(s)",
            total_failed
        );
        assert_eq!(
            total_timeouts, 0,
            "unified benchmark detected {} timeout(s)",
            total_timeouts
        );
        assert_eq!(
            total_numeric, 0,
            "unified benchmark detected {} numeric-only case(s)",
            total_numeric
        );
        assert_eq!(
            total_inconclusive, total_domain_frontier,
            "unified benchmark has {} inconclusive case(s), but only {} are known domain-frontier",
            total_inconclusive, total_domain_frontier
        );
        if total_domain_frontier > 0 {
            let safe_window_metrics =
                safe_window_metrics.get_or_insert_with(run_known_domain_frontier_safe_pair_tests);
            assert!(
                safe_window_mirror_closes_all_domain_frontiers(
                    total_domain_frontier,
                    safe_window_metrics,
                ),
                "safe-window mirror no longer closes all {} domain-frontier case(s): proved={}, numeric={}, inconclusive={}, failed={}, timeouts={}",
                total_domain_frontier,
                safe_window_metrics.proved_symbolic(),
                safe_window_metrics.numeric_only,
                safe_window_metrics.inconclusive,
                safe_window_metrics.failed,
                safe_window_metrics.timeouts
            );
        }
    } else {
        eprintln!();
        eprintln!(
            "ℹ️ Strict mode is diagnostic: it reports engine pressure honestly and does not gate on failures/timeouts."
        );
    }

    eprintln!();
}

fn run_unified_benchmark_threaded(shortcut_mode: MetatestShortcutMode, enforce_clean: bool) {
    // Every suite here delegates combos to child processes. A filter that names
    // no test makes the child exit 0 with nothing done, and the benchmark then
    // reports whatever that emptiness happens to mean per classifier — so check
    // the filters BEFORE spending the run.
    assert_child_test_filters_resolve();
    let handle = std::thread::Builder::new()
        .stack_size(METATEST_WORKER_STACK_SIZE_BYTES)
        .spawn(move || run_unified_benchmark(shortcut_mode, enforce_clean))
        .expect("Failed to spawn unified benchmark thread");
    handle.join().expect("Unified benchmark thread panicked");
}

#[test]
#[ignore]
fn metatest_unified_benchmark() {
    run_unified_benchmark_threaded(MetatestShortcutMode::StrictPressure, false);
}

#[test]
#[ignore]
fn metatest_unified_benchmark_smoke() {
    run_unified_benchmark_threaded(MetatestShortcutMode::SmokeClosure, true);
}

#[test]
#[ignore]
fn metatest_unified_benchmark_strict() {
    run_unified_benchmark_threaded(MetatestShortcutMode::StrictPressure, false);
}

#[test]
#[ignore]
fn metatest_unified_benchmark_nf_first() {
    run_unified_benchmark_threaded(MetatestShortcutMode::NfFirstPressure, false);
}
