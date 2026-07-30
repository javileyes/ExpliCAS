//! Steps-divergence gate: `eval`'s RESULT must be identical with steps off/on.
//!
//! The bug class being fenced (ledger 2026-07-24, `d9da1c8a5`): several engine
//! layers are gated on the steps mode — `!collect_steps` shortcuts, the
//! `!has_step_listener()` root-shortcut regions, post-Core early exits — so
//! the engine takes DIFFERENT rewrite paths per presentation mode. Whenever
//! those paths are not confluent, the printed result depends on whether the
//! user asked to see the steps: `abs(pi - 3)` folded to `pi - 3` with steps on
//! but survived as `|3 - pi|` without them. Individual pins exist for the
//! known cases (`semantics_cli_contract_tests.rs`, `cli_contract_tests.rs`);
//! this gate sweeps the real corpora so the NEXT divergence of the class is
//! caught by `cargo test`, not by a manual audit.
//!
//! Mechanically it is the in-memory twin of `cas_cli eval --format json`:
//! same entrypoint (`evaluate_eval_command_in_memory_with_state`), same
//! config defaults, same step collector — only the process spawn is skipped,
//! which is what makes a multi-thousand-expression sweep affordable. Off vs
//! On is the real engine axis (`collect_steps` + listener attachment);
//! Compact shares On's evaluation path, so it needs no sweep of its own.
//!
//! The full pressure corpora (docs/*.csv, ~7.7k expressions) run under
//! `--ignored`:
//!   cargo test -p cas_cli --test steps_divergence_gate_tests --release -- --ignored

use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalNumericDisplay,
    EvalStepsMode, EvalValueDomain,
};
use cas_didactic::Language;
use cas_session::eval::{evaluate_eval_command_in_memory_with_state, EvalCommandConfig};
use std::collections::HashSet;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Termination net per (expression, mode). The engine has known
/// pathological-loop inputs (expand↔factor oscillation); without a net one of
/// them wedges the whole sweep forever — and per the ledger, a gate's time
/// budget must be a fixed net, not profile-dependent. 30s is ~3 orders of
/// magnitude above the corpus p99 in either profile, so it only trips on
/// genuine non-termination.
const TERMINATION_NET: Duration = Duration::from_secs(30);

/// Exact mirror of the `cas_cli eval` defaults (see `eval_command_config` and
/// the clap defaults in `cli_args.rs`) — the gate must compare the same two
/// runs a user gets from `eval "<expr>"` vs `eval "<expr>" --steps on`.
/// The value domain is the ONE axis the gate varies beyond the steps mode:
/// the C2 audit class (2026-07-30, S4-001/S4-002) lived exactly in shortcut
/// layers that dropped `--value-domain complex`, so the complex profile
/// sweeps the same off/on invariant under that flag.
fn cli_config(
    expr: &str,
    steps_mode: EvalStepsMode,
    value_domain: EvalValueDomain,
) -> EvalCommandConfig<'_> {
    EvalCommandConfig {
        expr,
        auto_store: false,
        max_chars: 2000,
        time_budget_ms: None,
        steps_mode,
        budget_preset: EvalBudgetPreset::Standard,
        strict: false,
        domain: EvalDomainMode::Generic,
        context_mode: EvalContextMode::Auto,
        branch_mode: EvalBranchMode::Strict,
        expand_policy: EvalExpandPolicy::Off,
        complex_mode: EvalComplexMode::Auto,
        const_fold: EvalConstFoldMode::Off,
        value_domain,
        complex_branch: EvalBranchMode::Principal,
        inv_trig: EvalInvTrigPolicy::Strict,
        assume_scope: EvalAssumeScope::Real,
        numeric_display: EvalNumericDisplay::Exact,
    }
}

#[derive(PartialEq, Eq)]
enum Outcome {
    Value(String),
    Error(String),
    Panic(String),
    Timeout,
}

/// Divergences the FIRST sweep of this gate found (2026-07-24), inventoried
/// and quarantined so the gate lands green while still fencing NEW cases.
/// Every entry is a real member of the bug class — a steps-gated shortcut
/// (root-shortcut regions / preorder fast paths) producing a result the
/// staged pipeline does not reach. Each needs its shortcut made confluent
/// (run in both modes, or taught to the phase pipeline), like the Región-A
/// migration did for its shortcuts.
///
/// The quarantine is self-invalidating: if an input stops diverging, or
/// diverges DIFFERENTLY, the sweep fails and the entry must be updated or
/// removed — the list can only shrink, never rot.
struct KnownDivergence {
    input: &'static str,
    off: &'static str,
    on: &'static str,
}

const QUARANTINE: &[KnownDivergence] = &[];

/// Inputs that exceed the termination net in BOTH modes — an engine
/// non-termination bug (its own class, tracked separately), not a steps
/// divergence: the sweep must neither wedge on them nor certify them.
/// Self-invalidating like `QUARANTINE`: an entry that terminates again fails
/// the sweep until it is removed.
const HANG_QUARANTINE: &[&str] = &[];

/// Divergences of the COMPLEX profile (`--value-domain complex`), the C2
/// audit axis (2026-07-30): shortcut layers that drop the value-domain flag
/// take a real-only path with steps=off that the staged pipeline (steps=on)
/// does not take. Inventoried and quarantined exactly like the real-mode
/// list: every entry is a measured C2-class member awaiting its owner, and
/// the list is self-invalidating (an input that stops diverging, or diverges
/// differently, fails the sweep until the entry is updated or removed).
const COMPLEX_QUARANTINE: &[KnownDivergence] = &[
    // S4-002 symbolic rows (P1): the steps-off shortcut still emits the |·|
    // real-only form for scaled symbolic radicands; the refutador reclassified
    // them under the `assume_scope: real` contract and they are PINNED
    // (`substitution_identities.csv:132`). Their owner is the confluence
    // migration of that shortcut, not a decline (both spellings are
    // value-correct under the pinned contract).
    KnownDivergence {
        input: "sqrt(4*x^2)",
        off: "2 * |x|",
        on: "2 * sqrt(x^2)",
    },
    KnownDivergence {
        input: "sqrt(9*x^2)",
        off: "3 * |x|",
        on: "3 * sqrt(x^2)",
    },
    KnownDivergence {
        input: "sqrt(16*x^4)",
        off: "4 * x^2",
        on: "4 * sqrt(x^4)",
    },
];

const COMPLEX_HANG_QUARANTINE: &[&str] = &[];

/// A fresh (unquarantined) hang leaks a spinning thread per mode; cap them so
/// a pathological corpus drift cannot melt the machine before the report.
const MAX_FRESH_HANGS: usize = 3;

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Value(v) => write!(f, "ok: {v}"),
            Outcome::Error(e) => write!(f, "error: {e}"),
            Outcome::Panic(p) => write!(f, "PANIC: {p}"),
            Outcome::Timeout => write!(
                f,
                "TIMEOUT: exceeded the {}s termination net",
                TERMINATION_NET.as_secs()
            ),
        }
    }
}

/// Run one eval under the termination net. The eval itself happens on a
/// helper thread; if it exceeds the net the thread is abandoned (Rust cannot
/// kill it — it dies with the test process) and the outcome is `Timeout`.
/// The sweep caps fresh hangs so a pathological corpus cannot pile up leaked
/// spinning threads.
fn eval_outcome(input: &str, steps_mode: EvalStepsMode, value_domain: EvalValueDomain) -> Outcome {
    let (tx, rx) = mpsc::channel();
    let owned = input.to_string();
    thread::spawn(move || {
        let _ = tx.send(eval_outcome_blocking(&owned, steps_mode, value_domain));
    });
    rx.recv_timeout(TERMINATION_NET).unwrap_or(Outcome::Timeout)
}

fn eval_outcome_blocking(
    input: &str,
    steps_mode: EvalStepsMode,
    value_domain: EvalValueDomain,
) -> Outcome {
    let run = catch_unwind(AssertUnwindSafe(|| {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut state = cas_session::SessionState::new();
        evaluate_eval_command_in_memory_with_state(
            &mut engine,
            &mut state,
            cli_config(input, steps_mode, value_domain),
            Language::Es,
            |steps, events, ctx, mode| {
                cas_didactic::collect_step_payloads_with_events_localized(
                    steps,
                    events,
                    ctx,
                    mode,
                    Language::Es,
                )
            },
        )
    }));
    match run {
        Ok(Ok(wire)) => Outcome::Value(wire.result),
        Ok(Err(message)) => Outcome::Error(message),
        Err(payload) => {
            let text = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            Outcome::Panic(text)
        }
    }
}

/// Sweep a corpus and fail with EVERY divergence (no fail-fast), so one run
/// yields the complete inventory. `min_expected` guards the loader itself: a
/// format drift that silently loads 0 expressions must fail loudly, not pass
/// as a green sweep (no silent caps).
fn assert_steps_mode_invariant(corpus: &str, inputs: Vec<String>, min_expected: usize) {
    assert_steps_mode_invariant_in_domain(
        corpus,
        inputs,
        min_expected,
        EvalValueDomain::Real,
        QUARANTINE,
        HANG_QUARANTINE,
    );
}

fn assert_steps_mode_invariant_in_domain(
    corpus: &str,
    inputs: Vec<String>,
    min_expected: usize,
    value_domain: EvalValueDomain,
    quarantine: &[KnownDivergence],
    hang_quarantine: &[&str],
) {
    assert!(
        inputs.len() >= min_expected,
        "{corpus}: loader returned {} expressions (expected at least {min_expected}) — \
         corpus moved or parser drifted; fix the loader before trusting the sweep",
        inputs.len()
    );

    let mut divergences = Vec::new();
    let mut quarantined = 0usize;
    let mut fresh_hangs = 0usize;
    for input in &inputs {
        let off = eval_outcome(input, EvalStepsMode::Off, value_domain);
        let on = eval_outcome(input, EvalStepsMode::On, value_domain);

        let hang_known = hang_quarantine.contains(&input.as_str());
        let both_hang = off == Outcome::Timeout && on == Outcome::Timeout;
        if both_hang || hang_known {
            match (both_hang, hang_known) {
                (true, true) => quarantined += 1,
                (true, false) => {
                    fresh_hangs += 1;
                    divergences.push(format!(
                        "  - `{input}`\n      ENGINE HANG in both modes (not a steps \
                         divergence): exceeded the {}s termination net — fix it or add it \
                         to HANG_QUARANTINE",
                        TERMINATION_NET.as_secs()
                    ));
                    if fresh_hangs >= MAX_FRESH_HANGS {
                        divergences.push(format!(
                            "  (sweep aborted after {fresh_hangs} fresh hangs to avoid \
                             leaking more spinning threads)"
                        ));
                        break;
                    }
                }
                (false, true) => divergences.push(format!(
                    "  - `{input}`\n      HANG_QUARANTINE STALE: terminates again \
                     (off: {off} | on: {on}) — remove it (the list only shrinks)"
                )),
                (false, false) => unreachable!(),
            }
            continue;
        }

        let known = quarantine.iter().find(|k| k.input == input);
        match (off != on, known) {
            (false, None) => {}
            (false, Some(_)) => divergences.push(format!(
                "  - `{input}`\n      QUARANTINE STALE: no longer diverges — remove its \
                 `KnownDivergence` entry (the list only shrinks)"
            )),
            (true, Some(k))
                if off == Outcome::Value(k.off.to_string())
                    && on == Outcome::Value(k.on.to_string()) =>
            {
                quarantined += 1;
            }
            (true, _) => divergences.push(format!(
                "  - `{input}`\n      off: {off}\n      on:  {on}{}",
                if known.is_some() {
                    "\n      (differs from its quarantine entry — update or fix)"
                } else {
                    ""
                }
            )),
        }
    }
    if quarantined > 0 {
        println!("{corpus}: {quarantined} known divergence(s) still quarantined (backlog)");
    }

    assert!(
        divergences.is_empty(),
        "{} steps-mode divergence(s) in {corpus} ({} expressions swept) — the result must not \
         depend on --steps:\n{}",
        divergences.len(),
        inputs.len(),
        divergences.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Corpus loaders
// ---------------------------------------------------------------------------

/// Split one CSV line on commas at paren/bracket depth 0 (expressions contain
/// commas inside `f(a, b)` calls, so a plain `split(',')` would shear them).
fn split_top_level_commas(line: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, ch) in line.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                fields.push(line[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    fields.push(line[start..].trim());
    fields
}

fn data_lines(raw: &str) -> impl Iterator<Item = &str> {
    raw.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
}

fn dedup_preserving_order(exprs: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = HashSet::new();
    exprs
        .into_iter()
        .filter(|e| !e.is_empty() && seen.insert(e.clone()))
        .collect()
}

/// Unquoted CSV with a header row: pick the given depth-0 columns.
fn load_expr_columns(raw: &str, skip_header: bool, columns: &[usize]) -> Vec<String> {
    let exprs = data_lines(raw)
        .skip(usize::from(skip_header))
        .flat_map(|line| {
            let fields = split_top_level_commas(line);
            columns
                .iter()
                .filter_map(|&c| fields.get(c).map(|f| (*f).to_string()))
                .collect::<Vec<_>>()
        });
    dedup_preserving_order(exprs)
}

/// `web/examples.csv` is fully quoted (`"group","expression","description"`)
/// and contains no escaped quotes, so `","` is an unambiguous separator.
fn load_web_examples() -> Vec<String> {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../web/examples.csv"
    ));
    let exprs = raw.lines().skip(1).filter_map(|line| {
        let line = line.trim().strip_prefix('"')?.strip_suffix('"')?;
        let fields: Vec<&str> = line.split("\",\"").collect();
        fields.get(1).map(|expr| (*expr).to_string())
    });
    dedup_preserving_order(exprs)
}

/// `corpus/limits.txt` rows are REPL-command syntax `limit EXPR, VAR, POINT`
/// (comments/blank lines aside); rewrap them as the `limit(EXPR, VAR, POINT)`
/// function-call form the eval surface accepts, putting the limits pipeline
/// under the gate.
fn load_limits_corpus_as_calls() -> Vec<String> {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cas_solver/tests/corpus/limits.txt"
    ));
    let exprs = data_lines(raw).filter_map(|line| {
        let args = line.strip_prefix("limit ")?;
        (split_top_level_commas(args).len() == 3).then(|| format!("limit({args})"))
    });
    dedup_preserving_order(exprs)
}

/// `identity_pairs.csv` rows are `exp,simp,var[,mode]`: sweep the DIFF verb
/// over each identity's expression with its declared variable.
fn load_identity_pairs_as_diff_calls() -> Vec<String> {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cas_solver/tests/identity_pairs.csv"
    ));
    let exprs = data_lines(raw).filter_map(|line| {
        let fields = split_top_level_commas(line);
        match (fields.first(), fields.get(2)) {
            (Some(exp), Some(var)) if !exp.is_empty() && var.len() == 1 => {
                Some(format!("diff({exp}, {var})"))
            }
            _ => None,
        }
    });
    dedup_preserving_order(exprs)
}

/// `derive_pairs.csv` sources (column 2, mostly polynomial/collect shapes —
/// cheap to integrate) swept through the INTEGRATE verb; free symbols other
/// than `x` are constants, which is a valid integrand shape of its own.
fn load_derive_sources_as_integrate_calls() -> Vec<String> {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cas_solver/tests/derive_pairs.csv"
    ));
    let exprs = data_lines(raw).skip(1).filter_map(|line| {
        let fields = split_top_level_commas(line);
        fields
            .get(2)
            .filter(|src| !src.is_empty())
            .map(|src| format!("integrate({src}, x)"))
    });
    dedup_preserving_order(exprs)
}

/// `equation_corpus.csv` rows are `equation,solve_var,...` — swept through the
/// same `eval` surface as the pins do, as `solve(<eq>, <var>)`, so the solve
/// pipeline's steps-gated shortcut layers are under the gate too.
fn load_equation_corpus_as_solve_calls() -> Vec<String> {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cas_solver/tests/equation_corpus.csv"
    ));
    let exprs = data_lines(raw).filter_map(|line| {
        let fields = split_top_level_commas(line);
        match (fields.first(), fields.get(1)) {
            (Some(eq), Some(var)) if eq.contains('=') && !var.is_empty() => {
                Some(format!("solve({eq}, {var})"))
            }
            _ => None,
        }
    });
    dedup_preserving_order(exprs)
}

// ---------------------------------------------------------------------------
// The gate, one corpus per test so the harness sweeps them in parallel
// ---------------------------------------------------------------------------

/// Canary for the harness itself: the family that motivated the gate must
/// agree through THIS in-memory path exactly like it does through the CLI
/// pins. If this fails, fix the harness before reading corpus sweeps.
#[test]
fn harness_canary_original_abs_divergence_family_agrees() {
    assert_steps_mode_invariant(
        "canary (abs of provably signed constants)",
        [
            "abs(pi - 3)",
            "abs(3 - pi)",
            "abs(e - 3)",
            "abs(1 - sqrt(2))",
            "abs(phi - 1)",
            "abs(x - 1)",
            // Negated provably-signed constants: the abs shortcut stripped
            // the Neg single-shot (`|−π| → |π|`) and diverged from the
            // pipeline's fold to `π` (2026-07-24, cycle 4/4).
            "abs(-pi)",
            "abs(-e)",
            "abs(-sqrt(2))",
            "abs(-2*pi)",
            "abs(-(pi+1))",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
        11,
    );
}

#[test]
fn web_examples_result_is_steps_mode_invariant() {
    assert_steps_mode_invariant("web/examples.csv", load_web_examples(), 200);
}

#[test]
fn identity_pairs_result_is_steps_mode_invariant() {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cas_solver/tests/identity_pairs.csv"
    ));
    // Rows are `exp,simp,var[,mode]`: both sides are evaluable expressions.
    assert_steps_mode_invariant(
        "identity_pairs.csv",
        load_expr_columns(raw, false, &[0, 1]),
        850,
    );
}

#[test]
fn substitution_identities_result_is_steps_mode_invariant() {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cas_solver/tests/substitution_identities.csv"
    ));
    assert_steps_mode_invariant(
        "substitution_identities.csv",
        load_expr_columns(raw, false, &[0, 1]),
        150,
    );
}

#[test]
fn derive_pairs_result_is_steps_mode_invariant() {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cas_solver/tests/derive_pairs.csv"
    ));
    // Rows are `id,family,source,target,...`: columns 2 and 3 are expressions.
    assert_steps_mode_invariant(
        "derive_pairs.csv",
        load_expr_columns(raw, true, &[2, 3]),
        600,
    );
}

#[test]
fn equation_corpus_solve_result_is_steps_mode_invariant() {
    assert_steps_mode_invariant(
        "equation_corpus.csv as solve()",
        load_equation_corpus_as_solve_calls(),
        40,
    );
}

#[test]
fn limits_corpus_result_is_steps_mode_invariant() {
    assert_steps_mode_invariant(
        "corpus/limits.txt as limit()",
        load_limits_corpus_as_calls(),
        25,
    );
}

#[test]
fn diff_verb_result_is_steps_mode_invariant() {
    assert_steps_mode_invariant(
        "identity_pairs.csv as diff()",
        load_identity_pairs_as_diff_calls(),
        450,
    );
}

#[test]
#[ignore = "integrate sweep is release-budget work (~45s release, several minutes debug) — \
            runs with the pressure tier: cargo test -p cas_cli --test \
            steps_divergence_gate_tests --release -- --ignored"]
fn integrate_verb_result_is_steps_mode_invariant() {
    assert_steps_mode_invariant(
        "derive_pairs.csv sources as integrate()",
        load_derive_sources_as_integrate_calls(),
        300,
    );
}

/// SIBLING AXIS of the steps gate: the result must not depend on the INPUT
/// ASSOCIATIVITY either. `f + (z)` and `f + z` are the same sum, but the
/// canonicalizer re-associates grouped input differently and downstream
/// rules can bury an identically-zero subgroup (`cos(2x)/2 + (fracs-u)`
/// ends as an unreduced mixed quotient while the flat spelling folds —
/// ledger 2026-07-25). This sweep is the INVENTORY that the future
/// canonicalization front needs (how many shapes are sensitive), kept on
/// the ignored tier until that front lands: it FAILS while the class is
/// open, by design — green here will mean the axis is closed.
#[test]
#[ignore = "input-associativity inventory for the canonicalization front — measures a KNOWN-open \
            class (fails by design while open): cargo test -p cas_cli --test \
            steps_divergence_gate_tests --release -- --ignored"]
fn input_associativity_pairs_inventory() {
    let zero_mixed = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/simplify_zero_mixed_corpus.csv"
    ));
    // Columns 3/4 are the identically-zero source expressions.
    let zeros = load_expr_columns(zero_mixed, true, &[3, 4]);
    assert!(
        zeros.len() >= 30,
        "zero-source loader drifted: {}",
        zeros.len()
    );

    let carriers = ["x", "cos(x)/2", "1/x", "x^2", "sqrt(x)", "2/3"];
    let mut divergences = Vec::new();
    let mut swept = 0usize;
    for z in &zeros {
        for f in carriers {
            let grouped = format!("{f} + ({z})");
            let flat = format!("{f} + {z}");
            let g = eval_outcome(&grouped, EvalStepsMode::Off, EvalValueDomain::Real);
            let p = eval_outcome(&flat, EvalStepsMode::Off, EvalValueDomain::Real);
            swept += 1;
            if g != p {
                divergences.push(format!(
                    "  - carrier `{f}` + zero `{z}`\n      grouped: {g}\n      flat:    {p}"
                ));
            }
        }
    }

    assert!(
        divergences.is_empty(),
        "{} input-associativity divergence(s) in {swept} carrier×zero pairs (result must not \
         depend on input grouping):\n{}",
        divergences.len(),
        divergences.join("\n")
    );
}

#[test]
#[ignore = "full pressure sweep (~7.7k expressions, ~15k evals) — run explicitly: \
            cargo test -p cas_cli --test steps_divergence_gate_tests --release -- --ignored"]
fn full_pressure_corpora_result_is_steps_mode_invariant() {
    let zero_mixed = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/simplify_zero_mixed_corpus.csv"
    ));
    assert_steps_mode_invariant(
        "docs/simplify_zero_mixed_corpus.csv",
        load_expr_columns(zero_mixed, true, &[0]),
        5000,
    );

    let embedded = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/embedded_equivalence_context_corpus.csv"
    ));
    assert_steps_mode_invariant(
        "docs/embedded_equivalence_context_corpus.csv",
        load_expr_columns(embedded, true, &[0]),
        1000,
    );
}

// ---------------------------------------------------------------------------
// COMPLEX value-domain profile (C2 audit axis, 2026-07-30)
// ---------------------------------------------------------------------------

/// Canary for the complex profile: the audited S4-001/S4-002 families must
/// agree across steps modes under `--value-domain complex` (they are the
/// forms whose steps-off shortcuts dropped the axis), and the analytic
/// identities that are complex-valid must KEEP collapsing identically.
#[test]
fn complex_profile_canary_audited_families_agree() {
    assert_steps_mode_invariant_in_domain(
        "complex canary (S4 families + analytic identities)",
        [
            // S4-001: real-only zero identities now decline in both modes.
            "sqrt(x^2)-abs(x)",
            "abs(x)-sqrt(x^2)",
            "sqrt(x^4)-x^2",
            "sqrt(x^2*y^2)-abs(x*y)",
            "sqrt((x-1)^2)-abs(x-1)",
            "sqrt((x+2)^2)-abs(x+2)",
            "2*ln(abs(x))-ln(x^2)",
            "ln(x^2)-2*ln(abs(x))",
            // S4-002: literal-i radicands fold to their true value.
            "sqrt(4*i^2)",
            "sqrt(16*i^4)",
            "sqrt(9*i^2)",
            // S4-002 symbolic rows: KNOWN divergence (P1, pinned contract
            // under assume_scope: real) — quarantined, measured, fenced.
            "sqrt(4*x^2)",
            "sqrt(9*x^2)",
            "sqrt(16*x^4)",
            // Positive-scale extraction stays alive in C.
            "sqrt(8)",
            "sqrt(18)",
            // Analytic identities: complex-valid, must still collapse.
            "sin(x)^2+cos(x)^2-1",
            "(x+1)^2-x^2-2*x-1",
            "e^(x+y)-e^x*e^y",
            "sinh(x)^2-cosh(x)^2+1",
            // Complex arithmetic staples.
            "i^2",
            "(1+i)*(1-i)",
            "sqrt(-4)",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        18,
        EvalValueDomain::Complex,
        COMPLEX_QUARANTINE,
        COMPLEX_HANG_QUARANTINE,
    );
}

/// Sweep the root-shortcut-sensitive slice of the web examples corpus under
/// the complex profile: expressions carrying the carriers the C2 class lives
/// in (roots, abs, logs, the imaginary unit). Filtering keeps the always-on
/// cost proportional to the fenced class instead of doubling the whole gate.
#[test]
fn complex_profile_web_examples_root_families_agree() {
    let inputs: Vec<String> = load_web_examples()
        .into_iter()
        .filter(|expr| {
            ["sqrt", "abs", "ln(", "log", "cbrt"]
                .iter()
                .any(|token| expr.contains(token))
        })
        .collect();
    assert_steps_mode_invariant_in_domain(
        "web/examples.csv (root/abs/log slice, complex profile)",
        inputs,
        20,
        EvalValueDomain::Complex,
        COMPLEX_QUARANTINE,
        COMPLEX_HANG_QUARANTINE,
    );
}
