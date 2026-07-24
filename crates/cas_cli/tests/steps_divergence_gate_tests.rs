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
fn cli_default_config(expr: &str, steps_mode: EvalStepsMode) -> EvalCommandConfig<'_> {
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
        value_domain: EvalValueDomain::Real,
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

const QUARANTINE: &[KnownDivergence] = &[
    // substitution_identities.csv — sqrt root shortcut leaves |2x| where the
    // full pipeline normalizes to 2|x| (cosmetic form divergence).
    KnownDivergence {
        input: "sqrt(4*x^2)",
        off: "|2 * x|",
        on: "2 * |x|",
    },
    // identity_pairs.csv — same sqrt-shortcut family: the shortcut keeps the
    // whole odd power inside the bars, the pipeline extracts the even part.
    KnownDivergence {
        input: "sqrt(x^6)",
        off: "|x^3|",
        on: "|x| * x^2",
    },
    // derive_pairs.csv — perfect-square-over-difference preorder path cancels
    // to `a - b` only on the steps-off path; with steps the quotient survives
    // (capability divergence: asking for steps LOSES the simplification).
    KnownDivergence {
        input: "(a^2-2*a*b+b^2)/(a-b)",
        off: "a - b",
        on: "(a^2 + b^2 - 2 * a * b) / (a - b)",
    },
    // derive_pairs.csv — partial-fraction recombination cancels the common
    // factor `a` only on the steps-off path.
    KnownDivergence {
        input: "1/(2*a)*(1/(x-a) - 1/(x+a))",
        off: "1 / (x^2 - a^2)",
        on: "a / (a * x^2 - a^3)",
    },
];

fn quarantine_entry(input: &str) -> Option<&'static KnownDivergence> {
    QUARANTINE.iter().find(|k| k.input == input)
}

/// Inputs that exceed the termination net in BOTH modes — an engine
/// non-termination bug (its own class, tracked separately), not a steps
/// divergence: the sweep must neither wedge on them nor certify them.
/// Self-invalidating like `QUARANTINE`: an entry that terminates again fails
/// the sweep until it is removed.
const HANG_QUARANTINE: &[&str] = &[];

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
fn eval_outcome(input: &str, steps_mode: EvalStepsMode) -> Outcome {
    let (tx, rx) = mpsc::channel();
    let owned = input.to_string();
    thread::spawn(move || {
        let _ = tx.send(eval_outcome_blocking(&owned, steps_mode));
    });
    rx.recv_timeout(TERMINATION_NET).unwrap_or(Outcome::Timeout)
}

fn eval_outcome_blocking(input: &str, steps_mode: EvalStepsMode) -> Outcome {
    let run = catch_unwind(AssertUnwindSafe(|| {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut state = cas_session::SessionState::new();
        evaluate_eval_command_in_memory_with_state(
            &mut engine,
            &mut state,
            cli_default_config(input, steps_mode),
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
        let off = eval_outcome(input, EvalStepsMode::Off);
        let on = eval_outcome(input, EvalStepsMode::On);

        let hang_known = HANG_QUARANTINE.contains(&input.as_str());
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

        let known = quarantine_entry(input);
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
        ]
        .into_iter()
        .map(String::from)
        .collect(),
        6,
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
