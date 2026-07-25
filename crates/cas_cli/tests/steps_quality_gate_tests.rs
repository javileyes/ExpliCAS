//! Steps-quality gate: the didactic narration the web publishes must satisfy
//! a handful of invariants that no test used to check.
//!
//! The bug class being fenced (audit 2026-07-25,
//! `docs/AUDITORIA_STEPS_HIGHLIGHTS_2026-07-25.md`): 546 findings over the 210
//! rows of `web/examples.csv`, 53 of them P0. The previous quality campaign
//! ("frente E") closed against ITS OWN metrics — chain continuity, noops,
//! duplicates, artifacts — while magic steps, rule-name truthfulness and the
//! whole substep surface were never measured at all. Two rows picked at random
//! by the user both failed. This lane exists so that the residual of a quality
//! campaign is a NUMBER in `cargo test`, not a claim in a document.
//!
//! Mechanically it is the in-memory twin of `cas_cli eval --steps on --lang es
//! --format json` (the exact configuration the web serves), reusing the shape
//! of `steps_divergence_gate_tests.rs`: same entrypoint, same defaults, same
//! termination net, `min_expected` against a silently broken loader.
//!
//! Two tiers:
//!   - `steps_quality_canary` (default): 10 hand-picked rows, ~seconds. It
//!     includes the two rows the user reported, so a regression on either is a
//!     red `cargo test`, not a re-audit.
//!   - `steps_quality_corpus_gate` (`--ignored`): the full 210-row corpus.
//!     cargo test -p cas_cli --test steps_quality_gate_tests --release -- --ignored --nocapture
//!
//! Determinism was verified empirically before landing the lane: two full
//! corpus runs hash identically on 210/210 rows once `timings_us` is removed.
//! There is no flakiness source here.
//!
//! DETECTOR POLICY. Only invariants that are ALREADY at zero are asserted
//! hard; anything still non-zero is published as a measured counter with a
//! declared ceiling, and drops to a hard assertion in the cycle that closes it.
//! Landing a red lane with a 400-line inventory nobody audits would destroy the
//! "the list can only shrink" contract, so the inventory fixture arrives with
//! its first non-zero detector (C1.3), not today.

use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalNumericDisplay,
    EvalStepsMode, EvalValueDomain, EvalWireOutput,
};
use cas_didactic::Language;
use cas_session::eval::{evaluate_eval_command_in_memory_with_state, EvalCommandConfig};
use cas_solver_core::rule_names::{
    RULE_CONSERVAR_DERIVADA_RESIDUAL, RULE_CONSERVAR_INTEGRAL_RESIDUAL,
    RULE_CONSERVAR_LIMITE_RESIDUAL,
};
use std::collections::BTreeMap;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Same fixed net as the divergence gate: a gate's time budget must not depend
/// on the profile, and a trip is a HARD failure, never a silent skip.
const TERMINATION_NET: Duration = Duration::from_secs(30);

fn cli_default_config(expr: &str) -> EvalCommandConfig<'_> {
    EvalCommandConfig {
        expr,
        auto_store: false,
        max_chars: 2000,
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
        numeric_display: EvalNumericDisplay::Exact,
    }
}

fn eval_wire(input: &str) -> Option<EvalWireOutput> {
    let (tx, rx) = mpsc::channel();
    let owned = input.to_string();
    thread::spawn(move || {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut state = cas_session::SessionState::new();
        let out = evaluate_eval_command_in_memory_with_state(
            &mut engine,
            &mut state,
            cli_default_config(&owned),
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
        );
        let _ = tx.send(out.ok());
    });
    match rx.recv_timeout(TERMINATION_NET) {
        Ok(wire) => wire,
        Err(_) => panic!(
            "{input}: exceeded the {}s termination net",
            TERMINATION_NET.as_secs()
        ),
    }
}

/// The honest-residual steps are the ONE legitimate `before == after`
/// narration: their whole point is "this stays as itself".
fn is_honest_residual(rule: &str) -> bool {
    matches!(
        rule,
        RULE_CONSERVAR_DERIVADA_RESIDUAL
            | RULE_CONSERVAR_INTEGRAL_RESIDUAL
            | RULE_CONSERVAR_LIMITE_RESIDUAL
    )
}

/// `_`, `^` and `~` inside a `\text{…}` group are a HARD MathJax error
/// (`'_' allowed only in math mode`), and a hard error does not degrade one
/// line — it kills the rendering of the whole expression. Verified against
/// MathJax 3 itself, the renderer `web/index.html:18` loads.
///
/// Returns the offending characters found in text mode.
fn unescaped_specials_in_text_mode(latex: &str) -> Vec<char> {
    let chars: Vec<char> = latex.chars().collect();
    let mut found = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i..].starts_with(&['\\', 't', 'e', 'x', 't', '{']) {
            let mut depth = 1;
            let mut j = i + 6;
            let mut escaped = false;
            while j < chars.len() && depth > 0 {
                let ch = chars[j];
                if escaped {
                    escaped = false;
                } else {
                    match ch {
                        '\\' => escaped = true,
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        '_' | '^' | '~' => found.push(ch),
                        _ => {}
                    }
                }
                j += 1;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    found
}

fn braces_balanced(latex: &str) -> bool {
    let mut depth: i32 = 0;
    let mut escaped = false;
    for ch in latex.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

#[derive(Default)]
struct QualityReport {
    /// Hard invariants: any hit is a failure, with the offending row quoted.
    violations: Vec<String>,
    /// Published measures, `name -> (hits, rows)`.
    measures: BTreeMap<&'static str, (usize, usize)>,
}

impl QualityReport {
    fn measure(&mut self, name: &'static str, hits: usize) {
        let entry = self.measures.entry(name).or_insert((0, 0));
        entry.0 += hits;
        if hits > 0 {
            entry.1 += 1;
        }
    }

    fn print(&self) {
        for (name, (hits, rows)) in &self.measures {
            println!("{name} hits={hits} rows={rows}");
        }
    }
}

fn inspect_row(input: &str, report: &mut QualityReport) {
    let Some(wire) = eval_wire(input) else {
        // An evaluation error is a legitimate outcome for some rows (session
        // references evaluated standalone); it is not a quality violation.
        return;
    };

    // ---- hard invariant: the counter contract -------------------------------
    if wire.steps_count != wire.steps.len() {
        report.violations.push(format!(
            "[{input}] steps_count={} but steps.len()={} — steps_count must be exactly \
             the length of the `steps` array; solve_steps and substeps have their own counters",
            wire.steps_count,
            wire.steps.len()
        ));
    }

    // ---- hard invariant: no visible no-op steps -----------------------------
    let mut noops = 0;
    for step in &wire.steps {
        if step.before == step.after && !is_honest_residual(&step.rule) {
            noops += 1;
            report.violations.push(format!(
                "[{input}] step {} ({}) has before == after ({}) — a step whose visible \
                 state does not change narrates nothing",
                step.index, step.rule, step.before
            ));
        }
    }
    report.measure("D5_noop_step", noops);

    // ---- hard invariant: no consecutive duplicates --------------------------
    let mut duplicates = 0;
    for pair in wire.steps.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if a.rule == b.rule && a.before == b.before && a.after == b.after {
            duplicates += 1;
            report.violations.push(format!(
                "[{input}] steps {} and {} are identical ({}) — the same step shown twice",
                a.index, b.index, a.rule
            ));
        }
    }
    report.measure("D6_duplicate_consecutive", duplicates);

    // ---- hard invariant: every published LaTeX renders -----------------------
    // An unbalanced brace does not degrade the line, it breaks the whole row's
    // rendering in the browser.
    let mut unbalanced = 0;
    let mut text_mode_specials = 0;
    let mut check = |label: &str, latex: &str| {
        if !braces_balanced(latex) {
            unbalanced += 1;
            report
                .violations
                .push(format!("[{input}] {label} has unbalanced braces: {latex}"));
        }
        let specials = unescaped_specials_in_text_mode(latex);
        if !specials.is_empty() {
            text_mode_specials += specials.len();
            report.violations.push(format!(
                "[{input}] {label} puts {specials:?} raw inside \\text{{…}} — a hard MathJax \
                 error that kills the whole expression's rendering: {latex}"
            ));
        }
    };
    for step in &wire.steps {
        check(&format!("steps[{}].rule_latex", step.index), &step.rule_latex);
        check(
            &format!("steps[{}].before_latex", step.index),
            &step.before_latex,
        );
        check(
            &format!("steps[{}].after_latex", step.index),
            &step.after_latex,
        );
        for (j, sub) in step.substeps.iter().enumerate() {
            if let Some(latex) = &sub.before_latex {
                check(&format!("steps[{}].substeps[{j}].before_latex", step.index), latex);
            }
            if let Some(latex) = &sub.after_latex {
                check(&format!("steps[{}].substeps[{j}].after_latex", step.index), latex);
            }
        }
    }
    for step in &wire.solve_steps {
        check(&format!("solve_steps[{}].lhs_latex", step.index), &step.lhs_latex);
        check(&format!("solve_steps[{}].rhs_latex", step.index), &step.rhs_latex);
    }
    report.measure("D9_unbalanced_braces", unbalanced);
    report.measure("D10_text_mode_unescaped", text_mode_specials);

    // ---- published measures --------------------------------------------------
    report.measure("steps_total", wire.steps.len());
    report.measure("solve_steps_total", wire.solve_steps.len());
    report.measure("substeps_total", wire.substeps_count);

    // A substep that claims a manoeuvre and shows the same thing on both sides
    // ("factor the denominator": x^3 - 2 -> x^3 - 2). Non-zero today (audit
    // rows 032/033/194/195); C1.4 takes it to zero and it becomes an assertion.
    let mut substep_noops = 0;
    for step in &wire.steps {
        for sub in &step.substeps {
            if sub.before == sub.after {
                substep_noops += 1;
            }
        }
    }
    report.measure("E8_substep_noop", substep_noops);
}

/// Ceilings for the measures that are NOT yet zero. Each one names the cycle
/// that closes it; a measure that drops below its ceiling must have the ceiling
/// tightened in the same commit, so the numbers can only go down.
const E8_SUBSTEP_NOOP_CEILING: usize = 5; // → 0 in C1.4 (partial fractions)

fn run_gate(label: &str, inputs: &[String], min_expected: usize) {
    assert!(
        inputs.len() >= min_expected,
        "{label}: loader returned {} expressions (expected at least {min_expected}) — \
         the corpus format drifted and the gate would have passed on nothing",
        inputs.len()
    );

    let mut report = QualityReport::default();
    for input in inputs {
        inspect_row(input, &mut report);
    }

    println!("--- {label}: {} rows ---", inputs.len());
    report.print();

    let e8 = report.measures.get("E8_substep_noop").map_or(0, |m| m.0);
    assert!(
        e8 <= E8_SUBSTEP_NOOP_CEILING,
        "{label}: E8_substep_noop={e8} exceeds the declared ceiling \
         {E8_SUBSTEP_NOOP_CEILING} — a new substep claims a manoeuvre it does not perform"
    );

    assert!(
        report.violations.is_empty(),
        "{label}: {} quality violations\n{}",
        report.violations.len(),
        report.violations.join("\n")
    );
}

/// The rows the corpus gate must never regress on, small enough to run by
/// default. The first two are the ones the user reported.
const CANARY_ROWS: &[&str] = &[
    "taylor(sin(x), x, 0, 5)",
    "integrate(2*x/sqrt(4+x^4)+1, x)",
    "integrate(cos(t)^2, t, pi/6, pi/3)",
    "integrate(1/(x^4-1), x, 2, oo)",
    "solve(sin(x)=1/2,x)",
    "solve(x^2-2*x-3>0,x)",
    "diff(x^2*y, x, y)",
    "laplacian(ln(x^2+y^2), [x,y])",
    "limit((1+1/x)^x, x, infinity)",
    "dsolve(diff(y,x)=x*y, y, x)",
];

#[test]
fn steps_quality_canary() {
    let inputs: Vec<String> = CANARY_ROWS.iter().map(|s| (*s).to_string()).collect();
    run_gate("canary", &inputs, CANARY_ROWS.len());
}

/// The full corpus the web serves. `--ignored` because it is a sweep, not a
/// unit test; the scorecard runs it in release on every cycle.
#[test]
#[ignore]
fn steps_quality_corpus_gate() {
    run_gate("web_examples", &load_web_examples(), 200);
}

fn load_web_examples() -> Vec<String> {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../web/examples.csv"
    ));
    let mut seen = std::collections::HashSet::new();
    raw.lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.trim().strip_prefix('"')?.strip_suffix('"')?;
            let fields: Vec<&str> = line.split("\",\"").collect();
            fields.get(1).map(|expr| (*expr).to_string())
        })
        .filter(|expr| seen.insert(expr.clone()))
        .collect()
}
