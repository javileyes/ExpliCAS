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

// ---------------------------------------------------------------------------
// Language axis (C5.1): the narration must not leak Spanish into the English
// wire — measured over the GUARDRAIL corpora, not over the 210-row showcase.
//
// The audit reported "9 rule names, structural parity 210/210 correct". That
// describes `web/examples.csv`, which is a shop window: it routes through the
// well-translated calculus paths. The probe measured `identity_pairs.csv` and
// found 26 % of rows leaking — six times more. A counter that only ever sees
// the showcase is the same failure of method that produced this audit.
// ---------------------------------------------------------------------------

/// Whole-word Spanish markers that never occur in the English catalogue, plus
/// the accented letters. Deliberately small and unambiguous: the point is a
/// counter that cannot cry wolf, not a translator.
const SPANISH_MARKERS: &[&str] = &[
    "de", "del", "la", "el", "los", "las", "una", "por", "con", "para", "entre", "cada", "sobre",
    "según", "usar", "sacar", "aplicar", "calcular", "reescribir", "agrupar", "cancelar",
    "simplificar", "derivar", "integrar", "resolver", "evaluar", "sustituir",
];

fn looks_spanish(text: &str) -> bool {
    if text.chars().any(|c| matches!(c, 'á' | 'é' | 'í' | 'ó' | 'ú' | 'ñ' | '¿' | '¡')) {
        return true;
    }
    text.split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .any(|w| {
            let lower = w.to_lowercase();
            SPANISH_MARKERS.contains(&lower.as_str())
        })
}

fn eval_wire_in(input: &str, language: Language) -> Option<EvalWireOutput> {
    let (tx, rx) = mpsc::channel();
    let owned = input.to_string();
    thread::spawn(move || {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut state = cas_session::SessionState::new();
        let out = evaluate_eval_command_in_memory_with_state(
            &mut engine,
            &mut state,
            cli_default_config(&owned),
            language,
            move |steps, events, ctx, mode| {
                cas_didactic::collect_step_payloads_with_events_localized(
                    steps, events, ctx, mode, language,
                )
            },
        );
        let _ = tx.send(out.ok());
    });
    rx.recv_timeout(TERMINATION_NET).ok().flatten()
}

#[derive(Default)]
struct LanguageResidue {
    rules: usize,
    substep_titles: usize,
    solve_descriptions: usize,
    warnings: usize,
    rows_touched: usize,
    rows_swept: usize,
    /// The DISTINCT offending strings. A count says how big the class is; this
    /// says what to translate, and it is what turns the counter into work.
    offenders: BTreeMap<String, usize>,
}

fn sweep_language_residue(inputs: &[String], residue: &mut LanguageResidue) {
    for input in inputs {
        let Some(wire) = eval_wire_in(input, Language::En) else {
            continue;
        };
        residue.rows_swept += 1;
        let before = residue.rules + residue.substep_titles + residue.solve_descriptions;
        for step in &wire.steps {
            if looks_spanish(&step.rule) {
                residue.rules += 1;
                *residue.offenders.entry(format!("rule: {}", step.rule)).or_insert(0) += 1;
            }
            for sub in &step.substeps {
                if looks_spanish(&sub.title) {
                    residue.substep_titles += 1;
                    *residue
                        .offenders
                        .entry(format!("substep: {}", sub.title))
                        .or_insert(0) += 1;
                }
            }
        }
        for step in &wire.solve_steps {
            if looks_spanish(&step.description) {
                residue.solve_descriptions += 1;
                *residue
                    .offenders
                    .entry(format!("solve: {}", step.description))
                    .or_insert(0) += 1;
            }
        }
        // Warnings are a SEPARATE class: they do not go through the i18n
        // catalogue in either direction (C5.3 owns that). Counted apart so the
        // narration ceiling is not polluted by a defect it does not own.
        for warning in &wire.warnings {
            if looks_spanish(&warning.assumption) {
                residue.warnings += 1;
                *residue
                    .offenders
                    .entry(format!("warning: {}", warning.assumption))
                    .or_insert(0) += 1;
            }
        }
        if residue.rules + residue.substep_titles + residue.solve_descriptions > before {
            residue.rows_touched += 1;
        }
    }
}

fn load_expr_column_calls(raw: &str, skip: usize, column: usize, wrap: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    raw.lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .skip(skip)
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            let expr = fields.get(column)?.trim();
            if expr.is_empty() || expr.contains('"') {
                return None;
            }
            Some(wrap.replace("{}", expr))
        })
        .filter(|e| seen.insert(e.clone()))
        .collect()
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
// C1.4 took the partial-fraction pair (rows 032/033) out. The 3 survivors have
// named owners and are NOT of the same class:
//   - rows 194/195 `potential`: "Verificación exacta" restates φ — the vector
//     verbs' mega-substep, its own cycle.
//   - row 199 `limit(exp(z), z, i*pi)`: the residual-policy substep, where
//     before == after IS the narration (same contract as `Conservar …`).
const E8_SUBSTEP_NOOP_CEILING: usize = 3;

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

/// Ceilings for the language axis. Non-zero today by design: the point of C5.1
/// is to MEASURE the class over the guardrail corpora, and the translation work
/// itself is C5.2 (rule names / substep titles) and C5.3 (warnings).
// C5.1 los midió sobre 1 049 expresiones (382 / 262 / 13) y C5.2 los bajó.
//
// Los nombres de regla llegan a CERO y pasan a aserción dura: es contrato, no
// techo. El sub-paso superviviente es UNA frase cuya cola sigue en español
// DESPUÉS del parámetro («…los límites laterales en z = pi·i (por la izquierda
// y por la derecha): si coinciden…»): la traducción por prefijo no la alcanza
// por construcción, y su arreglo es migrarla a `desc_key` con argumentos.
// C5.3 enrutó los warnings por un catálogo bidireccional (13 → 2). Los 2 que
// sobreviven son de la MISMA clase que el sub-paso: la frase sigue en español
// DESPUÉS del parámetro, así que el prefijo no la alcanza y su arreglo es
// migrarlos a mensaje con argumentos.
const EN_RULE_RESIDUE_CEILING: usize = 0;
const EN_SUBSTEP_RESIDUE_CEILING: usize = 1;
const EN_WARNING_RESIDUE_CEILING: usize = 2;

/// The narration must not leak Spanish into the English wire — swept over the
/// GUARDRAIL corpora, not just the 210-row showcase.
#[test]
#[ignore]
fn steps_quality_language_residue_over_guardrail_corpora() {
    let corpora: Vec<(&str, Vec<String>)> = vec![
        ("web_examples", load_web_examples()),
        ("identity_pairs_diff", load_identity_pairs_as_diff_calls()),
        ("derive_sources_integrate", load_derive_sources_as_integrate_calls()),
    ];

    let mut total = LanguageResidue::default();
    for (name, inputs) in &corpora {
        assert!(
            inputs.len() >= 50,
            "{name}: loader returned {} expressions — the corpus format drifted \
             and the sweep would have passed on nothing",
            inputs.len()
        );
        let mut per_corpus = LanguageResidue::default();
        sweep_language_residue(inputs, &mut per_corpus);
        println!(
            "es_residue_{name} rules={} substeps={} solve={} warnings={} rows_touched={} rows_swept={}",
            per_corpus.rules,
            per_corpus.substep_titles,
            per_corpus.solve_descriptions,
            per_corpus.warnings,
            per_corpus.rows_touched,
            per_corpus.rows_swept
        );
        total.rules += per_corpus.rules;
        total.substep_titles += per_corpus.substep_titles;
        total.solve_descriptions += per_corpus.solve_descriptions;
        total.warnings += per_corpus.warnings;
        total.rows_touched += per_corpus.rows_touched;
        total.rows_swept += per_corpus.rows_swept;
        for (k, v) in per_corpus.offenders {
            *total.offenders.entry(k).or_insert(0) += v;
        }
    }

    let mut ranked: Vec<(&String, &usize)> = total.offenders.iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(a.1));
    println!("--- cadenas distintas que fugan ({}) ---", ranked.len());
    for (text, hits) in ranked {
        println!("  {hits:5}x  {text}");
    }

    println!("es_residue_in_en_rules hits={} rows={}", total.rules, total.rows_touched);
    println!(
        "es_residue_in_en_substeps hits={} rows={}",
        total.substep_titles, total.rows_touched
    );
    println!(
        "es_residue_in_en_solve_steps hits={} rows={}",
        total.solve_descriptions, total.rows_touched
    );
    println!(
        "es_residue_in_en_warnings hits={} rows={}",
        total.warnings, total.rows_touched
    );
    println!("language_rows_swept hits={} rows={}", total.rows_swept, total.rows_swept);

    assert_eq!(
        total.rules, EN_RULE_RESIDUE_CEILING,
        "rule-name residue is a CONTRACT at zero, not a ceiling: every visible \
         rule name must have its English entry"
    );
    assert!(
        total.substep_titles <= EN_SUBSTEP_RESIDUE_CEILING,
        "substep-title residue {} exceeds the declared ceiling {EN_SUBSTEP_RESIDUE_CEILING}",
        total.substep_titles
    );
    assert!(
        total.warnings <= EN_WARNING_RESIDUE_CEILING,
        "warning residue {} exceeds the declared ceiling {EN_WARNING_RESIDUE_CEILING}",
        total.warnings
    );
}

fn load_identity_pairs_as_diff_calls() -> Vec<String> {
    load_expr_column_calls(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cas_solver/tests/identity_pairs.csv"
        )),
        0,
        0,
        "diff({}, x)",
    )
}

fn load_derive_sources_as_integrate_calls() -> Vec<String> {
    load_expr_column_calls(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cas_solver/tests/derive_pairs.csv"
        )),
        1,
        2,
        "integrate({}, x)",
    )
}

// ---------------------------------------------------------------------------
// Claim shadow run (C1.8, paso 0): what a substep AFFIRMS, verified over the
// WIRE, in OBSERVER mode.
//
// The C1.8 design (inventory of 422 emission points) prescribes measuring
// before enforcing, and the reason is a hard number: a prototype that verified
// `Equality` over every substep refuted 80 of 214 — but ~51 of those were
// LEGITIMATE non-equality relations (an antiderivative is not an equality).
// Switching a global equality check on would delete more than half of the
// correct narration. So: declare the relation per family, verify only the
// families whose relation is unambiguous, and publish the tally.
//
// It runs on the published strings rather than on engine internals, which
// makes it (a) free of any performance risk to the engine, and (b) the exact
// skeleton C1.9 needs — the generative tier differs only in where the
// expressions come from.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Claim {
    /// `d(after)/dvar == before`
    Antiderivative,
    /// `after == d(before)/dvar`
    Derivative,
}

#[derive(Default, Debug)]
struct ClaimTally {
    verified: usize,
    trivial: usize,
    refuted: Vec<String>,
    undecided: usize,
}

/// The relation a substep title declares. Deliberately a SMALL table: only the
/// families the inventory found unambiguous. Everything else is an explicit
/// abstention, not a silent pass.
fn claim_of_title(title: &str) -> Option<Claim> {
    let t = title.to_lowercase();
    if t.contains("hallar la antiderivada") || t.contains("find the antiderivative") {
        return Some(Claim::Antiderivative);
    }
    if t.starts_with("derivar respecto de") || t.starts_with("differentiate with respect to") {
        return Some(Claim::Derivative);
    }
    None
}

fn parse_in(ctx: &mut cas_ast::Context, text: &str) -> Option<cas_ast::ExprId> {
    cas_parser::parse(text, ctx).ok()
}

fn simplifies_to_zero(ctx: &mut cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    let mut simplifier = cas_solver::runtime::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) =
        simplifier.simplify_with_stats(expr, cas_solver::runtime::SimplifyOptions::default());
    std::mem::swap(&mut simplifier.context, ctx);
    matches!(ctx.get(rewritten), cas_ast::Expr::Number(n) if num_traits::Zero::is_zero(n))
}

/// Free variable names appearing in a rendered expression. The shadow run needs
/// this because THE WIRE DOES NOT PUBLISH THE CLAIM'S PARAMETERS: an
/// `Antiderivative` is only meaningful with respect to a variable, and the
/// substep publishes two strings and a title. Inferring the variable from the
/// call text refuted `cos(t) ⟹ sin(t)` — true, but differentiated with respect
/// to `x`. That finding is the point of running this in observer mode first.
fn candidate_vars(text: &str) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphabetic() {
            current.push(ch);
        } else {
            if current.len() == 1 && current != "e" {
                seen.insert(current.clone());
            }
            current.clear();
        }
    }
    if current.len() == 1 && current != "e" {
        seen.insert(current);
    }
    seen.into_iter().collect()
}

fn check_claim(claim: Claim, before: &str, after: &str, _var: &str, tally: &mut ClaimTally) {
    if before == after {
        tally.trivial += 1;
        return;
    }
    let mut vars = candidate_vars(before);
    for v in candidate_vars(after) {
        if !vars.contains(&v) {
            vars.push(v);
        }
    }
    if vars.is_empty() {
        tally.undecided += 1;
        return;
    }
    // Until the claim carries its variable as DATA, the shadow run accepts the
    // relation if it holds for SOME free variable. This is deliberately weaker
    // than the real check and is why this tier only MEASURES.
    let mut any_undecidable = false;
    for var in &vars {
        let mut ctx = cas_ast::Context::new();
        let (Some(before_id), Some(after_id)) =
            (parse_in(&mut ctx, before), parse_in(&mut ctx, after))
        else {
            tally.undecided += 1;
            return;
        };
        let (derived_from, compare_to) = match claim {
            Claim::Antiderivative => (after_id, before_id),
            Claim::Derivative => (before_id, after_id),
        };
        let Some(derivative) =
            cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
                &mut ctx,
                derived_from,
                var,
            )
        else {
            any_undecidable = true;
            continue;
        };
        let difference = ctx.add(cas_ast::Expr::Sub(derivative, compare_to));
        if simplifies_to_zero(&mut ctx, difference) {
            tally.verified += 1;
            return;
        }
    }
    if any_undecidable {
        tally.undecided += 1;
    } else {
        // NOT proof of a lie: the simplifier may simply fail to reach zero.
        // Recorded with its witness so every entry is adjudicated by hand.
        tally.refuted.push(format!("{claim:?}: {before}  ⟹  {after}"));
    }
}

/// Refutations surviving manual adjudication. MEASURED at 2, both the same
/// witness: `∫dx/(x³−2)`, whose antiderivative is correct (the engine gates its
/// emission on its own exact verification) but whose difference the simplifier
/// does not reduce to zero through the cbrt terms. They are UNDECIDED dressed as
/// refuted, and the honest ceiling is the measured number, not a round one.
const CLAIM_REFUTED_CEILING: usize = 2;

/// Shadow run: measure, do not enforce. The subset C1.8 turns on is chosen from
/// THIS table, not from a guess.
#[test]
#[ignore]
fn substep_claim_shadow_run_over_guardrail_corpora() {
    let corpora: Vec<(&str, Vec<String>)> = vec![
        ("web_examples", load_web_examples()),
        ("derive_sources_integrate", load_derive_sources_as_integrate_calls()),
    ];
    let mut tally = ClaimTally::default();
    let mut declared = 0usize;
    let mut abstained = 0usize;

    for (_name, inputs) in &corpora {
        for input in inputs {
            let Some(wire) = eval_wire_in(input, Language::Es) else {
                continue;
            };
            // The variable of the calculus verb, read off the call itself.
            let var = input
                .rsplit_once(',')
                .map(|(_, tail)| tail.trim_end_matches(')').trim().to_string())
                .filter(|v| v.len() == 1)
                .unwrap_or_else(|| "x".to_string());
            for step in &wire.steps {
                for sub in &step.substeps {
                    match claim_of_title(&sub.title) {
                        Some(claim) => {
                            declared += 1;
                            check_claim(claim, &sub.before, &sub.after, &var, &mut tally);
                        }
                        None => abstained += 1,
                    }
                }
            }
        }
    }

    println!("substep_claim_declared hits={declared} rows={declared}");
    println!("substep_claim_abstained hits={abstained} rows={abstained}");
    println!("substep_claim_verified hits={} rows={}", tally.verified, tally.verified);
    println!("substep_claim_trivial hits={} rows={}", tally.trivial, tally.trivial);
    println!(
        "substep_claim_undecided hits={} rows={}",
        tally.undecided, tally.undecided
    );
    println!(
        "substep_claim_refuted hits={} rows={}",
        tally.refuted.len(),
        tally.refuted.len()
    );
    for witness in tally.refuted.iter().take(25) {
        println!("  REFUTED {witness}");
    }

    assert!(
        tally.refuted.len() <= CLAIM_REFUTED_CEILING,
        "claim refutations {} exceed the declared ceiling {CLAIM_REFUTED_CEILING}",
        tally.refuted.len()
    );
}
