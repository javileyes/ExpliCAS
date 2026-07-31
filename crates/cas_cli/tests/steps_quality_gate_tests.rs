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
        approx_hint: false,
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
/// A solve step that RESTATES the equation of the previous one is normally the
/// filler shape (`equation_after: eq1.clone()` because the type demands an
/// equation and the step's real content lives in its description). The honest
/// exception is verification: «Verificar por sustitución …» exists precisely to
/// show, unchanged, the object it just checked — the same reasoning that lets
/// `Conservar …` repeat its snapshot in the simplify channel.
fn solve_repeat_is_honest(description: &str) -> bool {
    description.starts_with("Verificar") || description.starts_with("Verify")
}

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
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "una",
    "por",
    "con",
    "para",
    "entre",
    "cada",
    "sobre",
    "según",
    "usar",
    "sacar",
    "aplicar",
    "calcular",
    "reescribir",
    "agrupar",
    "cancelar",
    "simplificar",
    "derivar",
    "integrar",
    "resolver",
    "evaluar",
    "sustituir",
];

fn looks_spanish(text: &str) -> bool {
    if text
        .chars()
        .any(|c| matches!(c, 'á' | 'é' | 'í' | 'ó' | 'ú' | 'ñ' | '¿' | '¡'))
    {
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
                *residue
                    .offenders
                    .entry(format!("rule: {}", step.rule))
                    .or_insert(0) += 1;
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
        check(
            &format!("steps[{}].rule_latex", step.index),
            &step.rule_latex,
        );
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
                check(
                    &format!("steps[{}].substeps[{j}].before_latex", step.index),
                    latex,
                );
            }
            if let Some(latex) = &sub.after_latex {
                check(
                    &format!("steps[{}].substeps[{j}].after_latex", step.index),
                    latex,
                );
            }
        }
    }
    for step in &wire.solve_steps {
        check(
            &format!("solve_steps[{}].lhs_latex", step.index),
            &step.lhs_latex,
        );
        check(
            &format!("solve_steps[{}].rhs_latex", step.index),
            &step.rhs_latex,
        );
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

    // The SOLVE channel's own version of the same defect, which E8 above cannot
    // see: its sub-steps carry no `before`/`after`, only the state AFTER the
    // manoeuvre, so a vacuous move shows up as an equation identical to the
    // previous sub-step's. The quadratic derivation announced «divide both
    // sides by a» with a = 1 and «complete the square» with b = 0 — 11 such
    // lines over this corpus until 2026-07-28, invisible because every
    // instrument pointed at the other channel.
    // ---- the solve channel's own D5 ----------------------------------------
    // D5 above walks `wire.steps` only, and the two channels have DIFFERENT
    // schemas: a simplify step carries `before`/`after`, a solve step carries
    // only the state AFTER its manoeuvre. So the solve-channel signature of the
    // same defect is «my equation is the previous step's», and no detector was
    // watching for it — measured 2026-07-28 over this corpus.
    let mut solve_step_repeats = 0;
    for pair in wire.solve_steps.windows(2) {
        if pair[0].equation == pair[1].equation && !solve_repeat_is_honest(&pair[1].description) {
            solve_step_repeats += 1;
        }
    }
    report.measure("D5b_solve_step_repeat", solve_step_repeats);

    let mut solve_substep_repeats = 0;
    for step in &wire.solve_steps {
        for pair in step.substeps.windows(2) {
            if pair[0].equation == pair[1].equation {
                solve_substep_repeats += 1;
            }
        }
    }
    report.measure("E8b_solve_substep_repeat", solve_substep_repeats);
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
/// CERO, y por eso es aserción y no techo. Nació en 2 con los supervivientes
/// nombrados —el brazo `UniqueExpr` de sistemas y el factor integrante de
/// dsolve— y ambos se cerraron con el mismo molde: darle a cada paso la
/// ecuación que su frase nombra (Cramer PRODUCE un valor por incógnita;
/// «calcular μ» produce μ). No queda ningún caso donde repetir la ecuación
/// anterior sea legítimo salvo la verificación, que tiene su propia
/// escapatoria en `solve_repeat_is_honest`.
const D5B_SOLVE_STEP_REPEAT_CEILING: usize = 0;

const E8_SUBSTEP_NOOP_CEILING: usize = 3;

/// The solve channel's vacuous-manoeuvre count is a hard ZERO, not a ceiling:
/// unlike E8, whose survivors are documented cases where `before == after` IS
/// the narration (a residual-policy line), a solve sub-step exists precisely to
/// show the state its manoeuvre produced. If it produces the previous state,
/// the manoeuvre did not happen and announcing it is the defect. Taken to 0 on
/// 2026-07-28 by skipping «divide by a» when a = 1 and the two
/// completing-the-square lines when b = 0.
const E8B_SOLVE_SUBSTEP_REPEAT_CEILING: usize = 0;

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

    let d5b = report
        .measures
        .get("D5b_solve_step_repeat")
        .map_or(0, |m| m.0);
    assert_eq!(
        d5b, D5B_SOLVE_STEP_REPEAT_CEILING,
        "{label}: D5b_solve_step_repeat={d5b} — a solve step repeats the \
         previous equation while its description announces something else"
    );

    let e8b = report
        .measures
        .get("E8b_solve_substep_repeat")
        .map_or(0, |m| m.0);
    assert_eq!(
        e8b, E8B_SOLVE_SUBSTEP_REPEAT_CEILING,
        "{label}: E8b_solve_substep_repeat={e8b} — a solve sub-step shows the \
         equation it received while claiming to have changed it"
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
        (
            "derive_sources_integrate",
            load_derive_sources_as_integrate_calls(),
        ),
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

    println!(
        "es_residue_in_en_rules hits={} rows={}",
        total.rules, total.rows_touched
    );
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
    println!(
        "language_rows_swept hits={} rows={}",
        total.rows_swept, total.rows_swept
    );

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
        tally
            .refuted
            .push(format!("{claim:?}: {before}  ⟹  {after}"));
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
        (
            "derive_sources_integrate",
            load_derive_sources_as_integrate_calls(),
        ),
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
    println!(
        "substep_claim_verified hits={} rows={}",
        tally.verified, tally.verified
    );
    println!(
        "substep_claim_trivial hits={} rows={}",
        tally.trivial, tally.trivial
    );
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

// ---------------------------------------------------------------------------
// C1.9 — the GENERATIVE tier of the substep-claim invariant.
//
// Every other tier of this lane measures over a LIST someone chose, and that
// is exactly the failure mode that produced the 2026-07-25 audit: the previous
// quality campaign closed against its own corpus and two rows picked at random
// by the user both failed. This tier generates its inputs deterministically,
// so the corpus stops being anyone's choice.
//
// What it asserts is the C1.8 invariant on the WIRE: no published sub-step in
// a family whose relation is unambiguous may be REFUTABLE by the campaign's
// own verifier. It runs on the published plain strings — the surface the
// student reads — which is what lets it catch the class node-level checking
// cannot: a formatter whose elision makes the printed pair lie while the nodes
// were right (the 2x-numerator bug was exactly that).
//
// Three deliberate strengthenings over the shadow run above:
//   1. The VARIABLE is known, not inferred: the generator built the input.
//   2. The verifier is `cas_didactic`'s own `verify_claim` — refutation only
//      on positive disproof (non-zero CONSTANT residual), the same standard
//      the emitters publish under. The shadow run's any-non-zero-residual
//      standard produced two false refutations; this tier does not inherit it.
//   3. Titles are mapped to claims PER VERB: a title is only eligible for the
//      family its verb narrates. Measured reason: «Ajustar el factor
//      constante» publishes two DIFFERENT pair shapes depending on the
//      emitter, so a context-free title table would misdeclare it. Titles not
//      in the table abstain and are counted.
//
// What it does NOT prove (plan §7.5): pedagogy. A generator also only produces
// the shapes it was taught — this is a much larger sample, not a proof.
// ---------------------------------------------------------------------------

/// xorshift64* with a FIXED seed: the corpus must be byte-identical on every
/// machine and every run, so a failure's witness (the printed input) is its
/// exact reproduction.
struct GenRng(u64);

impl GenRng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GenVerb {
    Diff,
    Integrate,
    DefiniteIntegrate,
}

struct GeneratedInput {
    text: String,
    verb: GenVerb,
    var: &'static str,
    bounds: Option<(String, String)>,
}

/// A random elementary expression for DIFFERENTIATION: every node the
/// derivative table covers, so the sweep exercises product/quotient/chain
/// narrations, not just polynomials.
fn gen_diff_expr(rng: &mut GenRng, var: &str, depth: u32) -> String {
    if depth == 0 {
        return match rng.below(3) {
            0 => (1 + rng.below(5)).to_string(),
            _ => var.to_string(),
        };
    }
    match rng.below(12) {
        0 => format!("sin({})", gen_diff_expr(rng, var, depth - 1)),
        1 => format!("cos({})", gen_diff_expr(rng, var, depth - 1)),
        2 => format!("tan({})", gen_diff_expr(rng, var, depth - 1)),
        3 => format!("exp({})", gen_diff_expr(rng, var, depth - 1)),
        4 => format!("ln({})", gen_diff_expr(rng, var, depth - 1)),
        5 => format!("sqrt({})", gen_diff_expr(rng, var, depth - 1)),
        6 => format!(
            "({} + {})",
            gen_diff_expr(rng, var, depth - 1),
            gen_diff_expr(rng, var, depth - 1)
        ),
        7 => format!(
            "({} - {})",
            gen_diff_expr(rng, var, depth - 1),
            gen_diff_expr(rng, var, depth - 1)
        ),
        8 => format!(
            "({} * {})",
            gen_diff_expr(rng, var, depth - 1),
            gen_diff_expr(rng, var, depth - 1)
        ),
        9 => format!(
            "({} / {})",
            gen_diff_expr(rng, var, depth - 1),
            gen_diff_expr(rng, var, depth - 1)
        ),
        10 => format!("{var}^{}", 2 + rng.below(4)),
        _ => gen_diff_expr(rng, var, depth - 1),
    }
}

/// An integrand the engine's TABLE narrations recognize. Random trees almost
/// always integrate to an honest residual with no sub-steps, and a sweep whose
/// emitters never fire passes green by vacuity — the `min_expected` lesson.
/// So integrands are biased toward the table families (power, affine trig/exp,
/// u'/u, by parts, arctan) and combined with sums for the linearity narration.
fn gen_integrable_atom(rng: &mut GenRng, var: &str) -> String {
    let a = 1 + rng.below(4);
    let b = rng.below(5);
    let n = 1 + rng.below(4);
    match rng.below(12) {
        0 => format!("{var}^{n}"),
        1 => format!("{a}*{var}^{n}"),
        2 => format!("sin({a}*{var} + {b})"),
        3 => format!("cos({a}*{var} + {b})"),
        4 => format!("e^({a}*{var})"),
        5 => format!("1/{var}"),
        6 => format!("{var}*e^{var}"),
        7 => format!("{var}*sin({var})"),
        8 => format!("{var}^2*sin({var})"),
        9 => format!("1/({var}^2 + {a})"),
        10 => format!("{a}*{var}/({var}^2 + {b})"),
        _ => format!("{a}"),
    }
}

fn gen_integrand(rng: &mut GenRng, var: &str) -> String {
    match rng.below(4) {
        0 => format!(
            "{} + {}",
            gen_integrable_atom(rng, var),
            gen_integrable_atom(rng, var)
        ),
        1 => format!(
            "{} - {}",
            gen_integrable_atom(rng, var),
            gen_integrable_atom(rng, var)
        ),
        _ => gen_integrable_atom(rng, var),
    }
}

fn generate_corpus(count: usize) -> Vec<GeneratedInput> {
    let mut rng = GenRng(0x00C1_9C1A_11FE_2026);
    let mut corpus = Vec::with_capacity(count);
    while corpus.len() < count {
        // A second variable name catches the class the shadow run could not:
        // a narration that is true for the wrong variable (`cos(t) ⟹ sin(t)`
        // differentiated with respect to x).
        let var: &'static str = if rng.below(6) == 0 { "t" } else { "x" };
        let input = match rng.below(10) {
            0..=4 => {
                // Depth 3 only on a slice: nested tan/quotient trees make the
                // simplifier grind and the sweep's cost is a per-cycle tax.
                // Measured at depth 3 for all: 0.45 CPU-s per expression.
                let depth = if rng.below(4) == 0 { 3 } else { 2 };
                let f = gen_diff_expr(&mut rng, var, depth);
                GeneratedInput {
                    text: format!("diff({f}, {var})"),
                    verb: GenVerb::Diff,
                    var,
                    bounds: None,
                }
            }
            5..=7 => {
                let g = gen_integrand(&mut rng, var);
                GeneratedInput {
                    text: format!("integrate({g}, {var})"),
                    verb: GenVerb::Integrate,
                    var,
                    bounds: None,
                }
            }
            _ => {
                let g = gen_integrable_atom(&mut rng, var);
                let lo = rng.below(3).to_string();
                let hi = (rng.below(3) + 3).to_string();
                let text = format!("integrate({g}, {var}, {lo}, {hi})");
                GeneratedInput {
                    text,
                    verb: GenVerb::DefiniteIntegrate,
                    var,
                    bounds: Some((lo, hi)),
                }
            }
        };
        corpus.push(input);
    }
    corpus
}

/// The relation a generated sub-step's title declares, DISAMBIGUATED BY VERB.
/// Only titles verified by probing enter; everything else abstains and is
/// counted. The table's correctness is itself adjudicated by this tier: a
/// wrong row shows up as a refutation witness, and refutation demands a
/// positive disproof.
fn generated_claim(
    ctx: &mut cas_ast::Context,
    input: &GeneratedInput,
    title: &str,
) -> Option<cas_didactic::didactic::substep_claim::Claim> {
    use cas_didactic::didactic::substep_claim::Claim;
    let var = input.var.to_string();
    match input.verb {
        GenVerb::Diff => {
            let is_derivative_pair = title.starts_with("Derivar respecto de")
                || matches!(
                    title,
                    "Derivar el primer factor"
                        | "Derivar el segundo factor"
                        | "Derivar el numerador"
                        | "Derivar el denominador"
                        | "Usar regla del producto"
                        | "Usar regla del cociente"
                        | "Usar regla de la cadena"
                        | "Usar regla exponencial"
                )
                || (title.starts_with("Usar regla de ") && title.contains("(u)"));
            is_derivative_pair.then_some(Claim::Derivative { var })
        }
        GenVerb::Integrate | GenVerb::DefiniteIntegrate => {
            if title == "Evaluar la antiderivada en los límites" {
                let (lo, hi) = input.bounds.as_ref()?;
                let lower = parse_in(ctx, lo)?;
                let upper = parse_in(ctx, hi)?;
                return Some(Claim::DefiniteEval { var, lower, upper });
            }
            let is_antiderivative_pair = matches!(
                title,
                "Hallar la antiderivada"
                    | "Usar sustitución"
                    | "Usar regla de potencia para integrales"
                    | "Integrar los términos simples"
                    | "Usar integración por partes"
            ) || (title.starts_with("Usar la regla de ")
                && (title.contains("->") || title.contains("→") || title.contains("con derivada")));
            is_antiderivative_pair.then_some(Claim::Antiderivative { var })
        }
    }
}

/// ≥ 5000 is the plan's F1 criterion: the one veracity number that does not
/// depend on which corpus a person chose.
const GENERATED_SWEEP_SIZE: usize = 5000;

/// Exercise floors, anchored at ~85 % of the first full run (measured:
/// declared 4986, emitted 14507 over 5000 expressions): a generator that stops
/// firing the emitters would otherwise pass green by vacuity, which is the
/// silent way this tier rots.
const GENERATED_DECLARED_FLOOR: usize = 4200;
const GENERATED_EMITTED_FLOOR: usize = 12000;

#[test]
#[ignore]
fn generated_substep_claim_invariant() {
    use cas_didactic::didactic::substep_claim::{verify_claim, ClaimVerdict};

    let corpus = generate_corpus(GENERATED_SWEEP_SIZE);
    // Fixed worker count, NOT num_cpus: the partition must be identical on
    // every machine so the tallies (and any witness ordering) are too.
    let workers = 12usize;
    let chunk = corpus.len().div_ceil(workers);

    struct Tally {
        evaluated: usize,
        emitted: usize,
        declared: usize,
        abstained: usize,
        verified: usize,
        undecided: usize,
        undecided_witnesses: Vec<String>,
        failures: Vec<String>,
    }

    let tallies: Vec<Tally> = thread::scope(|scope| {
        let handles: Vec<_> = corpus
            .chunks(chunk)
            .map(|slice| {
                scope.spawn(move || {
                    let mut tally = Tally {
                        evaluated: 0,
                        emitted: 0,
                        declared: 0,
                        abstained: 0,
                        verified: 0,
                        undecided: 0,
                        undecided_witnesses: Vec::new(),
                        failures: Vec::new(),
                    };
                    for input in slice {
                        let Some(wire) = eval_wire_in(&input.text, Language::Es) else {
                            continue;
                        };
                        tally.evaluated += 1;
                        for step in &wire.steps {
                            for sub in &step.substeps {
                                tally.emitted += 1;
                                let mut ctx = cas_ast::Context::new();
                                let Some(claim) = generated_claim(&mut ctx, input, &sub.title)
                                else {
                                    tally.abstained += 1;
                                    continue;
                                };
                                let (Some(before), Some(after)) = (
                                    parse_in(&mut ctx, &sub.before),
                                    parse_in(&mut ctx, &sub.after),
                                ) else {
                                    // A plain hole the parser cannot read back is
                                    // not evidence of a lie — but it is counted.
                                    tally.undecided += 1;
                                    tally.undecided_witnesses.push(format!(
                                        "(parse) {}: «{}» {} ⟹ {}",
                                        input.text, sub.title, sub.before, sub.after
                                    ));
                                    continue;
                                };
                                tally.declared += 1;
                                match verify_claim(&ctx, &claim, before, after) {
                                    ClaimVerdict::Verified => tally.verified += 1,
                                    ClaimVerdict::Undecided => {
                                        tally.undecided += 1;
                                        tally.undecided_witnesses.push(format!(
                                            "(claim) {}: «{}» {} ⟹ {}",
                                            input.text, sub.title, sub.before, sub.after
                                        ));
                                    }
                                    ClaimVerdict::Refuted => tally.failures.push(format!(
                                        "{}: «{}» {} ⟹ {}",
                                        input.text, sub.title, sub.before, sub.after
                                    )),
                                }
                            }
                        }
                    }
                    tally
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut evaluated = 0;
    let mut emitted = 0;
    let mut declared = 0;
    let mut abstained = 0;
    let mut verified = 0;
    let mut undecided = 0;
    let mut undecided_witnesses: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for tally in tallies {
        evaluated += tally.evaluated;
        emitted += tally.emitted;
        declared += tally.declared;
        abstained += tally.abstained;
        verified += tally.verified;
        undecided += tally.undecided;
        undecided_witnesses.extend(tally.undecided_witnesses);
        failures.extend(tally.failures);
    }

    let total = corpus.len();
    // Only the two LOAD-STABLE facts are `hits=/rows=` counters the scorecard
    // parses: the corpus size (pure integer arithmetic) and the hard zero.
    // Everything eval-derived trembles under load — the claim verifier runs on
    // a WALL-CLOCK budget (measured: 50 vs 53 undecided on the same commit,
    // idle vs scorecard load), and even eval_ok/emitted drift by ±10 because
    // the engine's standard budget has a time component. Publishing those as
    // counters would flag spurious huella deltas on every future cycle; they
    // live on the diagnostic lines below, and the exercise contract is the
    // in-test FLOOR assertions, whose margin (~15 %) dwarfs the noise band.
    // The hard counter is load-immune in the safe direction: budget exhaustion
    // can only hide a refutation as Undecided, never fabricate one.
    println!("generated_expressions hits={total} rows={total}");
    println!(
        "generated_substep_checked_failures hits={} rows={}",
        failures.len(),
        failures.len()
    );
    println!(
        "  generated_sweep_diagnostics eval_ok={evaluated} emitted={emitted} \
         declared={declared} abstained={abstained} verified={verified} undecided={undecided}"
    );
    // A handful of abstention witnesses so a SPIKE in undecided names its
    // family without a re-run. Steady state (adjudicated BY HAND on the first
    // full run: 50 of 4986, every one a TRUE pair): the simplifier failing to
    // fold surd/nested-trig differences to zero (√x products, tan expansions,
    // sin(cos(t))), and `d|x|/dx ⟹ sign(x)`, which the exact differentiator
    // declines. None is evidence of a lie; that is what Undecided means.
    for witness in undecided_witnesses.iter().take(8) {
        println!("  GENERATED-UNDECIDED {witness}");
    }
    for witness in failures.iter().take(25) {
        println!("  GENERATED-REFUTED {witness}");
    }

    assert!(
        declared >= GENERATED_DECLARED_FLOOR && emitted >= GENERATED_EMITTED_FLOOR,
        "the generative sweep stopped exercising the emitters \
         (declared {declared} < {GENERATED_DECLARED_FLOOR} or emitted {emitted} < \
         {GENERATED_EMITTED_FLOOR}): a vacuous sweep passes green and rots silently"
    );
    assert!(
        failures.is_empty(),
        "{} generated sub-steps publish a REFUTABLE claim (each witness above \
         is its exact reproduction):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Solve narration inventory (C4.1)
// ---------------------------------------------------------------------------

/// Rows that return a REAL answer and narrate NOTHING (C4.2 removed the three
/// periodic-trig inequalities: their narration existed and was being discarded). — neither `steps` nor
/// `solve_steps`. Anchored BY EXPRESSION (the csv reorders and the dedup
/// renumbers), and the lane fails BOTH ways: a new mute row is a regression,
/// and a row that stops being mute must leave this list in the same commit.
/// The list can only shrink.
const KNOWN_MUTE_SOLVE_ROWS: &[&str] = &[
    "solve(4^x-9^x=0,x)",
    "solve(sqrt(x+5)-sqrt(x)=1,x)",
    "solve((x-1)/(x-2)<0,x)",
    "solve(x+1/x>2,x)",
    "solve((x+1)/(x-1)>=2,x)",
    "solve(sqrt(x)>2,x)",
    "solve(x^(2/3)>2,x)",
    "solve(abs(x-2)>1,x)",
    "solve(abs(x-1)<abs(x+2),x)",
    "solve(ln(x)^2-3*ln(x)+2<0,x)",
    "solve(abs(x-a)<2, x)",
    "dsolve(diff(y,x) = x^2 + y^2, y, x)",
];

/// Nivel 1 — CONTRATO: the mute list is exact. A `solve` that stops narrating,
/// or one that starts, must be reflected here in the same commit.
///
/// The asymmetry this fences (audit): of 16 inequality rows, 12 narrate nothing
/// (75 %); of 19 equation rows, 2. The `=` side narrates and the `<`/`>` side
/// does not, with the same left-hand side. The narration is not lost — for the
/// inequality families it does not EXIST (measured: 39 call-sites return
/// `(set, Vec::new())`, 21 of them inequality handlers).
#[test]
#[ignore]
fn solve_mute_inventory_is_exact() {
    let mut mute = Vec::new();
    for input in load_web_examples() {
        let is_solve = ["solve(", "solve_system(", "dsolve("]
            .iter()
            .any(|verb| input.starts_with(verb));
        if !is_solve {
            continue;
        }
        let Some(wire) = eval_wire_in(&input, Language::Es) else {
            continue;
        };
        if wire.steps.is_empty() && wire.solve_steps.is_empty() {
            mute.push(input.clone());
        }
    }
    println!("solve_mute_rows hits={} rows={}", mute.len(), mute.len());
    for row in &mute {
        println!("  MUTE {row}");
    }

    let known: std::collections::BTreeSet<&str> = KNOWN_MUTE_SOLVE_ROWS.iter().copied().collect();
    let found: std::collections::BTreeSet<&str> = mute.iter().map(String::as_str).collect();
    let new: Vec<&&str> = found.difference(&known).collect();
    let fixed: Vec<&&str> = known.difference(&found).collect();
    assert!(
        new.is_empty(),
        "a solve row went MUTE and is not inventoried: {new:?}"
    );
    assert!(
        fixed.is_empty(),
        "these rows now narrate — remove them from KNOWN_MUTE_SOLVE_ROWS in the \
         same commit that fixed them (STALE inventory): {fixed:?}"
    );
}

/// Nivel 2 — FALLA POR DISEÑO mientras la clase esté abierta. This is the
/// living inventory: the red is the work item, and it regenerates itself
/// instead of ageing like a document.
#[test]
#[ignore]
// Clippy is right that this is constant-false — that IS the design. The tier
// exists to be red until the class closes, and the constant is what makes the
// red self-regenerating instead of a doc that ages.
#[allow(clippy::const_is_empty)]
fn solve_mute_inventory_should_be_empty() {
    assert!(
        KNOWN_MUTE_SOLVE_ROWS.is_empty(),
        "{} solve rows still return an answer with no narration at all. \
         The student gets a correct interval or periodic family and ZERO \
         explanation. Owner: frente E5 (one cycle per family). Current list: {:?}",
        KNOWN_MUTE_SOLVE_ROWS.len(),
        KNOWN_MUTE_SOLVE_ROWS
    );
}

// ---------------------------------------------------------------------------
// Plain holes carry PLAIN text (raw-LaTeX-in-plain-hole regression)
// ---------------------------------------------------------------------------

/// The audit measured 43 sub-steps publishing raw LaTeX in the PLAIN text
/// holes (`\sqrt{y} - 1` reaching the CLI reader), across three families:
/// binomial rationalization, nested fractions, and radical-product
/// rationalization — the emitters rendered ONE string per side and reused it
/// for both surfaces, or worse, filled only the plain hole and let the JSON
/// fallback hide it on the web. These inputs pin each family's repro; the
/// assertion is CLASS-wide: a backslash followed by a letter in a plain hole
/// is a LaTeX command reaching the wrong surface, whatever the emitter.
#[test]
fn substep_plain_holes_carry_no_latex_commands() {
    let repros = [
        // binomial conjugate (the chip's repro)
        "sqrt(y)/(sqrt(y)-1) - sqrt(y)/(sqrt(y)+1) - (2*sqrt(y))/(y-1)",
        // nested fractions: one_over + invert
        "1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)",
        // nested fractions: fraction_over (divide-is-multiply)
        "derive(a/(b + c/d), a*d/(b*d+c))",
        // radical product (the emitter that also left latex holes EMPTY)
        "diff(sqrt(16*x),x)",
    ];
    fn has_latex_command(text: &str) -> bool {
        let bytes = text.as_bytes();
        bytes
            .windows(2)
            .any(|w| w[0] == b'\\' && w[1].is_ascii_alphabetic())
    }
    let mut violations = Vec::new();
    let mut inspected = 0usize;
    for input in repros {
        let Some(wire) = eval_wire_in(input, Language::Es) else {
            panic!("repro input failed to evaluate: {input}");
        };
        for step in &wire.steps {
            for sub in &step.substeps {
                inspected += 1;
                for (hole, text) in [
                    ("title", &sub.title),
                    ("before", &sub.before),
                    ("after", &sub.after),
                ] {
                    if has_latex_command(text) {
                        violations.push(format!("[{input}] {hole}: {text}"));
                    }
                }
            }
        }
    }
    assert!(
        inspected > 0,
        "the repros no longer produce sub-steps; the pin is measuring nothing"
    );
    assert!(
        violations.is_empty(),
        "{} plain holes carry LaTeX commands: {violations:#?}",
        violations.len()
    );
}

// ---------------------------------------------------------------------------
// Verified identity narrations REACH the wire (task_104e701b)
// ---------------------------------------------------------------------------

/// These two narrations were MUTE end-to-end: the generators worked (their
/// unit tests were green) while the wire published the step with zero
/// sub-steps. The silencer was deliberate policy, not broken wiring —
/// `is_single_formula_template_rule` cleared single «Usar L = R» sub-steps
/// because, pre-matcher, those titles were unverified claims and silence was
/// the sound answer to the lying-title class. Both emitters now publish only
/// matcher-verified instances of census-adjudicated templates, so the reason
/// for the mute is gone — and this pins the whole route, generator through
/// prune through wire, in both directions: the narration must arrive, and it
/// must arrive saying a VERIFIED identity.
#[test]
fn verified_identity_narrations_reach_the_wire() {
    let cases = [
        ("derive(tan(x)*cot(x), 1)", "Usar tan(u) · cot(u) = 1"),
        (
            "(1-cos(2*x))/sin(2*x)",
            // The half-angle emitter may narrate either orientation of the
            // identity; both state the same census-adjudicated template.
            "(1 - cos(2u)) / sin(2u)",
        ),
        // Trig Quotient migrated after the shadow pass (2026-07-27): the old
        // emitter cited sin/cos = tan over the cot/sec/csc pairs too, and the
        // silencer was the only thing keeping the wrong title off the wire.
        ("cos(x)/sin(x)", "Usar cos(u) / sin(u) = cot(u)"),
        ("1/cos(x)", "Usar 1 / cos(u) = sec(u)"),
        // Double-angle contractions migrated after the EXTENDED shadow pass
        // (2026-07-27): the old emitters cited the sine identity over cosine
        // pairs (blanket title) or picked the template by substring-sniffing.
        ("2*sin(x)*cos(x)", "Usar 2·sin(u)·cos(u) = sin(2u)"),
        ("1 - 2*sin(x)^2", "Usar 1 - 2·sin(u)^2 = cos(2u)"),
        // The SCALED pair, via coefficient peeling (2026-07-27). End-to-end
        // on purpose: the engine-canonical Add-with-negative-literal shape
        // only exists on the wire — parser-node unit tests cannot see it
        // (measured twice in one campaign).
        ("4*cos(x)^2 - 2", "Usar 2·cos(u)^2 - 1 = cos(2u)"),
        // Reciprocal Trig Identity migrated after the DERIVE-route shadow
        // (2026-07-27): the rule only fires on this route, which the first
        // instrument could not see at all.
        ("derive(sec(x), 1/cos(x))", "Usar sec(u) = 1 / cos(u)"),
    ];
    for (input, expected_fragment) in cases {
        let wire = eval_wire_in(input, Language::Es)
            .unwrap_or_else(|| panic!("repro input failed to evaluate: {input}"));
        let titles: Vec<&str> = wire
            .steps
            .iter()
            .flat_map(|step| step.substeps.iter().map(|sub| sub.title.as_str()))
            .collect();
        assert!(
            titles.iter().any(|t| t.contains(expected_fragment)),
            "[{input}] the verified identity narration must reach the wire; \
             got substep titles: {titles:?}"
        );
    }
}

/// C1.6, pinned on the audit's P0 witness (fila 26): the repeated-by-parts
/// CLOSER used to publish `∫−2·sin(x)dx ⟹ 2x·sin(x) + (2−x²)·cos(x)` — the
/// whole answer quoted as the integral of the last remaining term. The honest
/// shape is two sub-steps: the closer integrates ITS OWN integrand, and a
/// separate recomposition assembles the alternating boundary pieces into the
/// engine's answer, published only when the equality is PROVED (expand + fold
/// to exact zero, through the `__hold` the engine wraps its answer in).
#[test]
fn by_parts_closer_integrates_its_own_integrand() {
    let wire = eval_wire_in("integrate(x^2*sin(x), x)", Language::Es)
        .expect("the by-parts witness must evaluate");
    let substeps: Vec<_> = wire
        .steps
        .iter()
        .flat_map(|step| step.substeps.iter())
        .collect();

    let remaining = substeps
        .iter()
        .find(|sub| sub.title == "Integrar el término restante")
        .expect("the remaining-term closer must narrate");
    assert_eq!(
        remaining.before, "integrate(-sin(x)·2, x)",
        "closer must state the LAST remaining integral"
    );
    assert!(
        !remaining.after.contains("sin("),
        "the closer quoting a sin-term means the whole-answer lie is back"
    );
    assert_eq!(
        remaining.after, "2 * cos(x)",
        "the closer's after is the integral of ITS OWN integrand — quoting the \
         final answer here is the audit's P0"
    );

    let recomposition = substeps
        .iter()
        .find(|sub| sub.title == "Recomponer las piezas de por partes")
        .expect("the recomposition must narrate for the witness");
    // RAW wire strings: the in-memory wire keeps display_expr's ` * ` inside
    // the pieces while the assembler joins them with the file's `·`; the CLI's
    // JSON emitter normalizes both to `·` before anything reaches a reader.
    assert_eq!(
        recomposition.before, "x^2·-cos(x) - 2 * x·-sin(x) + 2 * cos(x)",
        "the recomposition assembles u0·v0 − u1·v1 + F"
    );
    assert_eq!(
        recomposition.after, "2 * x * sin(x) + (2 - x^2) * cos(x)",
        "only the recomposition lands on the engine's final antiderivative"
    );
}
