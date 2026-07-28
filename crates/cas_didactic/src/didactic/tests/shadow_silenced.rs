//! Shadow pass over the SILENCED template rules — paso 0 of the matcher
//! migration queue (C1.8).
//!
//! `is_single_formula_template_rule` mutes the single «Usar L = R» sub-steps
//! of ~19 rules because, pre-matcher, those titles were unverified claims and
//! silence was the sound answer. Two rules already migrated to
//! `named_identity_substep` and speak again. The ledger's condition for the
//! rest: NO further migration until a shadow pass measures, per rule, whether
//! the instance↔template matcher would actually recognize the pairs the rule
//! produces — the matcher is incomplete by design, and migrating an emitter
//! whose instances it cannot see DELETES narration (the 51-substeps
//! precedent).
//!
//! This test is that instrument. It is rule-agnostic on the template side: the
//! census (`SCHEMATIC_IDENTITIES`, Proven rows only) is the template bank, and
//! a pair counts as covered when ANY proven template instantiates it — whole
//! pair or single-site rewrite, either orientation. Per-rule output:
//!   pairs seen · covered by census · coverage rate · uncovered samples
//! The uncovered samples name the census rows a migration would first need.
//!
//! Run manually (`--ignored`): this measures, it does not gate.

use crate::didactic::enrichment_pipeline::is_single_formula_template_rule;
use crate::didactic::substep::matching::{
    match_instance_structural, match_rewrite, parse_template, ParsedTemplate,
};
use crate::didactic::substep::schema::{SchemaStatus, SCHEMATIC_IDENTITIES};
use cas_ast::Context;
use cas_solver::runtime::{to_display_steps, Simplifier};
use std::collections::BTreeMap;

fn display(ctx: &Context, id: cas_ast::ExprId) -> String {
    format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
}

fn load_corpus() -> Vec<String> {
    let mut exprs = Vec::new();
    let web = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../web/examples.csv"
    ));
    for line in web.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let mut cols = line.splitn(3, ',');
        let _id = cols.next();
        if let Some(expr) = cols.next() {
            let expr = expr.trim();
            if !expr.is_empty() {
                exprs.push(expr.to_string());
            }
        }
    }
    let pairs = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cas_solver/tests/identity_pairs.csv"
    ));
    for line in pairs.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(expr) = trimmed.split(',').next() {
            if !expr.is_empty() {
                exprs.push(expr.to_string());
            }
        }
    }
    // The derive corpus's SOURCE column: the first sweep missed every rule
    // that only fires on these shapes (Reciprocal Trig Identity produced ZERO
    // pairs from web+identity while the derive audit exercises it constantly)
    // — a corpus blind spot is a rule the shadow silently reports as
    // unmigratable.
    let derive = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cas_solver/tests/derive_pairs.csv"
    ));
    for line in derive.lines().skip(1) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(expr) = trimmed.split(',').nth(2) {
            let expr = expr.trim();
            if !expr.is_empty() {
                exprs.push(expr.to_string());
            }
        }
    }
    exprs
}

fn proven_templates() -> Vec<(&'static str, &'static str, ParsedTemplate)> {
    SCHEMATIC_IDENTITIES
        .iter()
        .filter(|schema| matches!(schema.status, SchemaStatus::Proven))
        .filter_map(|schema| {
            parse_template(schema.lhs, schema.rhs).map(|t| (schema.lhs, schema.rhs, t))
        })
        .collect()
}

#[derive(Default)]
struct RuleTally {
    pairs: usize,
    covered: usize,
    /// Whole-pair STRUCTURAL instantiations only — the pass the migrated
    /// emitters trust first. `covered > covered_structural` flags pairs that
    /// only the DIRECTED mode accepts, and the 2026-07-27 probe showed that
    /// can be pure noise: sec²⟹1+tan² was "covered" via the factorial row
    /// ((k+1)·k!/k! = k+1), an engine-equal but title-truth-useless match.
    covered_structural: usize,
    uncovered_samples: Vec<String>,
}

/// MEASURE, do not gate: census coverage of the pairs each silenced rule
/// publishes over the corpus. The per-rule rates are the migration decision
/// table.
#[test]
#[ignore]
fn shadow_silenced_template_instance_rates() {
    let corpus = load_corpus();
    let templates = proven_templates();
    println!(
        "shadow corpus: {} expressions, census bank: {} proven templates",
        corpus.len(),
        templates.len()
    );

    let mut tallies: BTreeMap<String, RuleTally> = BTreeMap::new();

    for expr in &corpus {
        let mut simplifier = Simplifier::with_default_rules();
        simplifier.set_collect_steps(true);
        let Ok(parsed) = cas_parser::parse(expr, &mut simplifier.context) else {
            continue;
        };
        let (_result, raw_steps) = simplifier.simplify(parsed);
        let steps = to_display_steps(raw_steps);
        let ctx = &simplifier.context;
        for step in &steps {
            if !is_single_formula_template_rule(step.rule_name.as_str()) {
                continue;
            }
            let before = step.before_local().unwrap_or(step.before);
            let after = step.after_local().unwrap_or(step.after);
            let tally = tallies.entry(step.rule_name.to_string()).or_default();
            tally.pairs += 1;
            let covered = templates.iter().any(|(_, _, template)| {
                match_rewrite(template, ctx, before, after).is_some()
                    || match_rewrite(template, ctx, after, before).is_some()
            });
            if templates.iter().any(|(_, _, template)| {
                match_instance_structural(template, ctx, before, after).is_some()
                    || match_instance_structural(template, ctx, after, before).is_some()
            }) {
                tally.covered_structural += 1;
            }
            if covered {
                tally.covered += 1;
                for (lhs, rhs, template) in &templates {
                    if match_instance_structural(template, ctx, before, after).is_some()
                        || match_instance_structural(template, ctx, after, before).is_some()
                    {
                        println!(
                            "    PROBE-WHICH {} :: {} ⟹ {} :: BY {lhs} = {rhs}",
                            step.rule_name,
                            display(ctx, before),
                            display(ctx, after)
                        );
                    }
                }
            } else if tally.uncovered_samples.len() < 3 {
                tally.uncovered_samples.push(format!(
                    "{} ⟹ {}",
                    display(ctx, before),
                    display(ctx, after)
                ));
            }
        }
    }

    let mut total_pairs = 0usize;
    let mut total_covered = 0usize;
    for (rule, tally) in &tallies {
        total_pairs += tally.pairs;
        total_covered += tally.covered;
        let rate = if tally.pairs == 0 {
            0.0
        } else {
            100.0 * tally.covered as f64 / tally.pairs as f64
        };
        println!(
            "SHADOW-RULE {rule}: pairs={} covered={} structural={} rate={rate:.0}%",
            tally.pairs, tally.covered, tally.covered_structural
        );
        for sample in &tally.uncovered_samples {
            println!("    UNCOVERED {sample}");
        }
    }
    println!("SHADOW-TOTAL pairs={total_pairs} covered={total_covered}");

    // The instrument must EXERCISE: a corpus that stops producing silenced
    // pairs measures nothing and would read as "everything migrated".
    assert!(
        total_pairs >= 20,
        "the shadow corpus produced only {total_pairs} silenced pairs — the \
         instrument is no longer exercising the queue"
    );
}

/// The DERIVE route: `Reciprocal Trig Identity` and friends are produced by
/// `derive_command`, never by the plain simplifier, so the simplify-route
/// shadow reports them as unmigratable-by-absence. This tier evaluates
/// `derive(source, target)` through the real eval pipeline and reads the RAW
/// steps inside the step-payload collector — BEFORE the prune that deletes
/// silenced sub-steps, and with the live Context the matcher needs.
#[test]
#[ignore]
fn shadow_silenced_template_instance_rates_derive_route() {
    use cas_api_models::{
        EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
        EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalNumericDisplay,
        EvalStepsMode, EvalValueDomain,
    };
    use cas_session::eval::{evaluate_eval_command_pretty_with_session, EvalCommandConfig};
    use std::cell::RefCell;

    let derive_rows: Vec<(String, String)> = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cas_solver/tests/derive_pairs.csv"
    ))
    .lines()
    .skip(1)
    .filter_map(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return None;
        }
        let mut cols = trimmed.split(',');
        let _id = cols.next()?;
        let _family = cols.next()?;
        let source = cols.next()?.trim().to_string();
        let target = cols.next()?.trim().to_string();
        (!source.is_empty() && !target.is_empty()).then_some((source, target))
    })
    .collect();

    let templates = proven_templates();
    let tallies: RefCell<BTreeMap<String, RuleTally>> = RefCell::new(BTreeMap::new());

    for (source, target) in &derive_rows {
        let expr = format!("derive({source}, {target})");
        let config = EvalCommandConfig {
            expr: &expr,
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
        };
        let _json = evaluate_eval_command_pretty_with_session(
            None,
            config,
            crate::Language::Es,
            |steps, events, context, steps_mode| {
                let mut tallies = tallies.borrow_mut();
                for step in steps {
                    if !is_single_formula_template_rule(step.rule_name.as_str()) {
                        continue;
                    }
                    let before = step.before_local().unwrap_or(step.before);
                    let after = step.after_local().unwrap_or(step.after);
                    let tally = tallies.entry(step.rule_name.to_string()).or_default();
                    tally.pairs += 1;
                    let covered = templates.iter().any(|(_, _, template)| {
                        match_rewrite(template, context, before, after).is_some()
                            || match_rewrite(template, context, after, before).is_some()
                    });
                    if templates.iter().any(|(_, _, template)| {
                        match_instance_structural(template, context, before, after).is_some()
                            || match_instance_structural(template, context, after, before).is_some()
                    }) {
                        tally.covered_structural += 1;
                    }
                    if covered {
                        tally.covered += 1;
                        for (lhs, rhs, template) in &templates {
                            if match_instance_structural(template, context, before, after).is_some()
                                || match_instance_structural(template, context, after, before)
                                    .is_some()
                            {
                                println!(
                                    "    PROBE-WHICH {} :: {} ⟹ {} :: BY {lhs} = {rhs} :: DESC {}",
                                    step.rule_name,
                                    display(context, before),
                                    display(context, after),
                                    step.description
                                );
                            }
                        }
                    } else if tally.uncovered_samples.len() < 3 {
                        tally.uncovered_samples.push(format!(
                            "{} ⟹ {}",
                            display(context, before),
                            display(context, after)
                        ));
                    }
                }
                crate::collect_step_payloads_with_events(steps, events, context, steps_mode)
            },
        );
    }

    let tallies = tallies.into_inner();
    let mut total_pairs = 0usize;
    for (rule, tally) in &tallies {
        total_pairs += tally.pairs;
        let rate = if tally.pairs == 0 {
            0.0
        } else {
            100.0 * tally.covered as f64 / tally.pairs as f64
        };
        println!(
            "SHADOW-DERIVE-RULE {rule}: pairs={} covered={} structural={} rate={rate:.0}%",
            tally.pairs, tally.covered, tally.covered_structural
        );
        for sample in &tally.uncovered_samples {
            println!("    UNCOVERED {sample}");
        }
    }
    println!("SHADOW-DERIVE-TOTAL pairs={total_pairs}");
}
