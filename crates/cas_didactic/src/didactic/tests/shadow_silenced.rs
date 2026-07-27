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
use crate::didactic::substep::matching::{match_rewrite, parse_template, ParsedTemplate};
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
            if covered {
                tally.covered += 1;
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
            "SHADOW-RULE {rule}: pairs={} covered={} rate={rate:.0}%",
            tally.pairs, tally.covered
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
