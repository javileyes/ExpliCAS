//! Tests del orquestador, extraídos del módulo (P1).
//!
//! Vivían como `mod tests` inline dentro de `orchestrator.rs`, donde
//! eran 12.227 de sus 42.307 líneas.

use super::*;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

fn render(ctx: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

fn simplify_render(input: &str) -> String {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr =
        parse(input, &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    render(&simplifier.context, rewritten)
}

mod fractions;
mod general;
mod hyperbolic;
mod logs_exp;
mod pairing;
mod radicals_powers;
mod trig;
mod trig_angles;
mod zero_detection;
