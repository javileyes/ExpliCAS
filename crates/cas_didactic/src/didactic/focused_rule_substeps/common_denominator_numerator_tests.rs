//! Tests de `focused_rule_substeps`: `common_denominator_numerator_tests`, extraídos del módulo.

use super::generate_add_subtract_fractions_substeps;
use crate::runtime::Step;
use cas_ast::Context;
use cas_parser::parse;

/// The audit's arithmetically-false numerator: «Llevar a denominador
/// común» published `(c + x - b + x) / …` — worth `c − b + 2x` — for a
/// difference whose numerator is `c − b`. Two coupled defects: the lift
/// built `Mul(1, …)` nodes, and both renderers elided the `1 ·` without
/// re-checking the parenthesization their precedence decision assumed.
/// This pins the whole published chain, plain AND LaTeX.
#[test]
fn common_denominator_numerator_keeps_the_subtrahend_grouped() {
    let mut ctx = Context::new();
    let before = parse("1/(b+x) - 1/(c+x)", &mut ctx).expect("parse before");
    let after = parse("(c-b)/(x^2+(b+c)*x+b*c)", &mut ctx).expect("parse after");
    let step = Step::new_compact("desc", "Subtract Fractions", before, after);
    let subs = generate_add_subtract_fractions_substeps(&ctx, &step);
    assert!(!subs.is_empty(), "the narration must exist");
    let first = &subs[0];
    assert_eq!(
        first.after_expr, "(c + x - (b + x)) / ((b + x) * (c + x))",
        "plain numerator must keep the subtrahend grouped"
    );
    let latex = first.after_latex.as_deref().expect("latex populated");
    assert!(
        latex.contains("c + x - (b + x)"),
        "latex numerator must keep the subtrahend grouped: {latex}"
    );
    assert!(
        !first.after_expr.contains("x - b + x"),
        "the 2x lie must not resurface: {}",
        first.after_expr
    );
}
