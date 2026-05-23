use cas_api_models::StepWire;
use cas_solver::runtime::{Engine, EvalAction, EvalOptions, EvalRequest};
use cas_solver_core::engine_events::EngineEvent;
use cas_solver_core::soundness_label::SoundnessLabel;
use cas_solver_core::step_model::{Step, StepMeta};
use cas_solver_core::step_types::{ImportanceLevel, StepCategory};
use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

static STEP_WIRE_ON_CACHE: LazyLock<Mutex<BTreeMap<String, Vec<StepWire>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

const RATIONALIZE_LINEAR_ROOT_EXPR: &str = "1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)";
const PERFECT_SQUARE_ROOT_EXPR: &str = "sqrt(x^2 + 2*x + 1)";

fn eval_output_for(expr: &str) -> (Engine, cas_solver::runtime::EvalOutput) {
    let mut engine = Engine::new();
    let parsed = cas_parser::parse(expr, &mut engine.simplifier.context).expect("parse");
    let output = engine
        .eval_stateless(
            EvalOptions::default(),
            EvalRequest {
                raw_input: expr.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval");
    (engine, output)
}

fn step_payloads_on_for(expr: &str) -> Vec<StepWire> {
    if let Some(cached) = STEP_WIRE_ON_CACHE
        .lock()
        .expect("step wire cache poisoned")
        .get(expr)
        .cloned()
    {
        return cached;
    }

    let (engine, output) = eval_output_for(expr);
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    STEP_WIRE_ON_CACHE
        .lock()
        .expect("step wire cache poisoned")
        .insert(expr.to_string(), steps.clone());

    steps
}

fn synthetic_step(
    rule_name: &str,
    description: &str,
    before: cas_ast::ExprId,
    after: cas_ast::ExprId,
    global_before: cas_ast::ExprId,
    global_after: cas_ast::ExprId,
) -> Step {
    Step {
        description: description.into(),
        rule_name: rule_name.into(),
        before,
        after,
        global_before: Some(global_before),
        global_after: Some(global_after),
        importance: ImportanceLevel::Medium,
        category: StepCategory::Simplify,
        soundness: SoundnessLabel::Equivalence,
        meta: Some(Box::new(StepMeta {
            before_local: Some(before),
            after_local: Some(after),
            ..StepMeta::default()
        })),
    }
}

#[test]
fn step_wire_off_mode_is_empty() {
    let (engine, output) = eval_output_for("x + x");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "off");
    assert!(steps.is_empty());
}

#[test]
fn step_wire_on_mode_matches_deterministically() {
    let (engine, output) = eval_output_for("(x + 2) + (x + 3)");
    let first =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");
    let second =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");
    assert_eq!(first.len(), second.len());
    if let (Some(a), Some(b)) = (first.first(), second.first()) {
        assert_eq!(a.rule, b.rule);
        assert_eq!(a.before, b.before);
        assert_eq!(a.after, b.after);
    }
}

#[test]
fn step_wire_events_fallback_is_used_when_steps_are_missing() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let zero = ctx.num(0);
    let before = ctx.add(cas_ast::Expr::Add(x, zero));

    let steps = cas_didactic::collect_step_payloads_with_events(
        &[],
        &[EngineEvent::RuleApplied {
            rule_name: "Additive Identity".to_string(),
            before: x,
            after: x,
            global_before: Some(before),
            global_after: Some(x),
            is_chained: false,
        }],
        &ctx,
        "on",
    );

    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].rule, "Additive Identity");
    assert_eq!(steps[0].before, "x");
    assert_eq!(steps[0].after, "x");
}

#[test]
fn step_wire_events_fallback_respects_off_mode() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let steps = cas_didactic::collect_step_payloads_with_events(
        &[],
        &[EngineEvent::RuleApplied {
            rule_name: "test".to_string(),
            before: x,
            after: x,
            global_before: None,
            global_after: None,
            is_chained: false,
        }],
        &ctx,
        "off",
    );
    assert!(steps.is_empty());
}

#[test]
fn step_wire_log_fraction_gap_regression_cleans_identity_noise_without_breaking_after_focus() {
    let mut ctx = cas_ast::Context::new();

    let self_cancel_before = cas_parser::parse("1 - 1", &mut ctx).expect("parse");
    let self_cancel_after = ctx.num(0);
    let self_cancel_global_before =
        cas_parser::parse("1 - 1 + sqrt(y)/(sqrt(y)-1)", &mut ctx).expect("parse");
    let self_cancel_global_after =
        cas_parser::parse("sqrt(y)/(sqrt(y)-1)", &mut ctx).expect("parse");

    let expand_before = cas_parser::parse("sqrt(y)*(sqrt(y)-1)", &mut ctx).expect("parse");
    let expand_after = cas_parser::parse("sqrt(y)*sqrt(y) - 1*sqrt(y)", &mut ctx).expect("parse");
    let expand_global_before =
        cas_parser::parse("(sqrt(y)*(sqrt(y)-1))/(y-1)", &mut ctx).expect("parse");
    let expand_global_after =
        cas_parser::parse("(sqrt(y)*sqrt(y) - 1*sqrt(y))/(y-1)", &mut ctx).expect("parse");

    let steps = cas_didactic::collect_step_payloads(
        &[
            synthetic_step(
                "Subtraction Self-Cancel",
                "a - a = 0",
                self_cancel_before,
                self_cancel_after,
                self_cancel_global_before,
                self_cancel_global_after,
            ),
            synthetic_step(
                "Expand",
                "Expand",
                expand_before,
                expand_after,
                expand_global_before,
                expand_global_after,
            ),
        ],
        &ctx,
        "on",
    );

    for step in &steps {
        assert!(
            !step.before.contains(" + 0") && !step.before.starts_with("0 + "),
            "wire before still contains additive zero noise: {}",
            step.before
        );
        assert!(
            !step.after.contains(" + 0") && !step.after.starts_with("0 + "),
            "wire after still contains additive zero noise: {}",
            step.after
        );
        assert!(
            !step.before_latex.contains("+ 0") && !step.before_latex.contains("0 +"),
            "wire before_latex still contains additive zero noise: {}",
            step.before_latex
        );
        assert!(
            !step.after_latex.contains("+ 0") && !step.after_latex.contains("0 +"),
            "wire after_latex still contains additive zero noise: {}",
            step.after_latex
        );
    }

    if let Some(collapse_common_scale) = steps
        .iter()
        .find(|step| step.rule == "Collapse Common-Scale Equivalent Difference")
    {
        assert!(
            !collapse_common_scale
                .after_latex
                .starts_with("{\\color{green}{"),
            "after_latex should not highlight the entire surviving residual: {}",
            collapse_common_scale.after_latex
        );
        assert!(
            collapse_common_scale
                .after_latex
                .contains("\\frac{\\frac{1}{x} - \\frac{1}{y}}{\\frac{y - x}{x\\cdot y}}"),
            "expected surviving residual after removing the zero summand, got: {}",
            collapse_common_scale.after_latex
        );
    } else if let Some(self_cancel) = steps.iter().find(|step| {
        step.rule == "Restar dos expresiones iguales"
            && step.after.contains("sqrt(y)/(sqrt(y) - 1)")
    }) {
        assert!(
            !self_cancel.after_latex.starts_with("{\\color{green}{"),
            "after_latex should not highlight the entire surviving residual: {}",
            self_cancel.after_latex
        );
        assert!(
            self_cancel
                .after_latex
                .contains("\\frac{\\sqrt{y}}{\\sqrt{y} - 1}"),
            "expected surviving y-residual after removing the zero summand, got: {}",
            self_cancel.after_latex
        );
    } else {
        let collapsed_zero_chunk = steps
            .iter()
            .find(|step| {
                step.rule == "Collapse Exact Zero Additive Subexpression"
                    && step.after.contains("sqrt(y)/(sqrt(y) - 1)")
            })
            .expect("expected generic zero-chunk collapse step");
        assert!(
            !collapsed_zero_chunk
                .after_latex
                .starts_with("{\\color{green}{"),
            "after_latex should not highlight the entire surviving residual: {}",
            collapsed_zero_chunk.after_latex
        );
        assert!(
            collapsed_zero_chunk
                .after_latex
                .contains("\\frac{\\sqrt{y}}{\\sqrt{y} - 1}"),
            "expected surviving y-residual after removing the zero summand, got: {}",
            collapsed_zero_chunk.after_latex
        );
    }

    if let Some(self_cancel) = steps.iter().find(|step| {
        step.rule == "Restar dos expresiones iguales"
            && step.after.contains("sqrt(y)/(sqrt(y) - 1)")
    }) {
        assert!(
            !self_cancel.after_latex.starts_with("{\\color{green}{"),
            "after_latex should stay plain when the local zero disappears from the cleaned snapshot: {}",
            self_cancel.after_latex
        );
        assert!(
            self_cancel
                .after_latex
                .contains("\\frac{\\sqrt{y}}{\\sqrt{y} - 1}"),
            "expected surviving y-residual after removing the zero summand, got: {}",
            self_cancel.after_latex
        );
    } else {
        let final_zero_chunk = steps
            .iter()
            .find(|step| {
                step.rule == "Collapse Exact Zero Additive Subexpression"
                    && step.before.contains("sqrt(y)/(sqrt(y) - 1)")
                    && step.after == "0"
            })
            .expect("expected final generic zero-chunk collapse step");
        assert!(
            final_zero_chunk
                .before_latex
                .contains("\\frac{\\sqrt{y}}{\\sqrt{y} - 1}"),
            "expected the remaining y-residual to stay visible before the final collapse, got: {}",
            final_zero_chunk.before_latex
        );
    }

    if let Some(expand_inside_fraction) = steps.iter().find(|step| {
        step.rule == "Expandir la expresión"
            && step.before.contains("(sqrt(y) · (sqrt(y) - 1))/(y - 1)")
    }) {
        assert!(
            expand_inside_fraction
                .before_latex
                .contains("{\\color{red}{\\sqrt{y}\\cdot (\\sqrt{y} - 1)}}"),
            "expected the factorized product to stay highlighted in before_latex, got: {}",
            expand_inside_fraction.before_latex
        );
        assert!(
            expand_inside_fraction
                .after_latex
                .contains("{\\color{green}{\\sqrt{y}\\cdot \\sqrt{y} - 1\\cdot \\sqrt{y}}}")
                || expand_inside_fraction
                    .after_latex
                    .contains("{\\color{green}{\\sqrt{y}\\cdot \\sqrt{y} - \\sqrt{y}}}"),
            "expected the expanded product to stay highlighted in after_latex, got: {}",
            expand_inside_fraction.after_latex
        );
    } else {
        assert!(
            steps.iter().any(|step| {
                step.rule == "Collapse Exact Zero Additive Subexpression"
                    && step.before.contains("sqrt(y)/(sqrt(y) - 1)")
                    && step.before.contains("sqrt(y)/(sqrt(y) + 1)")
            }),
            "expected either the old distributive-fraction narrative or the new direct zero-chunk collapse"
        );
    }
}

#[test]
fn step_wire_pull_constant_from_fraction_highlights_the_rewritten_fraction() {
    let mut ctx = cas_ast::Context::new();
    let before = cas_parser::parse("(2*sqrt(y))/(y-1)", &mut ctx).expect("parse");
    let after = cas_parser::parse("2*(sqrt(y)/(y-1))", &mut ctx).expect("parse");
    let steps = cas_didactic::collect_step_payloads(
        &[
            synthetic_step(
                "Pull Constant From Fraction",
                "Pull Constant From Fraction",
                before,
                after,
                before,
                after,
            ),
            synthetic_step(
                "Pull Constant From Fraction",
                "Pull Constant From Fraction",
                before,
                after,
                before,
                after,
            ),
        ],
        &ctx,
        "on",
    );

    let pull_steps: Vec<_> = steps
        .iter()
        .filter(|step| step.rule == "Sacar constante de una fracción")
        .collect();

    assert_eq!(pull_steps.len(), 2, "expected the two pull-constant steps");

    assert!(
        pull_steps[0]
            .before_latex
            .starts_with("{\\color{red}{\\frac{2\\cdot \\sqrt{y}}{y - 1}}}"),
        "expected the first pull-constant step to highlight the full input fraction, got: {}",
        pull_steps[0].before_latex
    );
    assert!(
        pull_steps[0]
            .after_latex
            .starts_with("{\\color{green}{2\\cdot \\frac{\\sqrt{y}}{y - 1}}}"),
        "expected the first pull-constant step to highlight the full rewritten fraction, got: {}",
        pull_steps[0].after_latex
    );
    assert!(
        pull_steps[1]
            .after_latex
            .starts_with("{\\color{green}{2\\cdot \\frac{\\sqrt{y}}{y - 1}}}"),
        "expected the second pull-constant step to highlight the rewritten fraction, got: {}",
        pull_steps[1].after_latex
    );
}

#[test]
fn step_wire_post_diff_fraction_cleanup_uses_human_visible_rule_titles() {
    let (engine, output) = eval_output_for("diff(arctan(sqrt(x)), x)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let rule_titles: Vec<&str> = steps.iter().map(|step| step.rule.as_str()).collect();

    assert!(
        rule_titles.contains(&"Calcular la derivada"),
        "expected the main diff step to use a human title, got {rule_titles:?}"
    );
    assert!(
        rule_titles.contains(&"Presentar resultado de cálculo en forma compacta"),
        "expected post-diff compact presentation to use a human title, got {rule_titles:?}"
    );
    assert!(
        !rule_titles.contains(&"Symbolic Differentiation")
            && !rule_titles.contains(&"Pull Constant From Fraction")
            && !rule_titles.contains(&"Simplify Multiplication with Division")
            && !rule_titles.contains(&"Present calculus result in compact form"),
        "post-calculus visible trace should not leak internal rule names: {rule_titles:?}"
    );
}

#[test]
fn step_wire_post_diff_trace_drops_adjacent_inverse_fraction_roundtrip() {
    let (engine, output) = eval_output_for("diff(ln(sec(sqrt(x))+tan(sqrt(x))), x)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    for adjacent in steps.windows(2) {
        let [previous, current] = adjacent else {
            continue;
        };
        assert!(
            !(previous.rule == current.rule
                && previous.before == current.after
                && previous.after == current.before),
            "public step trace should not expose an adjacent inverse roundtrip: {previous:?} then {current:?}"
        );
    }

    assert!(
        steps
            .iter()
            .all(|step| step.after != "(1/(sqrt(x) · cos(sqrt(x))))/2"),
        "transient nested fraction presentation should be hidden from the public trace: {steps:?}"
    );

    let final_step = steps.last().expect("expected visible post-diff step");
    assert_eq!(
        final_step.after, "1/(2 · cos(sqrt(x)) · sqrt(x))",
        "post-diff result should put the numeric denominator factor first: {final_step:?}"
    );
    assert!(
        final_step
            .after_latex
            .contains("\\frac{1}{2\\cdot \\cos(\\sqrt{x})\\cdot \\sqrt{x}}"),
        "post-diff latex should put the numeric denominator factor first: {}",
        final_step.after_latex
    );
}

#[test]
fn step_wire_integration_uses_human_visible_rule_title() {
    let (engine, output) = eval_output_for("integrate(1/(2*x+1), x)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let rule_titles: Vec<&str> = steps.iter().map(|step| step.rule.as_str()).collect();

    assert!(
        rule_titles.contains(&"Calcular la integral"),
        "expected the integration step to use a human title, got {rule_titles:?}"
    );
    assert!(
        !rule_titles.contains(&"Symbolic Integration"),
        "integration visible trace should not leak internal rule names: {rule_titles:?}"
    );
}

#[test]
fn step_wire_substeps_preserve_math_latex_for_rationalization_example() {
    let steps = step_payloads_on_for(RATIONALIZE_LINEAR_ROOT_EXPR);

    let all_substep_math: Vec<&str> = steps
        .iter()
        .flat_map(|step| step.substeps.iter())
        .flat_map(|substep| {
            [
                substep.before_latex.as_deref(),
                substep.after_latex.as_deref(),
            ]
        })
        .flatten()
        .collect();

    assert!(
        !all_substep_math.is_empty(),
        "expected didactic substeps for rationalization example"
    );
    assert!(
        all_substep_math
            .iter()
            .any(|latex| latex.contains("\\frac") || latex.contains("\\sqrt")),
        "expected math-like didactic substep content, got: {all_substep_math:?}"
    );
    assert!(
        all_substep_math
            .iter()
            .all(|latex| !latex.starts_with("\\text{\\frac") && !latex.starts_with("\\text{\\sqrt")),
        "didactic wire payload should not wrap math latex in \\\\text{{...}}: {all_substep_math:?}"
    );
}

#[test]
fn step_wire_humanizes_root_notation_in_before_after_text() {
    let steps = step_payloads_on_for(RATIONALIZE_LINEAR_ROOT_EXPR);

    let rationalize_step = steps
        .iter()
        .find(|step| step.rule == "Racionalizar el denominador")
        .expect("expected rationalization step");
    assert!(
        rationalize_step.before.contains("sqrt(x)"),
        "expected wire before text to humanize roots, got: {}",
        rationalize_step.before
    );
    assert!(
        !rationalize_step.before.contains("x^(1/2)"),
        "expected wire before text to avoid fractional-exponent root notation, got: {}",
        rationalize_step.before
    );
}

#[test]
fn step_wire_combine_like_terms_explains_coefficient_sum_without_repeating_step_title() {
    let (engine, output) = eval_output_for("x + x");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Agrupar términos semejantes")
        .expect("expected combine like terms step");

    assert_eq!(
        step.substeps.len(),
        1,
        "expected a single focused didactic substep, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        step.substeps[0].title,
        "Sumar los coeficientes que acompañan a x"
    );
    assert!(
        step.substeps[0]
            .before_latex
            .as_deref()
            .is_some_and(|latex| latex == "1 + 1"),
        "expected the hidden substep to show only the coefficient sum, got: {:?}",
        step.substeps[0].before_latex
    );
    assert!(
        step.substeps[0]
            .after_latex
            .as_deref()
            .is_some_and(|latex| latex == "2"),
        "expected the hidden substep to end in the summed coefficient, got: {:?}",
        step.substeps[0].after_latex
    );
}

#[test]
fn step_wire_phase_shift_substeps_are_structured_without_narrative_lines() {
    let mut ctx = cas_ast::Context::new();
    let local_before = cas_parser::parse("3*sin(x) + 4*cos(x)", &mut ctx).expect("parse");
    let local_after = cas_parser::parse("5*sin(x + arctan(4/3))", &mut ctx).expect("parse");
    let global_before = ctx.add(cas_ast::Expr::Sub(local_before, local_after));
    let global_after = ctx.num(0);

    let step = Step {
        description: "Phase Shift Identity".into(),
        rule_name: "Phase Shift Identity".into(),
        before: local_before,
        after: local_after,
        global_before: Some(global_before),
        global_after: Some(global_after),
        importance: ImportanceLevel::Medium,
        category: StepCategory::Simplify,
        soundness: SoundnessLabel::Equivalence,
        meta: None,
    };
    let steps = cas_didactic::collect_step_payloads(&[step], &ctx, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Aplicar identidad de desfase")
        .expect("expected phase-shift step");
    assert_eq!(step.substeps.len(), 2);
    assert!(step.substeps.iter().all(|substep| substep.lines.is_empty()));
    assert!(
        step.substeps
            .iter()
            .all(|substep| substep.before_latex.is_some() && substep.after_latex.is_some()),
        "phase-shift substeps should always expose before/after expressions, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| (
                substep.title.as_str(),
                substep.before_latex.as_deref(),
                substep.after_latex.as_deref()
            ))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        step.substeps[0].title,
        "Usar a·sin(u) + b·cos(u) = R·sin(u + φ)"
    );
    let cancellation_substep = &step.substeps[1];
    assert_eq!(cancellation_substep.title, "Cancelar términos iguales");
    assert!(
        cancellation_substep
            .before_latex
            .as_deref()
            .is_some_and(
                |latex| latex.contains("5\\cdot \\sin") && latex.contains("- 5\\cdot \\sin")
            ),
        "phase-shift substep should show the concrete cancellation, got: {:?}",
        cancellation_substep.before_latex
    );
    assert_eq!(cancellation_substep.after_latex.as_deref(), Some("0"));
}

#[test]
fn step_wire_path_latex_renders_human_subtractive_products_for_factor_example() {
    let (engine, output) = eval_output_for("factor(a^3*(b - c) + b^3*(c - a) + c^3*(a - b))");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let first_distribute = steps
        .iter()
        .find(|step| step.rule == "Expandir la expresión")
        .expect("expected distributive step");
    let factor_step = steps
        .iter()
        .find(|step| step.rule == "Factor Polynomial")
        .expect("expected factor polynomial step");

    let first_after_latex = first_distribute.after_latex.as_str();
    assert!(
        first_after_latex.contains("{a}^{3}\\cdot b - {a}^{3}\\cdot c")
            || first_after_latex.contains("b\\cdot {a}^{3} - c\\cdot {a}^{3}"),
        "expected human subtraction in highlighted latex, got: {first_after_latex}"
    );
    assert!(
        !first_after_latex.contains("\\cdot -c"),
        "highlighted latex should not keep the negative factor inside the product: {first_after_latex}"
    );

    let factor_before_latex = factor_step.before_latex.as_str();
    assert!(
        !factor_before_latex.contains("- (a\\cdot {b}^{3})")
            && !factor_before_latex.contains("- ({c}^{3}\\cdot b)"),
        "simple subtracted products should not be wrapped in parentheses: {factor_before_latex}"
    );
}

#[test]
fn step_wire_sophie_germain_after_text_and_latex_stay_human() {
    let (engine, output) = eval_output_for("factor(x^4 + 4*y^4)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Factor Polynomial")
        .expect("expected factor step");

    assert!(
        !step.after.contains("  "),
        "expected no repeated spaces in human after text, got: {}",
        step.after
    );
    assert!(
        step.after == "(x^2 + 2 · y^2 + 2 · x · y) · (x^2 + 2 · y^2 - 2 · x · y)",
        "expected tightened human spacing, got: {}",
        step.after
    );
    assert!(
        !step.after_latex.contains("- (2\\cdot x\\cdot y)"),
        "resulting latex should not parenthesize the final simple product: {}",
        step.after_latex
    );
}

#[test]
fn step_wire_small_polynomial_product_emits_hidden_didactic_expansion_and_cancellation() {
    let (engine, output) = eval_output_for("(x - 1)*(x^5 + x^4 + x^3 + x^2 + x + 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Expandir y reagrupar un producto polinómico")
        .expect("expected polynomial product normalize step");

    let titles: Vec<&str> = step
        .substeps
        .iter()
        .map(|substep| substep.title.as_str())
        .collect();
    assert!(
        titles.contains(&"Distribuir cada término del producto")
            && titles.contains(&"Agrupar los términos del mismo grado")
            && titles.contains(&"Los términos intermedios se cancelan por parejas"),
        "expected full didactic substep narrative, got {titles:?}"
    );

    let expansion_substep = step
        .substeps
        .iter()
        .find(|substep| substep.title == "Distribuir cada término del producto")
        .expect("missing expansion substep");
    let joined_sides = format!(
        "{} | {}",
        expansion_substep
            .before_latex
            .as_deref()
            .unwrap_or_default(),
        expansion_substep.after_latex.as_deref().unwrap_or_default()
    );
    assert!(
        joined_sides.contains("{x}^{6} + {x}^{5} + {x}^{4} + {x}^{3} + {x}^{2} + x - {x}^{5} - {x}^{4} - {x}^{3} - {x}^{2} - x - 1")
            || joined_sides
                .contains("{x}^{6} + {x}^{5} + {x}^{4} + {x}^{3} + {x}^{2} + x - 1 - {x}^{5} - {x}^{4} - {x}^{3} - {x}^{2} - x")
            || joined_sides
                .contains("x^6 + x^5 + x^4 + x^3 + x^2 + x - x^5 - x^4 - x^3 - x^2 - x - 1")
            || joined_sides
                .contains("x^6 + x^5 + x^4 + x^3 + x^2 + x - 1 - x^5 - x^4 - x^3 - x^2 - x"),
        "expected a visible full expansion in the hidden web substep, got: {joined_sides}"
    );
}

#[test]
fn step_wire_uses_human_visible_rule_titles() {
    let steps = step_payloads_on_for(RATIONALIZE_LINEAR_ROOT_EXPR);

    let rule_titles: Vec<&str> = steps.iter().map(|step| step.rule.as_str()).collect();
    assert!(
        rule_titles.contains(&"Racionalizar el denominador"),
        "expected humanized visible rule title, got {:?}",
        rule_titles
    );
    assert!(
        rule_titles.contains(&"Restar dos expresiones iguales")
            || rule_titles.contains(&"Collapse Common-Scale Equivalent Difference")
            || steps.iter().any(|step| {
                step.rule == "Racionalizar el denominador"
                    && step
                        .after_latex
                        .contains("\\frac{1 + \\sqrt{x}}{x - 1}")
            }),
        "expected humanized visible cancellation or a compact rationalization step exposing the cancellation pair, got {:?}",
        rule_titles
    );
}

#[test]
fn step_wire_perfect_square_root_uses_human_visible_rule_title() {
    let steps = step_payloads_on_for(PERFECT_SQUARE_ROOT_EXPR);

    let rule_titles: Vec<&str> = steps.iter().map(|step| step.rule.as_str()).collect();
    assert!(
        rule_titles.contains(&"Reconocer un cuadrado perfecto bajo la raíz"),
        "expected humanized visible rule title, got {:?}",
        rule_titles
    );
}

#[test]
fn step_wire_common_factor_cancel_stays_direct_when_the_only_specific_substep_would_duplicate_the_parent_step(
) {
    let (engine, output) = eval_output_for("(2*x)/(4*x)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Cancelar un factor común")
        .expect("expected common factor cancel step");

    assert!(
        step.substeps.is_empty(),
        "when the only possible substep would duplicate the parent step, it should stay direct, got {:?}",
        step.substeps
            .iter()
            .map(|substep| substep.title.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn step_wire_log_cancellation_uses_concrete_substep_expressions() {
    let (engine, output) = eval_output_for("ln(x^3) + ln(y^2) - ln(x^3 * y^2)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.after == "0")
        .expect("expected final log cancellation step");

    if step.rule == "Expandir logaritmos y cancelar términos iguales" {
        let titles = step
            .substeps
            .iter()
            .map(|substep| substep.title.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            titles,
            vec![
                "Expandir el logaritmo del producto o del cociente",
                "Sacar exponentes fuera del logaritmo cuando sea necesario",
                "Cancelar términos iguales",
            ],
            "expected the new explicit three-phase log cancellation story, got {:?}",
            titles
        );
        assert_eq!(
            step.substeps[0].before_latex.as_deref(),
            Some("\\ln({x}^{3}\\cdot {y}^{2})")
        );
        assert_eq!(
            step.substeps[0].after_latex.as_deref(),
            Some("\\ln({x}^{3}) + \\ln({y}^{2})")
        );
        assert_eq!(
            step.substeps[1].before_latex.as_deref(),
            Some("\\ln({x}^{3}) + \\ln({y}^{2}) - \\ln({x}^{3}) - \\ln({y}^{2})")
        );
        let extracted_latex = step.substeps[1]
            .after_latex
            .as_deref()
            .expect("concrete extracted latex");
        assert!(
            extracted_latex.contains("3\\cdot \\ln(x)")
                && extracted_latex.contains("2\\cdot \\ln(|y|)")
                && extracted_latex.matches("3\\cdot \\ln(x)").count() == 2
                && extracted_latex.matches("2\\cdot \\ln(|y|)").count() == 2,
            "expected concrete extracted terms regardless of canonical order, got {extracted_latex}"
        );
        assert_eq!(
            step.substeps[2].before_latex.as_deref(),
            Some(extracted_latex)
        );
        assert_eq!(step.substeps[2].after_latex.as_deref(), Some("0"));
    } else {
        assert_eq!(
            step.rule, "Collapse Exact Zero Additive Subexpression",
            "expected explicit log-cancellation story or direct exact-zero collapse, got {:?}",
            step.rule
        );
        assert!(
            step.substeps.is_empty(),
            "direct exact-zero collapse should not emit redundant log substeps, got {:?}",
            step.substeps
        );
    }
}

#[test]
fn step_wire_cos_diff_sin_diff_quotient_uses_concrete_substep_expressions() {
    let (engine, output) = eval_output_for("(cos(a)-cos(b))/(sin(b)-sin(a))");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    assert_eq!(
        steps.len(),
        3,
        "expected the three-step quotient contraction"
    );
    assert!(
        steps[0].substeps.is_empty() && steps[1].substeps.is_empty(),
        "the first two steps should stay direct because the parent step already shows the concrete rewrite"
    );

    assert_eq!(
        steps[2].substeps[0].before_latex.as_deref(),
        Some(
            "\\frac{2\\cdot \\sin(\\frac{a + b}{2})\\cdot \\sin(\\frac{b - a}{2})}{2\\cdot \\cos(\\frac{a + b}{2})\\cdot \\sin(\\frac{b - a}{2})}"
        )
    );
    assert_eq!(
        steps[2].substeps[0].after_latex.as_deref(),
        Some("\\frac{\\sin(\\frac{a + b}{2})}{\\cos(\\frac{a + b}{2})}")
    );
    assert_eq!(
        steps[2].substeps[1].before_latex.as_deref(),
        Some("\\frac{\\sin(\\frac{a + b}{2})}{\\cos(\\frac{a + b}{2})}")
    );
    assert_eq!(
        steps[2].substeps[1].after_latex.as_deref(),
        Some("\\tan(\\frac{a + b}{2})")
    );
}

#[test]
fn step_wire_log_exponent_rules_stay_direct_when_substeps_only_repeat_the_rule() {
    let (engine, output) = eval_output_for("ln(x^3) + ln(y^2) - ln(x^3 * y^2)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let redundant_titles: Vec<_> = steps
        .iter()
        .filter(|step| step.rule == "Sacar un exponente fuera del logaritmo")
        .map(|step| {
            (
                step.before.clone(),
                step.substeps
                    .iter()
                    .map(|substep| substep.title.clone())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    assert!(
        redundant_titles.iter().all(|(_, titles)| titles.is_empty()),
        "log exponent steps should stay direct without formula-template substeps, got: {:?}",
        redundant_titles
    );
}

#[test]
fn step_wire_difference_of_squares_cancel_recaps_factorization() {
    let (engine, output) = eval_output_for("(x^2 - 1)/(x - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Factorizar una diferencia de cuadrados y cancelar")
        .expect("expected difference-of-squares cancel step");

    assert!(
        step.substeps.len() >= 2,
        "expected a two-phase didactic narrative, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title.contains("diferencia de cuadrados")),
        "expected a substep that recaps the factorization, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title.contains("Ahora se cancela el factor")),
        "expected one substep to show the exact cancellation after factoring, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
}

#[test]
fn step_wire_inverse_tan_relation_isolates_pair_before_applying_identity() {
    let (engine, output) = eval_output_for("atan(3) + (atan(1/3) - pi/2)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Aplicar identidad de arctangentes")
        .expect("expected inverse tan relation step");

    assert!(
        step.substeps.len() == 1,
        "expected a single preparatory substep, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        step.substeps[0].title,
        "Juntar la pareja que encaja con la identidad"
    );
    assert_eq!(
        step.substeps[0].before_latex.as_deref(),
        Some("\\arctan(\\frac{1}{3}) + \\arctan(3) - \\frac{1}{2}\\cdot \\pi")
    );
    assert_eq!(
        step.substeps[0].after_latex.as_deref(),
        Some("\\arctan(\\frac{1}{3}) + \\arctan(3)")
    );
}

#[test]
fn step_wire_cancel_reciprocal_exponents_uses_concrete_square_root_block() {
    let (engine, output) = eval_output_for("(sqrt(x^3) - 1)/(sqrt(x) - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Deshacer raíz y potencia")
        .expect("expected reciprocal exponent cancellation step");

    assert!(
        !step.substeps.is_empty(),
        "expected didactic substeps for reciprocal exponent cancellation"
    );
    assert_eq!(
        step.substeps[0].title,
        "Reemplazar ese bloque en la expresión"
    );
    assert_eq!(
        step.substeps[0].before_latex.as_deref(),
        Some("\\sqrt{x} + {\\sqrt{x}}^{2} + 1")
    );
    assert_eq!(
        step.substeps[0].after_latex.as_deref(),
        Some("\\sqrt{x} + x + 1")
    );
}

#[test]
fn step_wire_perfect_square_root_shows_square_then_absolute_value() {
    let steps = step_payloads_on_for(PERFECT_SQUARE_ROOT_EXPR);

    let step = steps
        .iter()
        .find(|step| step.rule == "Reconocer un cuadrado perfecto bajo la raíz")
        .expect("expected perfect-square sqrt step");

    assert!(
        step.substeps.len() >= 2,
        "expected a two-phase didactic narrative, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        step.substeps[0].title,
        "Reescribir el radicando como un cuadrado perfecto"
    );
    assert_eq!(
        step.substeps[1].title,
        "La raíz de un cuadrado da un valor absoluto"
    );
    assert!(
        step.substeps[0]
            .after_latex
            .as_deref()
            .is_some_and(|latex| latex.contains("^{2}")),
        "expected the first substep to show a square, got: {:?}",
        step.substeps[0].after_latex
    );
    assert!(
        step.substeps[1]
            .after_latex
            .as_deref()
            .is_some_and(|latex| latex.contains("|")),
        "expected the second substep to end in an absolute value, got: {:?}",
        step.substeps[1].after_latex
    );
}

#[test]
fn step_wire_rationalization_explains_conjugate_then_multiply_then_simplify() {
    let steps = step_payloads_on_for(RATIONALIZE_LINEAR_ROOT_EXPR);

    let step = steps
        .iter()
        .find(|step| step.rule == "Racionalizar el denominador")
        .expect("expected rationalization step");

    assert!(
        step.substeps.len() >= 3,
        "expected a three-phase didactic narrative, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        step.substeps[0].title,
        "Cambiar el signo para formar el conjugado"
    );
    assert_eq!(
        step.substeps[1].title,
        "Multiplicar numerador y denominador por ese conjugado"
    );
    assert_eq!(
        step.substeps[2].title,
        "En el denominador aparece una diferencia de cuadrados"
    );
    assert!(
        step.substeps[2]
            .after_latex
            .as_deref()
            .is_some_and(|latex| latex.contains("x - 1^{2}") && !latex.contains("{(-1)}^{2}")),
        "expected the didactic difference-of-squares substep to humanize the square as 1^2, got: {:?}",
        step.substeps[2].after_latex
    );
}

#[test]
fn step_wire_log_inverse_power_explains_inverse_log_relation_before_collapsing() {
    let (engine, output) = eval_output_for("(x^2 + 2)^(1/ln(x^2 + 2))");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Convertir potencia logarítmica inversa")
        .expect("expected log inverse power step");

    assert_eq!(step.substeps.len(), 2);
    assert_eq!(step.substeps[0].title, "Usar que e^(ln(u)) = u");
    assert_eq!(
        step.substeps[1].title,
        "El exponente exterior cancela el ln del exponente interior"
    );
    assert!(
        step.substeps[0]
            .after_latex
            .as_deref()
            .is_some_and(|latex| latex.contains("{e}^{") && latex.contains("\\ln")),
        "expected inverse-log relation in first substep, got: {:?}",
        step.substeps[0].after_latex
    );
    assert_eq!(step.substeps[1].after_latex.as_deref(), Some("e"));
}

#[test]
fn step_wire_rationalization_exact_cube_quotient_uses_notable_quotient_narrative() {
    let (engine, output) = eval_output_for("(x^3 + y^3)/((x + y) * (x^2 - x*y + y^2))");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Racionalizar el denominador")
        .expect("expected rationalization step");

    assert_eq!(step.substeps.len(), 2);
    assert_eq!(
        step.substeps[0].title,
        "Factorizar el numerador como suma de cubos"
    );
    assert_eq!(
        step.substeps[1].title,
        "Numerador y denominador quedan iguales, así que el cociente vale 1"
    );
    assert!(
        step.substeps[0]
            .after_latex
            .as_deref()
            .is_some_and(|latex| latex.contains("(x + y)") && latex.contains("x\\cdot y")),
        "expected the first substep to expose the exact notable quotient factorization, got: {:?}",
        step.substeps[0].after_latex
    );
    assert_eq!(step.substeps[1].after_latex.as_deref(), Some("1"));
}

#[test]
fn step_wire_rationalization_self_cancel_stays_direct_without_tautological_substeps() {
    let steps = step_payloads_on_for(RATIONALIZE_LINEAR_ROOT_EXPR);

    let step = steps.iter().find(|step| {
        step.rule == "Restar dos expresiones iguales"
            || step.rule == "Collapse Common-Scale Equivalent Difference"
    });

    if let Some(step) = step {
        assert!(
            step.substeps.is_empty(),
            "self-cancel step should stay direct without tautological substeps, got: {:?}",
            step.substeps
                .iter()
                .map(|substep| &substep.title)
                .collect::<Vec<_>>()
        );
    } else {
        let rationalize_step = steps
            .iter()
            .find(|step| step.rule == "Racionalizar el denominador")
            .expect("expected rationalization step");
        assert!(
            rationalize_step
                .after_latex
                .contains("\\frac{1 + \\sqrt{x}}{x - 1}"),
            "compact rationalization route should expose the matching fraction before final zero, got {}",
            rationalize_step.after_latex
        );
    }
}

#[test]
fn step_wire_integral_result_does_not_double_wrap_after_highlight() {
    let steps = step_payloads_on_for("integrate((2*x+1)/(x^2+x+1)^(3/2), x)");
    let step = steps
        .first()
        .expect("expected one integration step in wire payload");

    assert_eq!(step.rule, "Calcular la integral");
    assert!(
        step.after_latex.contains("{\\color{green}{"),
        "expected result highlight in after_latex, got {}",
        step.after_latex
    );
    assert!(
        !step
            .after_latex
            .contains("{\\color{green}{{\\color{green}{"),
        "after_latex should not apply the same highlight twice: {}",
        step.after_latex
    );
    assert!(
        step.after_latex.contains("{\\color{green}{-\\frac{2}{"),
        "after_latex should lift the negative sign outside the fraction numerator: {}",
        step.after_latex
    );
    assert!(
        !step.after_latex.contains("\\frac{-2}{"),
        "after_latex should not keep the negative sign inside the numerator: {}",
        step.after_latex
    );
}

#[test]
fn step_wire_cube_quotient_recaps_factor_then_exact_cancellation() {
    let (engine, output) = eval_output_for("((sqrt(x))^3 - 1)/(sqrt(x) - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Factorizar cubos y cancelar")
        .expect("expected sum/difference of cubes cancellation step");

    assert!(
        step.substeps.len() >= 3,
        "expected a factor/cancel/replace narrative, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title
                == "Factorizar el numerador como suma o diferencia de cubos"),
        "expected the step to factor the numerator first, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title.contains("Ahora se cancela el factor")),
        "expected the step to name the common factor being cancelled, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title == "Reemplazar ese bloque en la expresión"),
        "expected the step to close with an explicit replacement substep, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
}

#[test]
fn step_wire_root_denesting_keeps_didactic_substeps() {
    let expr = "sqrt(5 + 2*sqrt(6)) - (sqrt(2) + sqrt(3))";
    let (engine, output) = eval_output_for(expr);
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    if let Some(step) = steps.iter().find(|step| step.rule == "Root Denesting") {
        assert!(
            step.substeps.len() >= 2,
            "expected didactic root-denesting substeps, got: {:?}",
            step.substeps
                .iter()
                .map(|substep| &substep.title)
                .collect::<Vec<_>>()
        );
        assert!(
            step.substeps
                .iter()
                .any(|substep| substep.title.contains("Identificar la forma")),
            "expected root denesting narrative to identify the sqrt(a ± c·sqrt(d)) pattern, got: {:?}",
            step.substeps
                .iter()
                .map(|substep| &substep.title)
                .collect::<Vec<_>>()
        );
    } else {
        assert!(
            steps.iter().any(|step| {
                step.rule == "Collapse Exact Zero Additive Subexpression"
                    && step.before.contains("sqrt(2 · sqrt(6) + 5)")
                    && step.before.contains("sqrt(2)")
                    && step.before.contains("sqrt(3)")
            }),
            "expected either the old root-denesting narrative or the new direct zero-chunk collapse"
        );
    }
}
