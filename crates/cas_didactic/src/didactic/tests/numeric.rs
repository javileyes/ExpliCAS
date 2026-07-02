use super::super::shared_numeric::gcd_bigint;
use super::super::*;
use crate::didactic::display_policy::build_cli_substeps_render_plan;
use num_bigint::BigInt;
use num_rational::BigRational;

#[test]
fn test_format_fraction() {
    let half = BigRational::new(BigInt::from(1), BigInt::from(2));
    assert_eq!(format_fraction(&half), "\\frac{1}{2}");

    let three = BigRational::from_integer(BigInt::from(3));
    assert_eq!(format_fraction(&three), "3");
}

#[test]
fn test_gcd_lcm() {
    let a = BigInt::from(12);
    let b = BigInt::from(8);
    assert_eq!(gcd_bigint(&a, &b), BigInt::from(4));
    assert_eq!(lcm_bigint(&a, &b), BigInt::from(24));
}

#[test]
fn test_build_cli_substeps_render_plan_fraction_sum_deduped() {
    let sub_steps = vec![SubStep::new(
        "Find common denominator for fractions",
        "",
        "",
    )];
    let plan = build_cli_substeps_render_plan(&sub_steps);
    assert_eq!(plan.header, Some("[Suma de fracciones en exponentes]"));
    assert!(plan.dedupe_once);
}

#[test]
fn test_latex_to_plain_text_converts_frac_and_text() {
    let input = r"\text{Paso}: \frac{1}{x+1} \cdot y";
    let output = latex_to_plain_text(input);
    assert!(output.contains("Paso"));
    assert!(output.contains("1/(x+1)"));
    assert!(output.contains("·"));
}

#[test]
fn test_latex_to_plain_text_converts_matrix_environments() {
    // Multi-row matrix -> nested brackets (no `beginbmatrix` garbage).
    let matrix =
        latex_to_plain_text(r"\begin{bmatrix} -2 & 1 \\ \frac{3}{2} & -\frac{1}{2} \end{bmatrix}");
    assert_eq!(matrix, "[[-2, 1], [3/2, -1/2]]", "got {matrix}");
    assert!(!matrix.contains("bmatrix"));

    // Single-row vector/list -> flat brackets, matching the engine's own result display.
    let vector = latex_to_plain_text(r"\begin{bmatrix} 1 & 2 & 3 & 4 & 6 & 12 \end{bmatrix}");
    assert_eq!(vector, "[1, 2, 3, 4, 6, 12]", "got {vector}");

    // Nested inside a function call (e.g. det(...)).
    let det = latex_to_plain_text(r"\det(\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix})");
    assert_eq!(det, "det([[1, 2], [3, 4]])", "got {det}");
}

#[test]
fn test_cli_substep_render_prefers_explicit_latex_when_available() {
    let enriched = EnrichedStep {
        base_step: crate::runtime::Step::new_compact(
            "",
            "",
            cas_ast::ExprId::from_raw(0),
            cas_ast::ExprId::from_raw(0),
        ),
        sub_steps: vec![
            SubStep::new("test", "bad pseudo latex", "bad pseudo latex result")
                .with_before_latex("\\frac{1}{x}")
                .with_after_latex("\\frac{x+1}{x}"),
        ],
    };

    let mut state = super::super::display_policy::CliSubstepsRenderState::default();
    let lines =
        super::super::display_policy::render_cli_enriched_substeps_lines(&enriched, &mut state);
    assert!(lines.iter().any(|line| line.contains("1/x")));
    assert!(lines.iter().any(|line| line.trim() == "->"));
    assert!(lines.iter().any(|line| line.contains("(x+1)/x")));
    assert!(!lines.iter().any(|line| line.contains("bad pseudo latex")));
}

#[test]
fn test_latex_to_plain_text_handles_rationalization_fraction_and_sqrt() {
    let input = r"\frac{(1) \cdot (\sqrt{x} + 1)}{\sqrt{x} - 1  \cdot  (\sqrt{x} + 1)}";
    let output = latex_to_plain_text(input);
    assert!(!output.contains("frac{"), "unexpected raw frac in {output}");
    assert!(
        !output.contains("\\sqrt"),
        "unexpected raw sqrt in {output}"
    );
}

#[test]
fn test_latex_to_plain_text_parenthesizes_power_base_when_needed() {
    let input = r"\sqrt{{x + 1}^{2}}";
    let output = latex_to_plain_text(input);
    assert!(
        output.contains("sqrt((x + 1)^2)"),
        "expected grouped square base, got: {output}"
    );
}

#[test]
fn test_latex_to_plain_text_does_not_double_parenthesize_power_base() {
    let input = r"\frac{1 - x}{2\cdot \sqrt{x}\cdot {(x + 1)}^{2}}";
    let output = latex_to_plain_text(input);
    assert!(
        output.contains("(x + 1)^2"),
        "expected already-parenthesized power base to stay single-wrapped, got: {output}"
    );
    assert!(
        !output.contains("((x + 1))^2"),
        "unexpected duplicated parentheses around power base, got: {output}"
    );
}

#[test]
fn test_latex_to_plain_text_humanizes_even_negative_literal_square() {
    let input = r"x - ((-1))^2";
    let output = latex_to_plain_text(input);
    assert!(
        output.contains("x - 1^2"),
        "expected humanized even literal square, got: {output}"
    );
    assert!(
        !output.contains("(-1)^2") && !output.contains("((-1))^2"),
        "unexpected raw negative literal square, got: {output}"
    );
}

#[test]
fn test_latex_to_plain_text_avoids_extra_parens_for_function_power_base() {
    let input = r"{\sin(x)}^{2} + {\cos(x)}^{2}";
    let output = latex_to_plain_text(input);
    assert!(
        output.contains("sin(x)^2 + cos(x)^2"),
        "expected humanized function powers without extra parens, got: {output}"
    );
    assert!(
        !output.contains("(sin(x))^2") && !output.contains("(cos(x))^2"),
        "unexpected extra parens around function power bases, got: {output}"
    );
}

#[test]
fn test_latex_to_plain_text_parenthesizes_grouped_symbolic_exponents() {
    let input = r"{x}^{a + b} + {y}^{a + b + c}";
    let output = latex_to_plain_text(input);
    assert!(
        output.contains("x^(a + b)"),
        "expected grouped symbolic exponent for x, got: {output}"
    );
    assert!(
        output.contains("y^(a + b + c)"),
        "expected grouped symbolic exponent for y, got: {output}"
    );
}

#[test]
fn test_latex_to_plain_text_parenthesizes_nested_power_exponents() {
    let input = r"{e}^{{x}^{2}} + \sqrt{{e}^{{x}^{2}} + 1}";
    let output = latex_to_plain_text(input);
    assert!(
        output.contains("e^(x^2) + sqrt(e^(x^2) + 1)"),
        "expected nested power exponent to stay grouped, got: {output}"
    );
    assert!(
        !output.contains("e^x^2"),
        "nested exponent should not become visually ambiguous, got: {output}"
    );
}

#[test]
fn test_latex_to_plain_text_leaves_atomic_exponents_without_extra_parens() {
    let input = r"{x}^{2} + {y}^{a}";
    let output = latex_to_plain_text(input);
    assert!(
        output.contains("x^2 + y^a"),
        "expected atomic exponents without extra parens, got: {output}"
    );
    assert!(
        !output.contains("x^(2)") && !output.contains("y^(a)"),
        "unexpected redundant exponent parentheses, got: {output}"
    );
}

#[test]
fn test_latex_to_plain_text_parenthesizes_extra_braced_fraction_denominator_power() {
    let input = r"\frac{2}{{1 - {\sin(x)}^{2}}}";
    let output = latex_to_plain_text(input);
    assert!(
        output.contains("2/(1 - sin(x)^2)"),
        "expected protected denominator square, got: {output}"
    );
}

#[test]
fn test_latex_to_plain_text_strips_color_wrapped_fraction_denominator_power() {
    let input = r"\frac{2}{{\color{red}{1 - {\sin(x)}^{2}}}}";
    let output = latex_to_plain_text(input);
    assert!(
        output.contains("2/(1 - sin(x)^2)"),
        "expected protected denominator square after stripping color wrapper, got: {output}"
    );
    assert!(
        !output.contains("colorred"),
        "unexpected leaked color command in: {output}"
    );
}

#[test]
fn test_latex_to_plain_text_preserves_squared_sine_from_formatter_fraction() {
    let mut ctx = cas_ast::Context::new();
    let expr = cas_parser::parse("2/(1 - sin(x)^2)", &mut ctx).expect("parse");
    let latex = cas_formatter::LaTeXExpr {
        context: &ctx,
        id: expr,
    }
    .to_latex();
    let output = latex_to_plain_text(&latex);
    assert!(
        output.contains("2/(1 - sin(x)^2)"),
        "expected protected denominator square, latex={latex}, output={output}"
    );
}
