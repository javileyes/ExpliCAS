use cas_api_models::StepWire;
use cas_didactic::{
    collect_step_payloads, format_cli_simplification_steps_with_simplifier, StepDisplayMode,
};
use cas_formatter::{root_style::ParseStyleSignals, DisplayExpr};
use cas_solver::runtime::{to_display_steps, Simplifier};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct AuditCase {
    id: String,
    category: String,
    expr: String,
    focus: String,
}

#[derive(Debug)]
struct AuditArtifact {
    final_expr: String,
    step_count: usize,
    wire_substep_count: usize,
    cli_lines: Vec<String>,
    wire_steps: Vec<StepWire>,
    flags: Vec<String>,
}

fn load_audit_cases() -> Vec<AuditCase> {
    let csv = include_str!("didactic_step_quality_cases.csv");
    csv.lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut parts = line.splitn(4, ',');
            let id = parts.next().expect("missing id").trim().to_string();
            let category = parts.next().expect("missing category").trim().to_string();
            let expr = parts.next().expect("missing expr").trim().to_string();
            let focus = parts.next().expect("missing focus").trim().to_string();
            AuditCase {
                id,
                category,
                expr,
                focus,
            }
        })
        .collect()
}

fn simplify_case(case: &AuditCase) -> AuditArtifact {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    let parsed = cas_parser::parse(&case.expr, &mut simplifier.context).expect("parse");
    let (result, raw_steps) = simplifier.simplify(parsed);
    let display_steps = to_display_steps(raw_steps);
    let style_signals = ParseStyleSignals::from_input_string(&case.expr);

    let cli_lines = format_cli_simplification_steps_with_simplifier(
        &mut simplifier,
        parsed,
        display_steps.as_slice(),
        style_signals,
        StepDisplayMode::Verbose,
    );
    let wire_steps = collect_step_payloads(display_steps.as_slice(), &simplifier.context, "on");
    let final_expr = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result,
        }
    );

    let wire_substep_count = wire_steps
        .iter()
        .map(|step| step.substeps.len())
        .sum::<usize>();
    let mut flags = Vec::new();

    if display_steps.is_empty() {
        flags.push("no steps emitted".to_string());
    }
    if wire_substep_count == 0 {
        flags.push("no wire substeps emitted".to_string());
    }
    if display_steps.len() == 1 && wire_substep_count == 0 {
        flags.push("single step with no didactic substeps".to_string());
    }
    if wire_steps
        .iter()
        .flat_map(|step| step.substeps.iter())
        .any(|substep| substep.before_latex.is_none() || substep.after_latex.is_none())
    {
        flags.push("wire substeps with missing math sides".to_string());
    }

    AuditArtifact {
        final_expr,
        step_count: display_steps.len(),
        wire_substep_count,
        cli_lines,
        wire_steps,
        flags,
    }
}

fn report_output_path() -> PathBuf {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .expect("crates/")
        .parent()
        .expect("repo root")
        .join("docs/generated/DIDACTIC_STEP_QUALITY_AUDIT_REPORT.md")
}

fn build_report(cases: &[AuditCase], artifacts: &[AuditArtifact]) -> String {
    let mut out = String::new();

    writeln!(out, "# Didactic Step Quality Audit Report").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Generated from [didactic_step_quality_cases.csv](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/didactic_step_quality_cases.csv)."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Command: `cargo test -p cas_didactic --test didactic_step_quality_audit didactic_step_quality_audit_generates_markdown_report -- --ignored --exact --nocapture`"
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(out, "## Summary").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| id | category | steps | wire substeps | flags |").unwrap();
    writeln!(out, "| --- | --- | ---: | ---: | --- |").unwrap();

    for (case, artifact) in cases.iter().zip(artifacts.iter()) {
        let flags = if artifact.flags.is_empty() {
            "none".to_string()
        } else {
            artifact.flags.join("; ")
        };
        writeln!(
            out,
            "| `{}` | `{}` | {} | {} | {} |",
            case.id, case.category, artifact.step_count, artifact.wire_substep_count, flags
        )
        .unwrap();
    }

    for (case, artifact) in cases.iter().zip(artifacts.iter()) {
        writeln!(out).unwrap();
        writeln!(out, "## {} ({})", case.id, case.category).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "- Input: `{}`", case.expr).unwrap();
        writeln!(out, "- Focus: `{}`", case.focus).unwrap();
        writeln!(out, "- Final result: `{}`", artifact.final_expr).unwrap();
        writeln!(out, "- Step count: `{}`", artifact.step_count).unwrap();
        writeln!(
            out,
            "- Wire substep count: `{}`",
            artifact.wire_substep_count
        )
        .unwrap();
        if artifact.flags.is_empty() {
            writeln!(out, "- Flags: none").unwrap();
        } else {
            writeln!(out, "- Flags: {}", artifact.flags.join("; ")).unwrap();
        }

        writeln!(out).unwrap();
        writeln!(out, "### CLI Step By Step").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "```text").unwrap();
        if artifact.cli_lines.is_empty() {
            writeln!(out, "(no CLI lines emitted)").unwrap();
        } else {
            for line in &artifact.cli_lines {
                writeln!(out, "{}", line).unwrap();
            }
        }
        writeln!(out, "```").unwrap();

        writeln!(out).unwrap();
        writeln!(out, "### Wire / Web Steps").unwrap();
        writeln!(out).unwrap();

        if artifact.wire_steps.is_empty() {
            writeln!(out, "_No wire steps emitted._").unwrap();
            continue;
        }

        for (index, step) in artifact.wire_steps.iter().enumerate() {
            writeln!(out, "{}. `{}`", index + 1, step.rule).unwrap();
            writeln!(out, "   - before: `{}`", step.before).unwrap();
            writeln!(out, "   - after: `{}`", step.after).unwrap();
            writeln!(out, "   - before_latex: `{}`", step.before_latex).unwrap();
            writeln!(out, "   - after_latex: `{}`", step.after_latex).unwrap();
            if step.substeps.is_empty() {
                writeln!(out, "   - substeps: none").unwrap();
                continue;
            }

            writeln!(out, "   - substeps:").unwrap();
            for (sub_index, substep) in step.substeps.iter().enumerate() {
                writeln!(out, "     {}. `{}`", sub_index + 1, substep.title).unwrap();
                if !substep.lines.is_empty() {
                    writeln!(out, "        - lines: {}", substep.lines.join(" | ")).unwrap();
                }
                if let Some(before_latex) = &substep.before_latex {
                    writeln!(out, "        - before_latex: `{}`", before_latex).unwrap();
                }
                if let Some(after_latex) = &substep.after_latex {
                    writeln!(out, "        - after_latex: `{}`", after_latex).unwrap();
                }
            }
        }
    }

    out
}

#[test]
fn didactic_step_quality_corpus_is_unique_and_nonempty() {
    let cases = load_audit_cases();
    assert!(!cases.is_empty(), "didactic audit corpus must not be empty");

    let mut ids = HashSet::new();
    for case in cases {
        assert!(
            ids.insert(case.id.clone()),
            "duplicate didactic audit case id: {}",
            case.id
        );
        assert!(
            !case.expr.trim().is_empty(),
            "case {} must have expr",
            case.id
        );
    }
}

#[test]
fn didactic_step_quality_cases_simplify_and_emit_steps() {
    let cases = load_audit_cases();

    for case in &cases {
        let artifact = simplify_case(case);
        assert!(
            artifact.step_count > 0,
            "case {} should emit at least one step; expr={}",
            case.id,
            case.expr
        );
        assert!(
            !artifact.cli_lines.is_empty(),
            "case {} should emit CLI didactic lines; expr={}",
            case.id,
            case.expr
        );
        assert!(
            !artifact.wire_steps.is_empty(),
            "case {} should emit wire steps; expr={}",
            case.id,
            case.expr
        );
    }
}

#[test]
fn didactic_step_quality_priority_cases_emit_wire_substeps() {
    let cases = load_audit_cases();
    let priority_ids = [
        "combine_like_terms_basic",
        "same_denominator_fraction_focus",
        "cancel_factors_fraction",
        "difference_of_squares_quotient",
        "pythagorean_identity",
        "inverse_trig_identity",
        "polynomial_expansion_cancel",
        "perfect_square_root",
    ];

    for priority_id in priority_ids {
        let case = cases
            .iter()
            .find(|case| case.id == priority_id)
            .unwrap_or_else(|| panic!("missing priority audit case: {priority_id}"));
        let artifact = simplify_case(case);
        assert!(
            artifact.wire_substep_count > 0,
            "priority case {} should emit at least one wire substep; expr={}",
            case.id,
            case.expr
        );
    }
}

#[test]
fn didactic_step_quality_priority_cases_use_multiphase_human_narratives() {
    let cases = load_audit_cases();

    let qualitative_targets: &[(&str, &[&str])] = &[
        (
            "rationalize_linear_root",
            &[
                "Cambiar el signo para formar el conjugado",
                "Multiplicar numerador y denominador por ese conjugado",
                "En el denominador aparece una diferencia de cuadrados",
            ],
        ),
        ("cancel_factors_fraction", &["se cancela"]),
        (
            "difference_of_squares_quotient",
            &["diferencia de cuadrados", "Ahora se cancela el factor"],
        ),
        (
            "inverse_trig_identity",
            &[
                "Juntar la pareja que encaja con la identidad",
                "Esa pareja vale pi/2",
            ],
        ),
        (
            "perfect_square_root",
            &[
                "Reescribir el radicando como un cuadrado perfecto",
                "La raíz de un cuadrado da un valor absoluto",
            ],
        ),
        (
            "cube_quotient_radical",
            &[
                "Llamar t = sqrt(x) para reconocer la forma",
                "Ese cociente notable se convierte en t^2 + t + 1",
                "Volver a poner t = sqrt(x)",
                "El cuadrado deshace la raíz",
                "Reemplazar ese bloque en la expresión",
            ],
        ),
        (
            "geometric_product_cancellation",
            &[
                "Distribuir cada término del producto",
                "Agrupar los términos del mismo grado",
                "Los términos intermedios se cancelan por parejas",
            ],
        ),
    ];

    for &(case_id, expected_titles) in qualitative_targets {
        let case = cases
            .iter()
            .find(|case| case.id == case_id)
            .unwrap_or_else(|| panic!("missing qualitative audit case: {case_id}"));
        let artifact = simplify_case(case);
        let titles: Vec<&str> = artifact
            .wire_steps
            .iter()
            .flat_map(|step| step.substeps.iter())
            .map(|substep| substep.title.as_str())
            .collect();

        for expected_title in expected_titles {
            assert!(
                titles.iter().any(|title| title.contains(expected_title)),
                "case {} should contain didactic title {:?}, got {:?}",
                case.id,
                expected_title,
                titles
            );
        }
    }
}

#[test]
fn didactic_step_quality_priority_cases_make_cli_narrative_less_magic() {
    let cases = load_audit_cases();

    let rationalize_case = cases
        .iter()
        .find(|case| case.id == "rationalize_linear_root")
        .expect("missing rationalize_linear_root audit case");
    let rationalize_artifact = simplify_case(rationalize_case);
    let rationalize_cli = rationalize_artifact.cli_lines.join("\n");
    assert!(
        !rationalize_cli.contains("Quitar el factor 1 que no cambia el valor")
            && !rationalize_cli.contains("\n2. Quitar el factor 1"),
        "rationalize_linear_root CLI narrative should avoid a standalone factor-1 step when the current conjugate narrative already stays clear without it, got:\n{}",
        rationalize_cli
    );
    assert!(
        !rationalize_cli.contains("Calcular 1^2 = 1")
            && !rationalize_cli.contains("Calcular (-1)^2 = 1"),
        "rationalize_linear_root CLI narrative should avoid trivial literal-power micro-substeps, got:\n{}",
        rationalize_cli
    );
    assert!(
        rationalize_cli.contains("x - 1^2"),
        "rationalize_linear_root CLI narrative should humanize the difference-of-squares substep as x - 1^2, got:\n{}",
        rationalize_cli
    );
    assert!(
        !rationalize_cli.contains("x - ((-1))^2")
            && !rationalize_cli.contains("x - (-1)^2"),
        "rationalize_linear_root CLI narrative should avoid showing -1 squared inside the didactic substep, got:\n{}",
        rationalize_cli
    );
    assert!(
        !rationalize_cli.contains("Cambio local: sqrt(x) + 1 -> sqrt(x) + 1")
            && !rationalize_cli.contains("Cambio local: (sqrt(x) + 1) -> sqrt(x) + 1")
            && !rationalize_cli.contains("Rule: (sqrt(x) + 1) -> sqrt(x) + 1"),
        "rationalize_linear_root CLI narrative should not show a fake local change, got:\n{}",
        rationalize_cli
    );
    assert!(
        !rationalize_cli.contains("Cambio local: 1 / (sqrt(x) - 1) ->"),
        "rationalize_linear_root CLI narrative should avoid a technical local-change line when the conjugate substeps already explain the fraction move, got:\n{}",
        rationalize_cli
    );
    assert!(
        !rationalize_cli.contains("\n   Rule: "),
        "rationalize_linear_root CLI narrative should avoid raw Rule lines when human substeps already explain the move, got:\n{}",
        rationalize_cli
    );
    assert!(
        !rationalize_cli.contains("\n1. Multiplicar por el conjugado  [")
            && !rationalize_cli.contains("\n2. Calcular potencia numérica  [")
            && !rationalize_cli.contains("\n3. Restar dos expresiones iguales  ["),
        "rationalize_linear_root CLI narrative should avoid bracketed rule labels in the step header, got:\n{}",
        rationalize_cli
    );
    assert!(
        !rationalize_cli.contains("\n2. 1 * x = x")
            && !rationalize_cli.contains("\n2. Evaluate literal power")
            && !rationalize_cli.contains("\n3. a - a = 0")
            && !rationalize_cli.contains("Los dos términos ya son el mismo")
            && !rationalize_cli.contains("Restar algo consigo mismo da 0"),
        "rationalize_linear_root CLI narrative should use human headers instead of formula-like labels when a clearer title exists, got:\n{}",
        rationalize_cli
    );
    let self_cancel_step = rationalize_artifact
        .wire_steps
        .iter()
        .find(|step| step.rule == "Restar dos expresiones iguales")
        .expect("missing self-cancel wire step in rationalize_linear_root");
    assert!(
        self_cancel_step.substeps.is_empty(),
        "rationalize_linear_root self-cancel step should stay direct without tautological substeps, got {:?}",
        self_cancel_step
            .substeps
            .iter()
            .map(|substep| substep.title.as_str())
            .collect::<Vec<_>>()
    );

    let cube_case = cases
        .iter()
        .find(|case| case.id == "cube_quotient_radical")
        .expect("missing cube_quotient_radical audit case");
    let cube_artifact = simplify_case(cube_case);
    let cube_cli = cube_artifact.cli_lines.join("\n");
    assert!(
        cube_cli.contains("El cuadrado deshace la raíz"),
        "cube_quotient_radical CLI narrative should explain why the square cancels the root, got:\n{}",
        cube_cli
    );
    assert!(
        cube_cli.contains("Pasar la potencia al interior de la raíz"),
        "cube_quotient_radical CLI narrative should explain the first rewrite in human terms, got:\n{}",
        cube_cli
    );
    assert!(
        !cube_cli.contains("\n   Rule: "),
        "cube_quotient_radical CLI narrative should avoid raw Rule lines when human substeps already explain the move, got:\n{}",
        cube_cli
    );
    assert!(
        !cube_cli.contains("Canonicalize Nested Power")
            && !cube_cli.contains("opaque substitution"),
        "cube_quotient_radical CLI narrative should avoid internal engine jargon in headers, got:\n{}",
        cube_cli
    );
    assert!(
        !cube_cli.contains("\n1. Reescribir potencia de una raíz  [")
            && !cube_cli.contains("\n2. Reconocer un cociente notable  [")
            && !cube_cli.contains("\n3. Deshacer una raíz con su potencia  ["),
        "cube_quotient_radical CLI narrative should avoid bracketed rule labels in the step header, got:\n{}",
        cube_cli
    );
    assert!(
        !cube_cli.contains("Cambio local: sqrt(x)^(3) ->")
            && !cube_cli.contains("Cambio local: sqrt(x)^(2) -> x"),
        "cube_quotient_radical CLI narrative should avoid technical local-change lines when the human substeps already explain both rewrites, got:\n{}",
        cube_cli
    );
    assert!(
        !cube_cli.contains("Reescribir potencia de una raíz  [Reescribir potencia de una raíz]")
            && !cube_cli.contains("Reconocer un cociente notable  [Reconocer cociente notable]"),
        "cube_quotient_radical CLI narrative should avoid duplicated header labels, got:\n{}",
        cube_cli
    );
    assert!(
        cube_artifact
            .wire_steps
            .iter()
            .any(|step| step.rule == "Reconocer un cociente notable"),
        "cube_quotient_radical should use a human visible rule title, got {:?}",
        cube_artifact
            .wire_steps
            .iter()
            .map(|step| step.rule.as_str())
            .collect::<Vec<_>>()
    );

    let cube_wire_before: Vec<&str> = cube_artifact
        .wire_steps
        .iter()
        .map(|step| step.before.as_str())
        .collect();
    assert!(
        cube_wire_before
            .iter()
            .any(|before| before.contains("sqrt(x)")),
        "cube_quotient_radical wire text should keep roots human-readable, got {:?}",
        cube_wire_before
    );
    assert!(
        cube_wire_before
            .iter()
            .all(|before| !before.contains("x^(1/2)^2")),
        "cube_quotient_radical wire text should avoid fractional-exponent root notation, got {:?}",
        cube_wire_before
    );

    let geometric_case = cases
        .iter()
        .find(|case| case.id == "geometric_product_cancellation")
        .expect("missing geometric_product_cancellation audit case");
    let geometric_artifact = simplify_case(geometric_case);
    let geometric_cli = geometric_artifact.cli_lines.join("\n");
    assert!(
        geometric_cli.contains("Expandir y reagrupar un producto polinómico"),
        "geometric_product_cancellation CLI narrative should use a human step title, got:\n{}",
        geometric_cli
    );
    assert!(
        geometric_cli.contains("Distribuir cada término del producto")
            && geometric_cli.contains("Agrupar los términos del mismo grado")
            && geometric_cli.contains("Los términos intermedios se cancelan por parejas"),
        "geometric_product_cancellation should expose the full didactic narrative, got:\n{}",
        geometric_cli
    );
    assert!(
        geometric_cli.contains("x^6 + x^5 + x^4 + x^3 + x^2 + x - x^5 - x^4 - x^3 - x^2 - x - 1")
            || geometric_cli
                .contains("x^6 + x^5 + x^4 + x^3 + x^2 + x - 1 - x^5 - x^4 - x^3 - x^2 - x"),
        "geometric_product_cancellation should show the small full expansion, got:\n{}",
        geometric_cli
    );
    assert!(
        !geometric_cli.contains("\n   Cambio local: "),
        "geometric_product_cancellation should avoid a raw local-change line when the substeps already explain the move, got:\n{}",
        geometric_cli
    );

    let nested_fraction_case = cases
        .iter()
        .find(|case| case.id == "nested_fraction_one_over_sum")
        .expect("missing nested_fraction_one_over_sum audit case");
    let nested_fraction_cli = simplify_case(nested_fraction_case).cli_lines.join("\n");
    assert!(
        !nested_fraction_cli.contains("Simplify Complex Fraction"),
        "nested_fraction_one_over_sum CLI narrative should use a human header instead of internal rule names, got:\n{}",
        nested_fraction_cli
    );
    assert!(
        !nested_fraction_cli.contains("\n2. Simplify nested fraction"),
        "nested_fraction_one_over_sum CLI narrative should avoid English fallback descriptions when a humanized title exists, got:\n{}",
        nested_fraction_cli
    );
    assert!(
        !nested_fraction_cli.contains("Identificar la fracción anidada en el denominador")
            && !nested_fraction_cli.contains("Simplificar: 1/(a/b) = b/a"),
        "nested_fraction_one_over_sum CLI narrative should avoid preparatory fraction-inversion micro-steps when one direct explanation is enough, got:\n{}",
        nested_fraction_cli
    );
    assert!(
        nested_fraction_cli.contains("Dividir entre una fracción equivale a invertirla"),
        "nested_fraction_one_over_sum CLI narrative should explain the inversion directly, got:\n{}",
        nested_fraction_cli
    );
    assert!(
        !nested_fraction_cli.contains("Cambio local: 1 / x + 1 / y ->")
            && !nested_fraction_cli.contains("Cambio local: 1 / ((x + y) / (x * y)) ->"),
        "nested_fraction_one_over_sum CLI narrative should avoid technical local-change lines once the fraction substeps already explain the move, got:\n{}",
        nested_fraction_cli
    );

    let same_denominator_case = cases
        .iter()
        .find(|case| case.id == "same_denominator_fraction_focus")
        .expect("missing same_denominator_fraction_focus audit case");
    let same_denominator_cli = simplify_case(same_denominator_case).cli_lines.join("\n");
    assert!(
        !same_denominator_cli.contains("Cambio local: a / d + 1 ->")
            && !same_denominator_cli.contains("Cambio local: b / d + (a + d) / d ->"),
        "same_denominator_fraction_focus should avoid technical local-change lines once the fraction substeps already explain the move, got:\n{}",
        same_denominator_cli
    );

    let cancel_factor_case = cases
        .iter()
        .find(|case| case.id == "cancel_factors_fraction")
        .expect("missing cancel_factors_fraction audit case");
    let cancel_factor_cli = simplify_case(cancel_factor_case).cli_lines.join("\n");
    assert!(
        !cancel_factor_cli.contains("Cambio local: 2 * x / (4 * x) ->"),
        "cancel_factors_fraction should avoid a technical local-change line when the cancellation substep already explains the move, got:\n{}",
        cancel_factor_cli
    );

    let difference_quotient_case = cases
        .iter()
        .find(|case| case.id == "difference_of_squares_quotient")
        .expect("missing difference_of_squares_quotient audit case");
    let difference_quotient_cli = simplify_case(difference_quotient_case).cli_lines.join("\n");
    assert!(
        !difference_quotient_cli.contains("Cambio local: (x + 1) * (x - 1) / (x - 1) ->"),
        "difference_of_squares_quotient should avoid a technical local-change line when the factorization and cancellation substeps already explain the move, got:\n{}",
        difference_quotient_cli
    );

    let combine_case = cases
        .iter()
        .find(|case| case.id == "combine_like_terms_basic")
        .expect("missing combine_like_terms_basic audit case");
    let combine_cli = simplify_case(combine_case).cli_lines.join("\n");
    assert!(
        !combine_cli.contains("Identity Property of Addition"),
        "combine_like_terms_basic CLI narrative should avoid internal addition-identity labels, got:\n{}",
        combine_cli
    );
    assert!(
        !combine_cli.contains("\n1. 0 + x = x"),
        "combine_like_terms_basic CLI narrative should use a human header instead of the algebraic schema, got:\n{}",
        combine_cli
    );
    assert!(
        !combine_cli.contains("\n2. Combine like terms"),
        "combine_like_terms_basic CLI narrative should avoid English fallback headers when a humanized title exists, got:\n{}",
        combine_cli
    );
    assert!(
        !combine_cli.contains("Sumar 0 no cambia el valor"),
        "combine_like_terms_basic CLI narrative should avoid a trivial +0 micro-substep once the step title already says 'Quitar el 0', got:\n{}",
        combine_cli
    );

    let inverse_trig_case = cases
        .iter()
        .find(|case| case.id == "inverse_trig_identity")
        .expect("missing inverse_trig_identity audit case");
    let inverse_trig_cli = simplify_case(inverse_trig_case).cli_lines.join("\n");
    assert!(
        !inverse_trig_cli.contains("Canonicalize Trig Function Names")
            && !inverse_trig_cli.contains("Canonicalize]"),
        "inverse_trig_identity CLI narrative should avoid canonicalization jargon in headers, got:\n{}",
        inverse_trig_cli
    );
    assert!(
        !inverse_trig_cli.contains("\n1. atan -> arctan")
            && !inverse_trig_cli.contains("\n2. atan -> arctan")
            && !inverse_trig_cli.contains("\n4. arctan(x) + arctan(1/x) = π/2"),
        "inverse_trig_identity CLI narrative should avoid formula or rename schemas in headers when human wording exists, got:\n{}",
        inverse_trig_cli
    );
    assert!(
        !inverse_trig_cli.contains("\n1. Usar el nombre arctan")
            && !inverse_trig_cli.contains("\n2. Usar el nombre arctan")
            && !inverse_trig_cli.contains("Reordenar la expresión"),
        "inverse_trig_identity CLI narrative should absorb preparatory rename/reordering steps into the main identity step, got:\n{}",
        inverse_trig_cli
    );
    assert!(
        inverse_trig_cli.contains("Las dos partes se compensan exactamente"),
        "inverse_trig_identity CLI narrative should explain why the final opaque-substitution identity cancels to zero, got:\n{}",
        inverse_trig_cli
    );
    assert!(
        !inverse_trig_cli.contains("Cambio local: pi / 2 - 1/2 * pi -> 0"),
        "inverse_trig_identity CLI narrative should avoid a dry local-change line once the exact-cancellation substep explains it, got:\n{}",
        inverse_trig_cli
    );
    assert!(
        !inverse_trig_cli.contains("Cambio local: arctan(1/3) + arctan(3) ->"),
        "inverse_trig_identity CLI narrative should avoid a dry local-change line when the identity substeps already explain the move, got:\n{}",
        inverse_trig_cli
    );

    let perfect_square_case = cases
        .iter()
        .find(|case| case.id == "perfect_square_root")
        .expect("missing perfect_square_root audit case");
    let perfect_square_artifact = simplify_case(perfect_square_case);
    let perfect_square_cli = perfect_square_artifact.cli_lines.join("\n");
    assert!(
        perfect_square_cli.contains("Reconocer un cuadrado perfecto bajo la raíz"),
        "perfect_square_root CLI narrative should use a human header, got:\n{}",
        perfect_square_cli
    );
    assert!(
        !perfect_square_cli.contains("Cambio local: sqrt(x^(2) + 2 * x + 1) ->"),
        "perfect_square_root CLI narrative should avoid a dry local-change line when the square-pattern substeps already explain the move, got:\n{}",
        perfect_square_cli
    );
    assert!(
        perfect_square_artifact
            .wire_steps
            .iter()
            .any(|step| step.rule == "Reconocer un cuadrado perfecto bajo la raíz"),
        "perfect_square_root wire narrative should use the same human title, got {:?}",
        perfect_square_artifact
            .wire_steps
            .iter()
            .map(|step| step.rule.as_str())
            .collect::<Vec<_>>()
    );

    let polynomial_case = cases
        .iter()
        .find(|case| case.id == "polynomial_expansion_cancel")
        .expect("missing polynomial_expansion_cancel audit case");
    let polynomial_cli = simplify_case(polynomial_case).cli_lines.join("\n");
    assert!(
        !polynomial_cli.contains("\n3. -(2 * x) = -2 * x"),
        "polynomial_expansion_cancel CLI narrative should hide sign-canonicalization steps that do not create a visible human change, got:\n{}",
        polynomial_cli
    );
    assert!(
        !polynomial_cli.contains("Quitar paréntesis tras el signo menos"),
        "polynomial_expansion_cancel CLI narrative should avoid surfacing a standalone sign-cleanup step when the cancellation remains understandable without it, got:\n{}",
        polynomial_cli
    );
    assert!(
        !polynomial_cli.contains("Cambio local: (a + b)^(2) ->")
            && !polynomial_cli.contains("Cambio local: a^(2) - a^(2) ->")
            && !polynomial_cli.contains("Cambio local: 2 * a * b - 2 * a * b ->"),
        "polynomial_expansion_cancel CLI narrative should avoid dry local-change lines when the expansion and cancellation substeps already explain the move, got:\n{}",
        polynomial_cli
    );

    let pythagorean_case = cases
        .iter()
        .find(|case| case.id == "pythagorean_identity")
        .expect("missing pythagorean_identity audit case");
    let pythagorean_cli = simplify_case(pythagorean_case).cli_lines.join("\n");
    assert!(
        !pythagorean_cli.contains("Cambio local: sin(x)^(2) + cos(x)^(2) ->"),
        "pythagorean_identity CLI narrative should avoid a dry local-change line when the identity substep already explains the move, got:\n{}",
        pythagorean_cli
    );
}

#[test]
#[ignore]
fn didactic_step_quality_audit_generates_markdown_report() {
    let cases = load_audit_cases();
    let artifacts: Vec<AuditArtifact> = cases.iter().map(simplify_case).collect();
    let report = build_report(&cases, &artifacts);
    let report_path = report_output_path();
    fs::create_dir_all(report_path.parent().expect("report dir")).expect("create report dir");
    fs::write(&report_path, report).expect("write report");
    eprintln!("didactic audit report written to {}", report_path.display());
}
