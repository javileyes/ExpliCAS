use cas_ast::{Context, Equation, RelOp};
use cas_formatter::{html_escape, LaTeXExpr};
use cas_solver::SolveStep;

pub(super) fn render_equation_latex(context: &Context, equation: &Equation) -> String {
    format!(
        "{} {} {}",
        LaTeXExpr {
            context,
            id: equation.lhs
        }
        .to_latex(),
        solve_relop_to_latex(&equation.op),
        LaTeXExpr {
            context,
            id: equation.rhs
        }
        .to_latex()
    )
}

pub(super) fn render_solve_step_html(
    context: &Context,
    step_number: usize,
    step: &SolveStep,
) -> String {
    let eq_latex = render_equation_latex(context, &step.equation_after);

    let mut html = format!(
        r#"        <div class="step">
            <div class="step-number">Step {}</div>
            <div class="description">{}</div>
            <div class="equation">
                \[{}\]
            </div>
"#,
        step_number,
        html_escape(&step.description),
        eq_latex
    );

    if !step.substeps.is_empty() {
        let substep_id = format!("substeps-{}", step_number);
        html.push_str(&format!(
            r#"            <div class="substeps-toggle" onclick="toggleSubsteps('{}')">
                <span class="arrow">▶</span>
                <span>Show derivation ({} steps)</span>
            </div>
            <div id="{}" class="substeps-container">
"#,
            substep_id,
            step.substeps.len(),
            substep_id
        ));

        for (j, substep) in step.substeps.iter().enumerate() {
            let sub_eq_latex = render_equation_latex(context, &substep.equation_after);
            html.push_str(&format!(
                r#"                <div class="substep">
                    <div class="substep-number">Step {}.{}</div>
                    <div class="substep-description">{}</div>
                    <div class="substep-equation">
                        \[{}\]
                    </div>
                </div>
"#,
                step_number,
                j + 1,
                html_escape(&substep.description),
                sub_eq_latex
            ));
        }

        html.push_str("            </div>\n");
    }

    html.push_str("        </div>\n");
    html
}

pub(super) fn render_solve_final_result_html(var: &str, solution_latex: &str) -> String {
    format!(
        r#"        </div>
        <div class="final-result">
            \(\textbf{{Solution: }} {} = \)
            \[{}\]
        </div>
    </div>
"#,
        html_escape(var),
        solution_latex
    )
}

pub(super) fn solve_relop_to_latex(op: &RelOp) -> &'static str {
    match op {
        RelOp::Eq => "=",
        RelOp::Neq => "\\neq",
        RelOp::Lt => "<",
        RelOp::Gt => ">",
        RelOp::Leq => "\\leq",
        RelOp::Geq => "\\geq",
    }
}
