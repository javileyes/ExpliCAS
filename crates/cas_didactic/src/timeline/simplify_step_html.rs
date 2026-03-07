use cas_formatter::html_escape;
use cas_solver::Step;

use super::simplify_highlights::TimelineRenderedStepMath;

pub(super) fn render_timeline_step_html(
    step_number: usize,
    step: &Step,
    rendered_step_math: &TimelineRenderedStepMath,
    sub_steps_html: &str,
    rule_substeps_html: &str,
    domain_html: &str,
) -> String {
    let requires_html = "";

    format!(
        r#"            <div class="step">
                <div class="step-number">{}</div>
                <div class="step-content">
                    <h3>{}</h3>
                    <div class="math-expr before">
                        \(\textbf{{Before:}}\)
                        \[{}\]
                    </div>
                    {}
                    <div class="rule-description">
                        <div class="rule-name">\(\text{{{}}}\)</div>
                        <div class="local-change">
                            \[{}\]
                        </div>
                    </div>
                    {}
                    <div class="math-expr after">
                        \(\textbf{{After:}}\)
                        \[{}\]
                    </div>
{}{}                </div>
            </div>
"#,
        step_number,
        html_escape(&step.rule_name),
        rendered_step_math.global_before,
        sub_steps_html,
        step.description,
        rendered_step_math.local_change_latex,
        rule_substeps_html,
        rendered_step_math.global_after,
        requires_html,
        domain_html
    )
}
