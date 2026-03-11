use crate::cas_solver::Step;

pub(super) fn render_before_section(global_before: &str) -> String {
    format!(
        r#"                    <div class="math-expr before">
                        \(\textbf{{Before:}}\)
                        \[{}\]
                    </div>
"#,
        global_before
    )
}

pub(super) fn render_rule_section(step: &Step, local_change_latex: &str) -> String {
    format!(
        r#"                    <div class="rule-description">
                        <div class="rule-name">\(\text{{{}}}\)</div>
                        <div class="local-change">
                            \[{}\]
                        </div>
                    </div>
"#,
        step.description, local_change_latex
    )
}

pub(super) fn render_after_section(global_after: &str) -> String {
    format!(
        r#"                    <div class="math-expr after">
                        \(\textbf{{After:}}\)
                        \[{}\]
                    </div>
"#,
        global_after
    )
}
