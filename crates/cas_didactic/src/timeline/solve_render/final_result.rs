use cas_formatter::html_escape;

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
