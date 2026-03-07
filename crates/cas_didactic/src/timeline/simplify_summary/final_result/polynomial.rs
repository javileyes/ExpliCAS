use cas_formatter::html_escape;

pub(super) fn render_polynomial_final_result_html(poly_text: &str) -> String {
    let term_count = poly_text.matches('+').count() + 1;
    let mut html = String::from(
        r#"        </div>
        <div class="final-result">
            <strong>🧮 Final Result</strong> <span class="poly-badge">"#,
    );
    html.push_str(&format!("Polynomial: {} terms", term_count));
    html.push_str(
        r#"</span>
            <pre class="poly-output">"#,
    );
    html.push_str(&html_escape(poly_text));
    html.push_str(
        r#"</pre>
        </div>
"#,
    );
    html
}
