use crate::timeline::render_template;
use cas_formatter::html_escape;

pub(super) fn render_polynomial_final_result_html(poly_text: &str) -> String {
    let term_count = poly_text.matches('+').count() + 1;
    let term_count_text = term_count.to_string();
    let poly_text_html = html_escape(poly_text);
    render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/simplify_render/final_result_polynomial.html"
        )),
        &[
            ("__TERM_COUNT__", term_count_text.as_str()),
            ("__POLY_TEXT__", poly_text_html.as_str()),
        ],
    )
}
