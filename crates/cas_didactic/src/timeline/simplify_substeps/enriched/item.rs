use cas_formatter::html_escape;

pub(super) fn render_enriched_substep(sub: &crate::didactic::SubStep) -> String {
    let mut html = format!(
        r#"<div class="substep">
                                    <span class="substep-desc">{}</span>"#,
        html_escape(&sub.description)
    );
    if !sub.before_expr.is_empty() {
        html.push_str(&format!(
            r#"<div class="substep-math">\[{} \rightarrow {}\]</div>"#,
            sub.before_expr, sub.after_expr
        ));
    }
    html.push_str("</div>");
    html
}
