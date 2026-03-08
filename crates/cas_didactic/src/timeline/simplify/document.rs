use super::super::simplify_page::{
    render_simplify_timeline_html_header, simplify_timeline_html_footer,
};
use cas_formatter::clean_latex_identities;

pub(super) fn render_simplify_timeline_document(title: &str, body: &str) -> String {
    let mut html = render_simplify_timeline_html_header(title);
    html.push_str(body);
    html.push_str(&simplify_timeline_html_footer());
    clean_latex_identities(&html)
}
