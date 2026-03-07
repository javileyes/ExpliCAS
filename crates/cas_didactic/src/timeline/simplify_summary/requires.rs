use cas_ast::Context;
use cas_formatter::html_escape;
use cas_solver::{render_conditions_normalized, ImplicitCondition};

pub(super) fn render_timeline_global_requires_html(
    context: &mut Context,
    global_requires: &[ImplicitCondition],
) -> String {
    if global_requires.is_empty() {
        return String::new();
    }

    let requires_messages = render_conditions_normalized(context, global_requires);
    if requires_messages.is_empty() {
        return String::new();
    }

    let escaped: Vec<String> = requires_messages
        .iter()
        .map(|message| html_escape(message))
        .collect();
    format!(
        r#"        <div class="global-requires">
            <strong>ℹ️ Requires:</strong> {}
        </div>
"#,
        escaped.join(", ")
    )
}
