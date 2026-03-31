pub(super) type TemplateReplacement<'a> = (&'a str, &'a str);

pub(super) fn render_static_template(
    template: &str,
    replacements: &[TemplateReplacement<'_>],
) -> String {
    replacements
        .iter()
        .fold(template.to_string(), |acc, (key, value)| {
            acc.replace(key, value)
        })
}

pub(super) fn push_rendered_static_template(
    out: &mut String,
    template: &str,
    replacements: &[TemplateReplacement<'_>],
) {
    out.push_str(&render_static_template(template, replacements));
}

macro_rules! timeline_asset {
    ($relative:literal) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/",
            $relative
        ))
    };
}

pub(super) use timeline_asset;

macro_rules! render_timeline_asset {
    ($relative:literal, $replacements:expr $(,)?) => {
        $crate::timeline::render_template::render_static_template(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/timeline/",
                $relative
            )),
            $replacements,
        )
    };
}

pub(super) use render_timeline_asset;

macro_rules! push_timeline_asset {
    ($out:expr, $relative:literal, $replacements:expr $(,)?) => {
        $crate::timeline::render_template::push_rendered_static_template(
            $out,
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/timeline/",
                $relative
            )),
            $replacements,
        )
    };
}

pub(super) use push_timeline_asset;

#[cfg(test)]
mod tests {
    use super::{push_rendered_static_template, render_static_template, TemplateReplacement};

    #[test]
    fn render_static_template_replaces_all_placeholders() {
        let replacements: [TemplateReplacement<'_>; 2] =
            [("__NAME__", "timeline"), ("__VALUE__", "42")];
        let rendered = render_static_template("Hello __NAME__ = __VALUE__", &replacements);
        assert_eq!(rendered, "Hello timeline = 42");
    }

    #[test]
    fn push_rendered_static_template_appends_fragment() {
        let replacements: [TemplateReplacement<'_>; 1] = [("__ITEM__", "done")];
        let mut out = String::from("before:");
        push_rendered_static_template(&mut out, "__ITEM__", &replacements);
        assert_eq!(out, "before:done");
    }
}
