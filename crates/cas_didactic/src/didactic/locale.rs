//! Localization for didactic sub-step titles (and other keyed strings).
//!
//! The step-by-step is generated in Spanish (the source language). To support other languages
//! without threading a `Language` into every generator, a generator may emit a sub-step with an
//! i18n KEY plus positional ARGS (`SubStep::keyed`); the wire layer then renders the title for the
//! requested language via [`translate`]. The key table below is the single source of truth for every
//! language, so adding a third language is just another arm.
//!
//! Args are already-rendered, language-neutral fragments (math, numbers) substituted into the
//! template's `{0}`, `{1}`, … placeholders. A key with no template arm falls back to its Spanish
//! template (or, failing that, the raw key) so a partially-migrated table never breaks output.

use cas_solver_core::eval_option_axes::Language;

/// Render the i18n `key` for `lang`, substituting `args` into the template placeholders.
pub(crate) fn translate(key: &str, args: &[&str], lang: Language) -> String {
    let template = template_for(key, lang)
        .or_else(|| template_for(key, Language::Es))
        .unwrap_or(key);
    fill(template, args)
}

/// Substitute `{0}`, `{1}`, … in `template` with `args` (out-of-range placeholders are left as-is).
fn fill(template: &str, args: &[&str]) -> String {
    if args.is_empty() || !template.contains('{') {
        return template.to_string();
    }
    let mut out = String::with_capacity(template.len());
    let mut chars = template.char_indices().peekable();
    while let Some((i, ch)) = chars.next() {
        if ch == '{' {
            // Read the digits until '}'.
            let rest = &template[i + 1..];
            if let Some(end) = rest.find('}') {
                if let Ok(idx) = rest[..end].parse::<usize>() {
                    if let Some(arg) = args.get(idx) {
                        out.push_str(arg);
                    } else {
                        out.push('{');
                        out.push_str(&rest[..end]);
                        out.push('}');
                    }
                    // Advance past the consumed `{..}`.
                    for _ in 0..(end + 1) {
                        chars.next();
                    }
                    continue;
                }
            }
            out.push(ch);
        } else {
            out.push(ch);
        }
    }
    out
}

/// The localized template for `key`, or `None` if the key/language pair is not in the table.
fn template_for(key: &str, lang: Language) -> Option<&'static str> {
    let (es, en) = match key {
        // Integration by parts.
        "by_parts.repeated" => (
            "Usar integración por partes repetida",
            "Use repeated integration by parts",
        ),
        "by_parts.use" => ("Usar integración por partes", "Use integration by parts"),
        "by_parts.choose_u_dv" => ("Elegir u y dv", "Choose u and dv"),
        "by_parts.compute_du_v" => ("Calcular du y v", "Compute du and v"),
        "by_parts.apply_formula" => (
            "Aplicar la fórmula de integración por partes",
            "Apply the integration-by-parts formula",
        ),
        "by_parts.integrate_remaining" => (
            "Integrar el término restante",
            "Integrate the remaining term",
        ),
        _ => return None,
    };
    Some(match lang {
        Language::Es => es,
        Language::En => en,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_static_key_both_languages() {
        assert_eq!(
            translate("by_parts.choose_u_dv", &[], Language::Es),
            "Elegir u y dv"
        );
        assert_eq!(
            translate("by_parts.choose_u_dv", &[], Language::En),
            "Choose u and dv"
        );
    }

    #[test]
    fn unknown_key_passes_through() {
        assert_eq!(translate("not.a.key", &[], Language::En), "not.a.key");
    }

    #[test]
    fn fill_substitutes_positional_args() {
        assert_eq!(fill("Compute Δ = {0}", &["a^2 - b"]), "Compute Δ = a^2 - b");
        assert_eq!(fill("{0} over {1}", &["x", "y"]), "x over y");
        assert_eq!(fill("no args here", &["unused"]), "no args here");
    }
}
