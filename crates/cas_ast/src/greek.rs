//! Greek-letter aliases shared by the parser and the formatters.
//!
//! Single-glyph Greek letters are accepted as INPUT aliases of their spelled
//! names — `α` lexes as the identifier `alpha`, exactly like `sen` aliases
//! `sin` at the builtin level — so internally ONE canonical symbol exists and
//! `α` ≡ `alpha` everywhere (and `π`/`φ` reach the existing constant arms).
//! On OUTPUT, `latex_variable_name` maps the spelled names back to the real
//! LaTeX commands, so both spellings render as proper Greek in MathJax.
//!
//! Deliberate exclusions — lookalike traps, where a silently-accepted glyph
//! that is a visually identical but DIFFERENT symbol is worse than an honest
//! parse error:
//! - lowercase omicron `ο` (identical to Latin `o`)
//! - uppercase Alpha/Beta/Epsilon/Zeta/Eta/Iota/Kappa/Mu/Nu/Omicron/Rho/Tau
//!   (identical to Latin A/B/E/Z/H/I/K/M/N/O/P/T) and Upsilon (Y)
//!
//! Robustness aliases folded into the same canonical name:
//! - `ς` (final sigma) → `sigma` (the same letter, word-final form)
//! - `µ` MICRO SIGN U+00B5 → `mu` (what many keyboards emit for micro)
//! - `ϕ`/`ϑ`/`ϵ` symbol variants → `phi`/`theta`/`epsilon`

/// Canonical (glyph, spelled name, LaTeX command) table.
///
/// The spelled name is the internal symbol identity; the LaTeX command is
/// what `latex_variable_name` emits for that name. Alias glyphs (ς, µ, ϕ,
/// ϑ, ϵ) appear as extra rows pointing at the same name, so glyph→name is
/// many-to-one while name→LaTeX stays one-to-one (first row wins).
const GREEK: &[(char, &str, &str)] = &[
    ('α', "alpha", "\\alpha"),
    ('β', "beta", "\\beta"),
    ('γ', "gamma", "\\gamma"),
    ('δ', "delta", "\\delta"),
    ('ε', "epsilon", "\\epsilon"),
    ('ϵ', "epsilon", "\\epsilon"),
    ('ζ', "zeta", "\\zeta"),
    ('η', "eta", "\\eta"),
    ('θ', "theta", "\\theta"),
    ('ϑ', "theta", "\\theta"),
    ('ι', "iota", "\\iota"),
    ('κ', "kappa", "\\kappa"),
    ('λ', "lambda", "\\lambda"),
    ('μ', "mu", "\\mu"),
    ('µ', "mu", "\\mu"), // MICRO SIGN U+00B5
    ('ν', "nu", "\\nu"),
    ('ξ', "xi", "\\xi"),
    ('π', "pi", "\\pi"),
    ('ρ', "rho", "\\rho"),
    ('σ', "sigma", "\\sigma"),
    ('ς', "sigma", "\\sigma"),
    ('τ', "tau", "\\tau"),
    ('υ', "upsilon", "\\upsilon"),
    ('φ', "phi", "\\phi"),
    ('ϕ', "phi", "\\phi"),
    ('χ', "chi", "\\chi"),
    ('ψ', "psi", "\\psi"),
    ('ω', "omega", "\\omega"),
    // Uppercase: only the glyphs visually distinct from Latin capitals.
    ('Γ', "Gamma", "\\Gamma"),
    ('Δ', "Delta", "\\Delta"),
    ('Θ', "Theta", "\\Theta"),
    ('Λ', "Lambda", "\\Lambda"),
    ('Ξ', "Xi", "\\Xi"),
    ('Π', "Pi", "\\Pi"),
    ('Σ', "Sigma", "\\Sigma"),
    ('Φ', "Phi", "\\Phi"),
    ('Ψ', "Psi", "\\Psi"),
    ('Ω', "Omega", "\\Omega"),
];

/// Spelled name for a Greek glyph accepted as an input alias (`α` → `alpha`).
/// `None` for everything else, including the deliberate lookalike exclusions.
pub fn greek_glyph_name(c: char) -> Option<&'static str> {
    GREEK.iter().find(|(g, _, _)| *g == c).map(|(_, n, _)| *n)
}

/// LaTeX command for a spelled Greek name (`alpha` → `\alpha`), used by the
/// variable renderer so `α`-typed and `alpha`-typed symbols BOTH display as
/// real Greek. Exact whole-name match only (`beta1` stays `beta1`).
pub fn greek_name_latex(name: &str) -> Option<&'static str> {
    GREEK.iter().find(|(_, n, _)| *n == name).map(|(_, _, l)| *l)
}

/// Canonicalize a raw var/func TOKEN from the command wire: a token that is
/// exactly one Greek glyph (after trimming) becomes its spelled name, and
/// anything else passes through unchanged. Command layers that carry
/// variable NAMES as strings (solve's var, limit's var, dsolve's func,
/// solve_system's unknown list) must apply this so the name matches the
/// symbol the expression parser interned (`λ` ≡ `lambda`).
pub fn canonical_greek_token(token: &str) -> &str {
    let trimmed = token.trim();
    let mut chars = trimmed.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => greek_glyph_name(c).unwrap_or(token),
        _ => token,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_aliases_resolve_to_spelled_names() {
        assert_eq!(greek_glyph_name('α'), Some("alpha"));
        assert_eq!(greek_glyph_name('π'), Some("pi"));
        assert_eq!(greek_glyph_name('φ'), Some("phi"));
        // Robustness aliases fold into the same canonical letter.
        assert_eq!(greek_glyph_name('ς'), Some("sigma"));
        assert_eq!(greek_glyph_name('µ'), Some("mu")); // U+00B5 MICRO SIGN
        assert_eq!(greek_glyph_name('ϕ'), Some("phi"));
        // Uppercase distinct glyphs map to capitalized names.
        assert_eq!(greek_glyph_name('Δ'), Some("Delta"));
        // Lookalike exclusions stay rejected: omicron and Latin letters.
        assert_eq!(greek_glyph_name('ο'), None);
        assert_eq!(greek_glyph_name('o'), None);
        assert_eq!(greek_glyph_name('x'), None);
    }

    #[test]
    fn spelled_names_map_to_latex_commands() {
        assert_eq!(greek_name_latex("alpha"), Some("\\alpha"));
        assert_eq!(greek_name_latex("Omega"), Some("\\Omega"));
        // Exact whole-name match only — suffixed names are untouched.
        assert_eq!(greek_name_latex("beta1"), None);
        assert_eq!(greek_name_latex("x"), None);
    }

    #[test]
    fn canonical_token_maps_single_glyphs_only() {
        assert_eq!(canonical_greek_token("λ"), "lambda");
        assert_eq!(canonical_greek_token(" θ "), "theta");
        assert_eq!(canonical_greek_token("x"), "x");
        assert_eq!(canonical_greek_token("lambda"), "lambda");
        assert_eq!(canonical_greek_token("αβ"), "αβ"); // not a single glyph
    }
}
