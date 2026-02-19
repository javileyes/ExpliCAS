//! Compatibility shim: LaTeX identity cleanup now lives in `cas_formatter`.

pub(super) use cas_formatter::latex_clean::clean_latex_identities;
