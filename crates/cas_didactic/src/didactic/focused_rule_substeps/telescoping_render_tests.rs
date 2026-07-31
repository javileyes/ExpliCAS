//! Tests de `focused_rule_substeps`: `telescoping_render_tests`, extraídos del módulo.

use super::{group_factor_latex, group_factor_plain, render_power2_plain};

/// Audit 2026-07-30 ficha D2-003: `render_power2_plain("1 + 2")` published
/// `1 + 2^2` (= 5) while its LaTeX twin said `(1+2)^2` (= 9) in the same
/// substep. A composite base must be grouped before `^` is appended.
#[test]
fn power2_plain_groups_composite_bases_only() {
    assert_eq!(render_power2_plain("1 + 2"), "(1 + 2)^2");
    assert_eq!(render_power2_plain("m + 1"), "(m + 1)^2");
    assert_eq!(render_power2_plain("k^2"), "(k^2)^2");
    assert_eq!(render_power2_plain("n"), "n^2");
    assert_eq!(render_power2_plain("2"), "2^2");
}

#[test]
fn factor_grouping_wraps_composites_and_leaves_atoms() {
    assert_eq!(group_factor_plain("2 - 1"), "(2 - 1)");
    assert_eq!(group_factor_plain("n"), "n");
    assert_eq!(group_factor_latex("n + 1"), "\\left(n + 1\\right)");
    assert_eq!(group_factor_latex("m"), "m");
}
