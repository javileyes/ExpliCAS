//! Contract tests for reciprocal-square trig equations `A/trig(x)^2 = c`
//! (2026-07-13b, family F5 of docs/AUDITORIA_FRONTERA_2026-07-13b.md). `sec(x)^2`,
//! `csc(x)^2`, `1/cos(x)^2`, `1/sin(x)^2` all canonicalize to
//! `Div(A, Pow(cos|sin(g), 2))`, which the bare-squared reducer never matched, so
//! the generic isolation returned only the finite principal-value roots and
//! dropped the periodic family. The reciprocal is now inverted to the equivalent
//! `trig(g)^2 = -A/k` and fed to the existing double-angle reducer. Answers match
//! sympy's `solveset`.

use cas_ast::{Equation, RelOp};
use cas_parser::parse;
use cas_solver::api::solve;
use cas_solver::command_api::solve::display_solution_set;
use cas_solver::runtime::Simplifier;

fn solve_display(lhs: &str, rhs: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse(lhs, &mut simplifier.context).expect("parse lhs");
    let rhs = parse(rhs, &mut simplifier.context).expect("parse rhs");
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (set, _) = solve(&eq, "x", &mut simplifier).expect("solve");
    // Normalize the multiplication separator (the harness mixes ` * ` and `·`) so the
    // assertions are render-agnostic.
    display_solution_set(&simplifier.context, &set)
        .replace(" * ", "·")
        .replace('*', "·")
}

#[test]
fn reciprocal_square_trig_emits_the_full_periodic_family() {
    // `1/cos^2 = 2` <=> `cos^2 = 1/2` -> {pi/4 + k*pi/2}.
    assert_eq!(
        solve_display("1/cos(x)^2", "2"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("1/sin(x)^2", "2"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    // sec^2 / csc^2 canonicalize to the same Div form.
    assert_eq!(
        solve_display("sec(x)^2", "4"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("csc(x)^2", "4"),
        "{ 1/6·pi + k·pi, 5/6·pi + k·pi : k ∈ ℤ }"
    );
    // Constant-shifted form `sec^2 - 2 = 0` handled on the difference.
    assert_eq!(
        solve_display("sec(x)^2 - 2", "0"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("csc(x)^2 - 2", "0"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
}

#[test]
fn direct_and_power_one_trig_forms_are_unchanged() {
    // The direct squared forms this reduces TO are unchanged.
    assert_eq!(
        solve_display("cos(x)^2", "1/2"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("cos(x)^2", "1/4"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    // Power-1 reciprocal (sec = 2) keeps its own 2*pi-period owner.
    assert_eq!(
        solve_display("sec(x)", "2"),
        "{ 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // Boundary sin^2 = 1 single-family case unchanged.
    assert_eq!(solve_display("sin(x)^2", "1"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
}

#[test]
fn coefficient_and_even_power_spellings_keep_the_periodic_family() {
    // SOUNDNESS (auditoría 2026-07-30, ficha S1c-001): el coeficiente ≠1
    // simplificaba a `Mul(c, Div(1, cos²))` — el matcher F5 exigía un `Div`
    // desnudo, no casaba, y el fallback de inverso unario emitía SOLO las
    // raíces principales finitas (`{π/3, 2π/3}`) perdiendo la familia
    // periódica entera. Tres formas estructurales medidas: producto por
    // coeficiente, potencia par 2m, y constante en el denominador.
    assert_eq!(
        solve_display("2*sec(x)^2", "8"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("3*sec(x)^2", "12"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("2*csc(x)^2", "8"),
        "{ 1/6·pi + k·pi, 5/6·pi + k·pi : k ∈ ℤ }"
    );
    // Potencia par: sec⁴ = 16 ⟺ sec² = 4 (la rama negativa no existe:
    // potencia par ≥ 0).
    assert_eq!(
        solve_display("sec(x)^4", "16"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    // ⚠️ `sec²/2 = 2` reduce a cos² = 1/4 — la MISMA familia que sec² = 4,
    // no la de cos² = 1/2 (aviso literal de la aceptación de la ficha).
    assert_eq!(
        solve_display("sec(x)^2/2", "2"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    // Argumento escalado: el periodo se divide con el argumento.
    assert_eq!(
        solve_display("2*sec(2*x)^2", "8"),
        "{ 1/6·pi + k·1/2·pi, 1/3·pi + k·1/2·pi : k ∈ ℤ }"
    );
}

#[test]
fn impossible_even_power_targets_are_empty_not_finite() {
    // Potencia par de un real jamás es negativa: el conjunto vacío es una
    // AFIRMACIÓN con argumento de completitud (no un decline). Antes el
    // fallback podía fabricar raíces de arccos fuera de dominio.
    assert_eq!(solve_display("2*sec(x)^2", "-8"), "No solution");
    assert_eq!(solve_display("sec(x)^4", "-16"), "No solution");
}
