//! Helper compartido de los binarios de contrato de inecuaciones.
//!
//! Vive en su propio módulo (y no en `test_utils`) a propósito: los wrappers
//! de compatibilidad de cas_engine incluyen `test_utils` bajo
//! `extern crate cas_engine as cas_solver`, así que todo lo que entre allí
//! debe resolver bajo AMBAS identidades — y `display_solution_set` no existe
//! en la superficie del engine. Este módulo solo lo incluyen binarios de
//! cas_solver, sin alias.

#![allow(dead_code)]

use cas_parser::parse;
use cas_solver::runtime::Simplifier;

/// Resolver `lhs <op> rhs` para `x` y renderizar el conjunto solución.
///
/// Definición COMPARTIDA: eran 8 copias byte-idénticas (ledger L13 — el diffeo
/// previo separó este cluster de las otras 6 variantes de `solve_display`,
/// que tienen semánticas propias y NO se fusionan).
pub fn solve_display(lhs: &str, op: cas_ast::RelOp, rhs: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse(lhs, &mut simplifier.context).expect("parse lhs");
    let rhs = parse(rhs, &mut simplifier.context).expect("parse rhs");
    let eq = cas_ast::Equation { lhs, rhs, op };
    let (set, _) = cas_solver::api::solve(&eq, "x", &mut simplifier).expect("solve");
    cas_solver::command_api::solve::display_solution_set(&simplifier.context, &set)
}
