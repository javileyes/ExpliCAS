//! The SCHEMA census: every identity the narration states with metavariables.
//!
//! A sub-step titled «Usar 2·sin(A)·cos(B) = sin(A+B) + sin(A-B)» asserts a
//! THEOREM, not a fact about the expression in front of the reader. The
//! `Equality` arm cannot see that — it compares two sides, not a sentence —
//! and the audit named this the class where the TITLE is what lies.
//!
//! ## Why the sides are `&'static str`, and why that is the whole design
//!
//! A template is a CONSTANT of the source. The first attempt at this census
//! detected templates textually — "both sides mention a metavariable letter" —
//! and drowned: a user expression containing a symbol named `a` is
//! indistinguishable, by text, from a schema over `a`. The fix is the type
//! system: [`schema_substep`]'s sides demand `'static`, so the COMPILER
//! partitions template sites from instance sites. Measured on migration: 100
//! call sites, 35 templates (17 literal + 18 bound through `match`/helper
//! tuples), 65 instances that borrow from per-emission renders and cannot
//! outlive them.
//!
//! ## Why a table and a one-shot test, not a per-emission verifier
//!
//! The truth of `sin(u)²·cos(u)² = sin(2u)²/4` is a property of the template.
//! Re-deciding it at render time proves the same 68 facts once per emission —
//! the guardrail corpora emit schema sub-steps 213 times. The proof runs ONCE,
//! in `every_proven_schema_folds_to_zero`; the runtime claim is a LOOKUP plus
//! a debug-build assertion that no emitter states a schema this census has
//! never adjudicated.
//!
//! ## How the rows were obtained
//!
//! MEASURED, not transcribed: the 78 distinct pairs were extracted from the
//! 35 template sites (resolving the `match`-tuple and helper-fn indirections),
//! then each was fed through the engine as `lhs − rhs` with the metavariables
//! free, with NUMERIC SAMPLING as the only refuter — a residual the
//! simplifier fails to fold is not evidence of falsehood, and two true
//! quadruple-angle rows exist precisely because of that distinction.

/// What the census has adjudicated about a schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaStatus {
    /// `lhs ≡ rhs` with every metavariable FREE; the simplifier folds the
    /// difference to zero. Pinned by `every_proven_schema_folds_to_zero`.
    Proven,
    /// True at every numeric sample, but the simplifier cannot fold the
    /// difference. NOT proven — and not deletable either: the auto-invalidating
    /// test promotes these to `Proven` the day the simplifier learns the fold.
    OpenUnproven,
    /// FALSE as a free identity (numeric counterexample) because some
    /// metavariable is DERIVED from the others: `a·sin(u) + b·cos(u) =
    /// R·sin(u+φ)` holds only with `R = √(a²+b²)`, `tan φ = b/a`. True as the
    /// emitter uses it; needs binding constraints to be checkable.
    DerivedMetavar,
    /// Quantified over a FUNCTION metavariable under a hypothesis
    /// (`f(-u) = f(u)` for even `f`). Not an identity; the hypothesis is the
    /// claim, and it lives with the emitter that established parity.
    FunctionMetavar,
    /// Display notation the parser does not accept (`log_b(c)` subscripts,
    /// `···` chains). True statements with no expression node behind them.
    DisplayNotation,
}

/// One schema exactly as the emitter states it.
#[derive(Debug, Clone, Copy)]
pub struct Schema {
    pub lhs: &'static str,
    pub rhs: &'static str,
    pub status: SchemaStatus,
}

use SchemaStatus::*;

/// The census. Rows are MEASURED; a row added by hand without adjudication is
/// how a table like this rots.
pub const SCHEMATIC_IDENTITIES: &[Schema] = &[
    Schema { lhs: "R·sin(u + φ)", rhs: "a·sin(u) + b·cos(u)", status: DerivedMetavar },
    Schema { lhs: "a·sin(u) + b·cos(u)", rhs: "R·sin(u + φ)", status: DerivedMetavar },
    Schema { lhs: "log_b(a) · log_a(c)", rhs: "log_b(c)", status: DisplayNotation },
    Schema { lhs: "log_b(c)", rhs: "log_a(c) · log_b(a)", status: DisplayNotation },
    Schema { lhs: "log_{u0}(u1) · log_{u1}(u2) · ... · log_{u_{n-1}}(u_n)", rhs: "log_{u0}(u_n)", status: DisplayNotation },
    Schema { lhs: "log_{u0}(u_n)", rhs: "log_{u0}(u1) · log_{u1}(u2) · ... · log_{u_{n-1}}(u_n)", status: DisplayNotation },
    Schema { lhs: "f(-u)", rhs: "-f(u)", status: FunctionMetavar },
    Schema { lhs: "f(-u)", rhs: "f(u)", status: FunctionMetavar },
    Schema { lhs: "4 · sin(u) · cos(u)^3 - 4 · sin(u)^3 · cos(u)", rhs: "sin(4u)", status: OpenUnproven },
    Schema { lhs: "sin(4u)", rhs: "4 · sin(u) · cos(u)^3 - 4 · sin(u)^3 · cos(u)", status: OpenUnproven },
    Schema { lhs: "((k + 1) · k!) / k!", rhs: "k + 1", status: Proven },
    Schema { lhs: "(3 · tan(u) - tan(u)^3) / (1 - 3 · tan(u)^2)", rhs: "tan(3u)", status: Proven },
    Schema { lhs: "(e^A)^n", rhs: "e^(n·A)", status: Proven },
    Schema { lhs: "(k + 1)! / k!", rhs: "((k + 1) · k!) / k!", status: Proven },
    Schema { lhs: "(tanh(A) + tanh(B)) / (1 + tanh(A)·tanh(B))", rhs: "tanh(A+B)", status: Proven },
    Schema { lhs: "(tanh(A) - tanh(B)) / (1 - tanh(A)·tanh(B))", rhs: "tanh(A-B)", status: Proven },
    Schema { lhs: "1/e^A", rhs: "e^(-A)", status: Proven },
    Schema { lhs: "16 · cos(u)^5 - 20 · cos(u)^3 + 5 · cos(u)", rhs: "cos(5u)", status: Proven },
    Schema { lhs: "2·cos(A)·cos(B)", rhs: "cos(A+B) + cos(A-B)", status: Proven },
    Schema { lhs: "2·cos(A)·sin(B)", rhs: "sin(A+B) - sin(A-B)", status: Proven },
    Schema { lhs: "2·cosh((A+B)/2)·cosh((A-B)/2)", rhs: "cosh(A)+cosh(B)", status: Proven },
    Schema { lhs: "2·cosh((A+B)/2)·sinh((A-B)/2)", rhs: "sinh(A)-sinh(B)", status: Proven },
    Schema { lhs: "2·cosh(A)·cosh(B)", rhs: "cosh(A+B) + cosh(A-B)", status: Proven },
    Schema { lhs: "2·sin(A)·cos(B)", rhs: "sin(A+B) + sin(A-B)", status: Proven },
    Schema { lhs: "2·sin(A)·sin(B)", rhs: "cos(A-B) - cos(A+B)", status: Proven },
    Schema { lhs: "2·sinh((A+B)/2)·cosh((A-B)/2)", rhs: "sinh(A)+sinh(B)", status: Proven },
    Schema { lhs: "2·sinh((A+B)/2)·sinh((A-B)/2)", rhs: "cosh(A)-cosh(B)", status: Proven },
    Schema { lhs: "2·sinh(A)·cosh(B)", rhs: "sinh(A+B) + sinh(A-B)", status: Proven },
    Schema { lhs: "2·sinh(A)·sinh(B)", rhs: "cosh(A+B) - cosh(A-B)", status: Proven },
    Schema { lhs: "3 · sin(u) - 4 · sin(u)^3", rhs: "sin(3u)", status: Proven },
    Schema { lhs: "4 · cos(u)^3 - 3 · cos(u)", rhs: "cos(3u)", status: Proven },
    Schema { lhs: "5 · sin(u) - 20 · sin(u)^3 + 16 · sin(u)^5", rhs: "sin(5u)", status: Proven },
    Schema { lhs: "8 · cos(u)^4 - 8 · cos(u)^2 + 1", rhs: "cos(4u)", status: Proven },
    Schema { lhs: "cos(3u)", rhs: "4 · cos(u)^3 - 3 · cos(u)", status: Proven },
    Schema { lhs: "cos(4u)", rhs: "8 · cos(u)^4 - 8 · cos(u)^2 + 1", status: Proven },
    Schema { lhs: "cos(5u)", rhs: "16 · cos(u)^5 - 20 · cos(u)^3 + 5 · cos(u)", status: Proven },
    Schema { lhs: "cos(A) · cos(B) + sin(A) · sin(B)", rhs: "cos(A-B)", status: Proven },
    Schema { lhs: "cos(A) · cos(B) - sin(A) · sin(B)", rhs: "cos(A+B)", status: Proven },
    Schema { lhs: "cos(A+B)", rhs: "cos(A) · cos(B) - sin(A) · sin(B)", status: Proven },
    Schema { lhs: "cos(A-B)", rhs: "cos(A) · cos(B) + sin(A) · sin(B)", status: Proven },
    Schema { lhs: "cos(u - φ)", rhs: "sin(u + (π/2 - φ))", status: Proven },
    Schema { lhs: "cos(u)^2", rhs: "(1 + cos(2u)) / 2", status: Proven },
    Schema { lhs: "cosh(A) · cosh(B) + sinh(A) · sinh(B)", rhs: "cosh(A+B)", status: Proven },
    Schema { lhs: "cosh(A) · cosh(B) - sinh(A) · sinh(B)", rhs: "cosh(A-B)", status: Proven },
    Schema { lhs: "cosh(A)+cosh(B)", rhs: "2·cosh((A+B)/2)·cosh((A-B)/2)", status: Proven },
    Schema { lhs: "cosh(A)-cosh(B)", rhs: "2·sinh((A+B)/2)·sinh((A-B)/2)", status: Proven },
    Schema { lhs: "cosh(A+B)", rhs: "cosh(A) · cosh(B) + sinh(A) · sinh(B)", status: Proven },
    Schema { lhs: "cosh(A+B) + cosh(A-B)", rhs: "2·cosh(A)·cosh(B)", status: Proven },
    Schema { lhs: "cosh(A+B) - cosh(A-B)", rhs: "2·sinh(A)·sinh(B)", status: Proven },
    Schema { lhs: "cosh(A-B)", rhs: "cosh(A) · cosh(B) - sinh(A) · sinh(B)", status: Proven },
    Schema { lhs: "e^(-A)", rhs: "1/e^A", status: Proven },
    Schema { lhs: "e^(A+B)", rhs: "e^A · e^B", status: Proven },
    Schema { lhs: "e^(A-B)", rhs: "e^A / e^B", status: Proven },
    Schema { lhs: "e^(n·A)", rhs: "(e^A)^n", status: Proven },
    Schema { lhs: "e^A / e^B", rhs: "e^(A-B)", status: Proven },
    Schema { lhs: "e^A · e^B", rhs: "e^(A+B)", status: Proven },
    Schema { lhs: "sin(3u)", rhs: "3 · sin(u) - 4 · sin(u)^3", status: Proven },
    Schema { lhs: "sin(5u)", rhs: "5 · sin(u) - 20 · sin(u)^3 + 16 · sin(u)^5", status: Proven },
    Schema { lhs: "sin(A) · cos(B) + cos(A) · sin(B)", rhs: "sin(A+B)", status: Proven },
    Schema { lhs: "sin(A) · cos(B) - cos(A) · sin(B)", rhs: "sin(A-B)", status: Proven },
    Schema { lhs: "sin(A+B)", rhs: "sin(A) · cos(B) + cos(A) · sin(B)", status: Proven },
    Schema { lhs: "sin(A-B)", rhs: "sin(A) · cos(B) - cos(A) · sin(B)", status: Proven },
    Schema { lhs: "sin(u + φ)", rhs: "cos(u - (π/2 - φ))", status: Proven },
    Schema { lhs: "sin(u)^2", rhs: "(1 - cos(2u)) / 2", status: Proven },
    Schema { lhs: "sin(u)^2 · cos(u)^2", rhs: "(1 - cos(4u)) / 8", status: Proven },
    Schema { lhs: "sin(u)^2 · cos(u)^2", rhs: "sin(2u)^2 / 4", status: Proven },
    Schema { lhs: "sinh(A) · cosh(B) + cosh(A) · sinh(B)", rhs: "sinh(A+B)", status: Proven },
    Schema { lhs: "sinh(A) · cosh(B) - cosh(A) · sinh(B)", rhs: "sinh(A-B)", status: Proven },
    Schema { lhs: "sinh(A)+sinh(B)", rhs: "2·sinh((A+B)/2)·cosh((A-B)/2)", status: Proven },
    Schema { lhs: "sinh(A)-sinh(B)", rhs: "2·cosh((A+B)/2)·sinh((A-B)/2)", status: Proven },
    Schema { lhs: "sinh(A+B)", rhs: "sinh(A) · cosh(B) + cosh(A) · sinh(B)", status: Proven },
    Schema { lhs: "sinh(A+B) + sinh(A-B)", rhs: "2·sinh(A)·cosh(B)", status: Proven },
    Schema { lhs: "sinh(A-B)", rhs: "sinh(A) · cosh(B) - cosh(A) · sinh(B)", status: Proven },
    Schema { lhs: "sinh(u) / cosh(u)", rhs: "tanh(u)", status: Proven },
    Schema { lhs: "tan(3u)", rhs: "(3 · tan(u) - tan(u)^3) / (1 - 3 · tan(u)^2)", status: Proven },
    Schema { lhs: "tanh(A+B)", rhs: "(tanh(A) + tanh(B)) / (1 + tanh(A)·tanh(B))", status: Proven },
    Schema { lhs: "tanh(A-B)", rhs: "(tanh(A) - tanh(B)) / (1 - tanh(A)·tanh(B))", status: Proven },
    Schema { lhs: "tanh(u)", rhs: "sinh(u) / cosh(u)", status: Proven },
];

/// What the census says about a schema, or `None` when it has never been
/// adjudicated — a DECLARED gap, never a pass.
pub fn schema_status(lhs: &str, rhs: &str) -> Option<SchemaStatus> {
    SCHEMATIC_IDENTITIES
        .iter()
        .find(|s| s.lhs == lhs && s.rhs == rhs)
        .map(|s| s.status)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Display notation is not parser input. `sin²(u)` means `sin(u)^2`, and
    /// reading it as `sin^2(u)` makes the parser see `sin^(2·u)` — a mistake
    /// that silently turned two TRUE schemas into refutations while this
    /// census was being measured. One normalizer, tested on its own.
    fn to_parser_input(display: &str) -> String {
        let mut out = display
            .replace('·', "*")
            .replace(['−', '–'], "-")
            .replace('φ', "phi")
            .replace('π', "pi")
            .replace('√', "sqrt");
        for (marker, exponent) in [('²', "2"), ('³', "3")] {
            while let Some(at) = out.find(marker) {
                let head = &out[..at];
                let rest = &out[at + marker.len_utf8()..];
                if let Some(stripped) = rest.strip_prefix('(') {
                    let mut depth = 1usize;
                    let mut end = stripped.len();
                    for (i, c) in stripped.char_indices() {
                        match c {
                            '(' => depth += 1,
                            ')' => {
                                depth -= 1;
                                if depth == 0 {
                                    end = i;
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    out = format!("{head}({})^{exponent}{}", &stripped[..end], &stripped[end + 1..]);
                } else {
                    out = format!("{head}^{exponent}{rest}");
                }
            }
        }
        out
    }

    #[test]
    fn the_normalizer_reads_a_squared_function_as_a_squared_call() {
        assert_eq!(to_parser_input("sin²(u)·cos²(u)"), "sin(u)^2*cos(u)^2");
        assert_eq!(to_parser_input("cosh²(u/2)"), "cosh(u/2)^2");
        assert_eq!(to_parser_input("a² + b²"), "a^2 + b^2");
    }

    fn folds_to_zero(schema: &Schema) -> Option<bool> {
        let mut context = cas_ast::Context::new();
        let lhs = cas_parser::parse(&to_parser_input(schema.lhs), &mut context).ok()?;
        let rhs = cas_parser::parse(&to_parser_input(schema.rhs), &mut context).ok()?;
        let difference = context.add(cas_ast::Expr::Sub(lhs, rhs));
        let mut simplifier = cas_solver::runtime::Simplifier::with_default_rules();
        std::mem::swap(&mut simplifier.context, &mut context);
        let (folded, _, _) = cas_engine::with_suppressed_depth_overflow_warnings(|| {
            simplifier.simplify_with_stats(
                difference,
                cas_solver::runtime::SimplifyOptions {
                    time_budget_ms: Some(2_000),
                    ..Default::default()
                },
            )
        });
        std::mem::swap(&mut simplifier.context, &mut context);
        Some(
            matches!(context.get(folded), cas_ast::Expr::Number(n) if num_traits::Zero::is_zero(n)),
        )
    }

    /// The proof, run ONCE for the whole census instead of once per emission.
    #[test]
    fn every_proven_schema_folds_to_zero() {
        let broken: Vec<String> = SCHEMATIC_IDENTITIES
            .iter()
            .filter(|s| s.status == Proven)
            .filter(|s| folds_to_zero(s) != Some(true))
            .map(|s| format!("{}  ⇒  {}", s.lhs, s.rhs))
            .collect();
        assert!(
            broken.is_empty(),
            "{} schemas are declared Proven but do not fold to zero: {broken:#?}",
            broken.len()
        );
    }

    /// AUTO-INVALIDATING, the property that keeps a census from rotting into an
    /// excuse list: a row that stops being an exception BREAKS this test and
    /// must be promoted to `Proven` in the same commit that improved the
    /// simplifier. Without it the table would keep claiming gaps that no
    /// longer exist.
    #[test]
    fn no_unproven_schema_silently_became_provable() {
        let promotable: Vec<String> = SCHEMATIC_IDENTITIES
            .iter()
            .filter(|s| s.status != Proven)
            .filter(|s| folds_to_zero(s) == Some(true))
            .map(|s| format!("{:?}: {}  ⇒  {}", s.status, s.lhs, s.rhs))
            .collect();
        assert!(
            promotable.is_empty(),
            "these schemas now PROVE and must be promoted to Proven in the \
             commit that made them fold: {promotable:#?}"
        );
    }

    /// `OpenUnproven` must never shelter a falsehood: each such row is
    /// re-checked NUMERICALLY here, and a clear nonzero refutes it. Same
    /// discipline as the claim verifiers: refutation requires a POSITIVE
    /// witness, and "the simplifier did not fold" is not one.
    #[test]
    fn open_unproven_schemas_hold_at_numeric_samples() {
        // Every metavariable the census uses, with two sample points each.
        let vars: Vec<String> = ["u", "A", "B", "a", "b", "c", "n", "k", "m", "x"]
            .iter()
            .map(|v| v.to_string())
            .collect();
        let samples = [
            [0.4285, 0.4, 0.3333, 2.0, 3.0, 5.0, 4.0, 3.0, 2.0, 0.5],
            [-0.6667, 0.1428, -0.6, 3.0, 5.0, 2.0, 3.0, 2.0, 1.0, -0.3333],
        ];
        for schema in SCHEMATIC_IDENTITIES {
            if schema.status != OpenUnproven {
                continue;
            }
            let mut context = cas_ast::Context::new();
            let lhs = cas_parser::parse(&to_parser_input(schema.lhs), &mut context)
                .expect("OpenUnproven lhs must parse — otherwise it is DisplayNotation");
            let rhs = cas_parser::parse(&to_parser_input(schema.rhs), &mut context)
                .expect("OpenUnproven rhs must parse — otherwise it is DisplayNotation");
            for values in &samples {
                let (lv, rv) = (
                    cas_math::numeric_eval::eval_f64_with_substitution(
                        &context, lhs, &vars, values,
                    ),
                    cas_math::numeric_eval::eval_f64_with_substitution(
                        &context, rhs, &vars, values,
                    ),
                );
                if let (Some(lv), Some(rv)) = (lv, rv) {
                    assert!(
                        (lv - rv).abs() < 1e-9,
                        "OpenUnproven schema is numerically FALSE: {}  ⇒  {}  \
                         ({lv} vs {rv})",
                        schema.lhs,
                        schema.rhs
                    );
                }
            }
        }
    }

    /// A census with duplicate keys would make `schema_status` answer by
    /// whichever row came first — a lookup that is not a function.
    #[test]
    fn the_census_has_no_duplicate_pairs() {
        let mut seen = std::collections::BTreeSet::new();
        for schema in SCHEMATIC_IDENTITIES {
            assert!(
                seen.insert((schema.lhs, schema.rhs)),
                "duplicate schema row: {}  ⇒  {}",
                schema.lhs,
                schema.rhs
            );
        }
    }

    #[test]
    fn an_unadjudicated_schema_is_a_declared_gap_not_a_pass() {
        assert_eq!(schema_status("nunca(u)", "medido(u)"), None);
    }
}
