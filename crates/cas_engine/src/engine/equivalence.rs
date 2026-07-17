//! Equivalence checking between mathematical expressions.
//!
//! Provides both simple boolean equivalence (`are_equivalent`) and
//! extended tri-state equivalence with domain conditions (`are_equivalent_extended`).

use super::{eval_f64, EquivalenceResult, Simplifier};
use cas_ast::{Expr, ExprId};
use num_traits::Zero;

fn expr_contains_calculus_call(ctx: &cas_ast::Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Function(fn_id, args) => {
            matches!(ctx.sym_name(*fn_id), "diff" | "integrate" | "limit")
                || args
                    .iter()
                    .any(|arg| expr_contains_calculus_call(ctx, *arg))
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            expr_contains_calculus_call(ctx, *left) || expr_contains_calculus_call(ctx, *right)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_calculus_call(ctx, *inner),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|entry| expr_contains_calculus_call(ctx, *entry)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

impl Simplifier {
    pub fn are_equivalent(&mut self, a: ExprId, b: ExprId) -> bool {
        let diff = self.context.add(Expr::Sub(a, b));
        if expr_contains_calculus_call(&self.context, a)
            || expr_contains_calculus_call(&self.context, b)
        {
            let (direct_diff, _) = self.simplify(diff);
            if matches!(self.context.get(direct_diff), Expr::Number(n) if n.is_zero()) {
                return true;
            }
        }

        let expand_id = self.context.intern_symbol("expand");
        let expanded_diff = self.context.add(Expr::Function(expand_id, vec![diff]));
        let (simplified_diff, _) = self.simplify(expanded_diff);

        let result_expr = {
            let expr = self.context.get(simplified_diff);
            if let Expr::Function(fn_id, args) = expr {
                let name = self.context.sym_name(*fn_id);
                if name == "expand" && args.len() == 1 {
                    args[0]
                } else {
                    simplified_diff
                }
            } else {
                simplified_diff
            }
        };

        let expr = self.context.get(result_expr);
        match expr {
            Expr::Number(n) => n.is_zero(),
            _ => {
                let (direct_diff, _) = self.simplify(diff);
                if matches!(self.context.get(direct_diff), Expr::Number(n) if n.is_zero()) {
                    return true;
                }

                if !self.allow_numerical_verification {
                    return false;
                }
                let vars = cas_ast::collect_variables(&self.context, result_expr);
                let var_map = cas_solver_core::equivalence::default_equiv_probe_map(vars);

                if let Some(val) = eval_f64(&self.context, result_expr, &var_map) {
                    cas_solver_core::equivalence::is_numeric_equiv_zero(val)
                } else {
                    false
                }
            }
        }
    }

    /// Extended equivalence check returning tri-state result with domain conditions.
    ///
    /// V2.14.45: This method uses the same simplifier pipeline as the REPL,
    /// and properly interprets SoundnessLabel and Requires from rules.
    ///
    /// Returns:
    /// - `True` if A-B simplifies to 0 with pure Equivalence rules
    /// - `ConditionalTrue` if A-B simplifies to 0 but rules with
    ///   EquivalenceUnderIntroducedRequires were used or domain conditions introduced
    /// - `Unknown` if cannot simplify to 0 but no counterexample found
    /// - `False` if numeric verification finds counterexample
    pub fn are_equivalent_extended(&mut self, a: ExprId, b: ExprId) -> EquivalenceResult {
        use crate::rule::SoundnessLabel;
        use cas_math::semantic_equality::SemanticEqualityChecker;

        // Enable step collection to track soundness labels
        let was_collecting = self.collect_steps();
        self.set_collect_steps(true);

        // =================================================================
        // OPTION 2: Normal forms comparison
        // Simplify A and B separately, then compare.
        // This catches cases like tan(x)*tan(pi/3-x)*tan(pi/3+x) ≡ tan(3x)
        // where both simplify to tan(3x) but diff doesn't cancel due to
        // expansion rules firing before cancellation.
        // =================================================================
        let (simplified_a, steps_a) = self.simplify(a);

        // Early check: compare simplified_a with b (before simplifying b)
        // This catches cases where A simplifies to exactly B
        // e.g., tan(x)*tan(pi/3-x)*tan(pi/3+x) → tan(3x) ≡ tan(3*x)
        let checker = SemanticEqualityChecker::new(&self.context);
        if checker.are_equal(simplified_a, b) {
            let has_conditional_rules = steps_a
                .iter()
                .any(|step| step.soundness != SoundnessLabel::Equivalence);
            let mut requires: Vec<String> = Vec::new();

            for step in &steps_a {
                for req in step.required_conditions() {
                    let condition_str = req.display(&self.context);
                    if !requires.contains(&condition_str) {
                        requires.push(condition_str);
                    }
                }
            }

            self.set_collect_steps(was_collecting);
            cas_solver_core::equivalence::normalize_requires(&mut requires);

            return if has_conditional_rules || !requires.is_empty() {
                EquivalenceResult::ConditionalTrue { requires }
            } else {
                EquivalenceResult::True
            };
        }

        // Also check: b simplified vs a (before simplifying a further)
        let (simplified_b, steps_b) = self.simplify(b);

        // Check if simplified forms are semantically equal
        let checker = SemanticEqualityChecker::new(&self.context);
        if checker.are_equal(simplified_a, simplified_b) {
            // Merge steps for soundness analysis
            let mut all_steps = steps_a;
            all_steps.extend(steps_b);

            let mut has_conditional_rules = false;
            let mut requires: Vec<String> = Vec::new();

            for step in &all_steps {
                if step.soundness != SoundnessLabel::Equivalence {
                    has_conditional_rules = true;
                }
                for req in step.required_conditions() {
                    let condition_str = req.display(&self.context);
                    if !requires.contains(&condition_str) {
                        requires.push(condition_str);
                    }
                }
            }

            // Check blocked hints
            for hint in &self.last_blocked_hints {
                let expr_display = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &self.context,
                        id: hint.expr_id
                    }
                );
                let hint_str = match &hint.key {
                    crate::AssumptionKey::NonZero { .. } => {
                        format!("{} ≠ 0", expr_display)
                    }
                    crate::AssumptionKey::Positive { .. } => {
                        format!("{} > 0", expr_display)
                    }
                    crate::AssumptionKey::NonNegative { .. } => {
                        format!("{} ≥ 0", expr_display)
                    }
                    _ => format!("{} ({})", expr_display, hint.rule),
                };
                if !requires.contains(&hint_str) {
                    requires.push(hint_str);
                }
            }

            self.set_collect_steps(was_collecting);
            cas_solver_core::equivalence::normalize_requires(&mut requires);

            return if has_conditional_rules || !requires.is_empty() {
                EquivalenceResult::ConditionalTrue { requires }
            } else {
                EquivalenceResult::True
            };
        }

        // =================================================================
        // Fallback: Try A - B = 0
        // =================================================================
        let diff = self.context.add(Expr::Sub(a, b));
        let (simplified_diff, steps) = self.simplify(diff);

        self.set_collect_steps(was_collecting);

        let result_expr = simplified_diff;

        // Check if result is 0
        let is_zero = match self.context.get(result_expr) {
            Expr::Number(n) => n.is_zero(),
            _ => false,
        };

        if is_zero {
            // Success! Now determine if unconditional or conditional
            // Check for any soundness label worse than Equivalence
            let mut has_conditional_rules = false;
            let mut requires: Vec<String> = Vec::new();

            for step in &steps {
                // Check soundness label
                if step.soundness != SoundnessLabel::Equivalence {
                    has_conditional_rules = true;
                }

                // Collect required_conditions from steps
                for req in step.required_conditions() {
                    let condition_str = req.display(&self.context);
                    if !requires.contains(&condition_str) {
                        requires.push(condition_str);
                    }
                }
            }

            // Also check blocked hints (from Strict mode)
            for hint in &self.last_blocked_hints {
                // Build condition string from AssumptionKey
                let expr_display = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &self.context,
                        id: hint.expr_id
                    }
                );
                let hint_str = match &hint.key {
                    crate::AssumptionKey::NonZero { .. } => {
                        format!("{} ≠ 0", expr_display)
                    }
                    crate::AssumptionKey::Positive { .. } => {
                        format!("{} > 0", expr_display)
                    }
                    crate::AssumptionKey::NonNegative { .. } => {
                        format!("{} ≥ 0", expr_display)
                    }
                    _ => format!("{} ({})", expr_display, hint.rule),
                };
                if !requires.contains(&hint_str) {
                    requires.push(hint_str);
                }
            }

            cas_solver_core::equivalence::normalize_requires(&mut requires);

            if has_conditional_rules || !requires.is_empty() {
                EquivalenceResult::ConditionalTrue { requires }
            } else {
                EquivalenceResult::True
            }
        } else {
            // =================================================================
            // Fallback 2: Try expand(A - B) = 0
            // When normal simplify can't bridge trig identities gated behind
            // expand_mode (e.g., sin(a+b) → sin(a)cos(b) + cos(a)sin(b)),
            // the expand pipeline enables those rules and may prove equivalence.
            // =================================================================
            let (expanded_diff, expand_steps) = self.expand(diff);
            let is_expand_zero = match self.context.get(expanded_diff) {
                Expr::Number(n) => n.is_zero(),
                _ => false,
            };

            if is_expand_zero {
                let mut has_conditional_rules = false;
                let mut requires: Vec<String> = Vec::new();

                for step in &expand_steps {
                    if step.soundness != SoundnessLabel::Equivalence {
                        has_conditional_rules = true;
                    }
                    for req in step.required_conditions() {
                        let condition_str = req.display(&self.context);
                        if !requires.contains(&condition_str) {
                            requires.push(condition_str);
                        }
                    }
                }

                for hint in &self.last_blocked_hints {
                    let expr_display = format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &self.context,
                            id: hint.expr_id
                        }
                    );
                    let hint_str = match &hint.key {
                        crate::AssumptionKey::NonZero { .. } => {
                            format!("{} ≠ 0", expr_display)
                        }
                        crate::AssumptionKey::Positive { .. } => {
                            format!("{} > 0", expr_display)
                        }
                        crate::AssumptionKey::NonNegative { .. } => {
                            format!("{} ≥ 0", expr_display)
                        }
                        _ => format!("{} ({})", expr_display, hint.rule),
                    };
                    if !requires.contains(&hint_str) {
                        requires.push(hint_str);
                    }
                }

                cas_solver_core::equivalence::normalize_requires(&mut requires);

                return if has_conditional_rules || !requires.is_empty() {
                    EquivalenceResult::ConditionalTrue { requires }
                } else {
                    EquivalenceResult::True
                };
            }

            // Not zero symbolically - try numeric verification
            if self.allow_numerical_verification {
                let vars = cas_ast::collect_variables(&self.context, result_expr);
                let var_map = cas_solver_core::equivalence::default_equiv_probe_map(vars);

                if let Some(val) = eval_f64(&self.context, result_expr, &var_map) {
                    if cas_solver_core::equivalence::is_numeric_equiv_zero(val) {
                        // Numeric evidence suggests equivalence but couldn't prove symbolically
                        return EquivalenceResult::Unknown;
                    } else {
                        // Found counterexample
                        return EquivalenceResult::False;
                    }
                }

                // COMPLEX REFUTE-ONLY NET (B1): under the ComplexEnabled sticky
                // domain, an `i`-carrying difference that the f64 evaluator
                // cannot touch gets one principal-branch complex probe. A
                // clearly non-zero modulus REFUTES the identity; a near-zero
                // one stays Unknown — a probe may never CONFIRM equivalence
                // (see eval/actions.rs). Gated on the sticky domain so real
                // mode is byte-identical (in RealOnly, `i` stays opaque).
                if self.sticky_value_domain == crate::semantics::ValueDomain::ComplexEnabled {
                    let complex_map: std::collections::HashMap<String, cas_math::Complex64> =
                        var_map
                            .iter()
                            .map(|(k, v)| (k.clone(), cas_math::Complex64::real(*v)))
                            .collect();
                    if let Some(z) =
                        cas_math::eval_complex(&self.context, result_expr, &complex_map)
                    {
                        if !cas_solver_core::equivalence::is_numeric_equiv_zero(z.abs()) {
                            return EquivalenceResult::False;
                        }
                    }
                }
            }
            // Can't determine
            EquivalenceResult::Unknown
        }
    }
}

#[cfg(test)]
mod complex_refute_tests {
    use crate::{EquivalenceResult, Simplifier};

    fn extended(src_a: &str, src_b: &str, complex: bool) -> EquivalenceResult {
        let mut s = Simplifier::with_default_rules();
        if complex {
            s.set_sticky_value_domain(crate::semantics::ValueDomain::ComplexEnabled);
        }
        let a = cas_parser::parse(src_a, &mut s.context).expect("parse a");
        let b = cas_parser::parse(src_b, &mut s.context).expect("parse b");
        s.are_equivalent_extended(a, b)
    }

    #[test]
    fn complex_probe_refutes_false_identity() {
        // e^(iπ) = 1 is FALSE (true value -1): the complex probe refutes it.
        assert!(matches!(
            extended("e^(i*pi)", "1", true),
            EquivalenceResult::False
        ));
        // ln(-1) = -i·π is FALSE (principal branch gives +i·π).
        assert!(matches!(
            extended("ln(-1)", "-i*pi", true),
            EquivalenceResult::False
        ));
    }

    #[test]
    fn complex_probe_never_confirms_true_identity() {
        // B2 (Euler) landed: e^(iπ) = -1 now confirms via the EXACT fold —
        // the intended graduation of the original pin.
        assert!(matches!(
            extended("e^(i*pi)", "-1", true),
            EquivalenceResult::True | EquivalenceResult::ConditionalTrue { .. }
        ));
        // The never-confirm property lives on with a TRUE identity that has
        // no exact fold yet: i^i = e^(-π/2) waits for B3+B4 — the probe sees
        // a near-zero difference and must stay Unknown, never True. (ln(-1)
        // is unsuitable here: it still collapses to `undefined` pre-B3.)
        assert!(matches!(
            extended("i^i", "e^(-pi/2)", true),
            EquivalenceResult::Unknown
        ));
    }

    #[test]
    fn real_mode_keeps_i_expressions_unknown() {
        // Footprint guard: without the ComplexEnabled sticky domain the net is
        // inert — i-carrying differences stay Unknown exactly as before B1.
        assert!(matches!(
            extended("e^(i*pi)", "1", false),
            EquivalenceResult::Unknown
        ));
    }
}
