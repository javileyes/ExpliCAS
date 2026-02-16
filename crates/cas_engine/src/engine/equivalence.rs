//! Equivalence checking between mathematical expressions.
//!
//! Provides both simple boolean equivalence (`are_equivalent`) and
//! extended tri-state equivalence with domain conditions (`are_equivalent_extended`).

use super::evaluator::eval_f64;
use super::{EquivalenceResult, Simplifier};
use cas_ast::{Expr, ExprId};
use num_traits::Zero;
use std::collections::HashMap;

impl Simplifier {
    pub fn are_equivalent(&mut self, a: ExprId, b: ExprId) -> bool {
        let diff = self.context.add(Expr::Sub(a, b));
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
                if !self.allow_numerical_verification {
                    return false;
                }
                let vars = cas_ast::collect_variables(&self.context, result_expr);
                let mut var_map = HashMap::new();
                for var in vars {
                    var_map.insert(var, 1.23456789);
                }

                if let Some(val) = eval_f64(&self.context, result_expr, &var_map) {
                    val.abs() < 1e-9
                } else {
                    false
                }
            }
        }
    }

    /// Normalize, deduplicate, and sort requires strings for stable output.
    ///
    /// Normalization rules:
    /// - `expr ≠ 0` where expr starts with `-` → canonicalize to positive form
    /// - Deduplicate by exact string match
    /// - Sort alphabetically for deterministic output
    fn normalize_requires(&self, requires: &mut Vec<String>) {
        use std::collections::HashSet;

        // Normalize each require string
        for req in requires.iter_mut() {
            // Handle "expr ≠ 0" → strip leading negative if present
            if let Some(expr_part) = req.strip_suffix(" ≠ 0") {
                let trimmed = expr_part.trim();
                // If expression starts with "-(" and ends with ")", remove the negative
                if let Some(inner) = trimmed.strip_prefix("-(") {
                    if let Some(inner) = inner.strip_suffix(")") {
                        *req = format!("{} ≠ 0", inner.trim());
                    }
                } else if let Some(inner) = trimmed.strip_prefix("-") {
                    // Simple negative like "-x" → "x"
                    if !inner.starts_with('(') && !inner.contains(' ') {
                        *req = format!("{} ≠ 0", inner.trim());
                    }
                }
            }
        }

        // Deduplicate
        let mut seen = HashSet::new();
        requires.retain(|r| seen.insert(r.clone()));

        // Sort for deterministic output
        requires.sort();
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
        use crate::semantic_equality::SemanticEqualityChecker;

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
            self.normalize_requires(&mut requires);

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
                    cas_ast::DisplayExpr {
                        context: &self.context,
                        id: hint.expr_id
                    }
                );
                let hint_str = match &hint.key {
                    crate::assumptions::AssumptionKey::NonZero { .. } => {
                        format!("{} ≠ 0", expr_display)
                    }
                    crate::assumptions::AssumptionKey::Positive { .. } => {
                        format!("{} > 0", expr_display)
                    }
                    crate::assumptions::AssumptionKey::NonNegative { .. } => {
                        format!("{} ≥ 0", expr_display)
                    }
                    _ => format!("{} ({})", expr_display, hint.rule),
                };
                if !requires.contains(&hint_str) {
                    requires.push(hint_str);
                }
            }

            self.set_collect_steps(was_collecting);
            self.normalize_requires(&mut requires);

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
                    cas_ast::DisplayExpr {
                        context: &self.context,
                        id: hint.expr_id
                    }
                );
                let hint_str = match &hint.key {
                    crate::assumptions::AssumptionKey::NonZero { .. } => {
                        format!("{} ≠ 0", expr_display)
                    }
                    crate::assumptions::AssumptionKey::Positive { .. } => {
                        format!("{} > 0", expr_display)
                    }
                    crate::assumptions::AssumptionKey::NonNegative { .. } => {
                        format!("{} ≥ 0", expr_display)
                    }
                    _ => format!("{} ({})", expr_display, hint.rule),
                };
                if !requires.contains(&hint_str) {
                    requires.push(hint_str);
                }
            }

            self.normalize_requires(&mut requires);

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
                        cas_ast::DisplayExpr {
                            context: &self.context,
                            id: hint.expr_id
                        }
                    );
                    let hint_str = match &hint.key {
                        crate::assumptions::AssumptionKey::NonZero { .. } => {
                            format!("{} ≠ 0", expr_display)
                        }
                        crate::assumptions::AssumptionKey::Positive { .. } => {
                            format!("{} > 0", expr_display)
                        }
                        crate::assumptions::AssumptionKey::NonNegative { .. } => {
                            format!("{} ≥ 0", expr_display)
                        }
                        _ => format!("{} ({})", expr_display, hint.rule),
                    };
                    if !requires.contains(&hint_str) {
                        requires.push(hint_str);
                    }
                }

                self.normalize_requires(&mut requires);

                return if has_conditional_rules || !requires.is_empty() {
                    EquivalenceResult::ConditionalTrue { requires }
                } else {
                    EquivalenceResult::True
                };
            }

            // Not zero symbolically - try numeric verification
            if self.allow_numerical_verification {
                let vars = cas_ast::collect_variables(&self.context, result_expr);
                let mut var_map = HashMap::new();
                for var in &vars {
                    var_map.insert(var.clone(), 1.23456789);
                }

                if let Some(val) = eval_f64(&self.context, result_expr, &var_map) {
                    if val.abs() < 1e-9 {
                        // Numeric evidence suggests equivalence but couldn't prove symbolically
                        return EquivalenceResult::Unknown;
                    } else {
                        // Found counterexample
                        return EquivalenceResult::False;
                    }
                }
            }
            // Can't determine
            EquivalenceResult::Unknown
        }
    }
}
