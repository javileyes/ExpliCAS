//! Step visualization - pure output generation via ReplReply
//!
//! This module generates step-by-step output as strings, no direct I/O.

use super::output::{reply_output, ReplReply};
use super::*;

/// Buffer for accumulating step output lines
struct StepBuf {
    s: String,
}

impl StepBuf {
    fn new() -> Self {
        Self { s: String::new() }
    }

    fn line(&mut self, text: impl AsRef<str>) {
        self.s.push_str(text.as_ref());
        self.s.push('\n');
    }

    fn into_reply(self) -> ReplReply {
        if self.s.is_empty() {
            vec![]
        } else {
            reply_output(self.s.trim_end().to_string())
        }
    }
}

impl Repl {
    /// Show simplification steps - returns ReplReply (no I/O)
    pub(crate) fn show_simplification_steps_core(
        &mut self,
        expr: cas_ast::ExprId,
        steps: &[cas_didactic::Step],
        style_signals: cas_ast::root_style::ParseStyleSignals,
    ) -> ReplReply {
        use cas_ast::root_style::StylePreferences;
        use cas_formatter::DisplayExprStyled;

        if self.verbosity == Verbosity::None {
            return vec![];
        }

        let mut buf = StepBuf::new();

        // Create global style preferences from input signals + AST
        let style_prefs = StylePreferences::from_expression_with_signals(
            &self.core.engine.simplifier.context,
            expr,
            Some(&style_signals),
        );

        if steps.is_empty() {
            // Even with no engine steps, show didactic sub-steps if there are fraction sums
            let standalone_substeps =
                cas_didactic::get_standalone_substeps(&self.core.engine.simplifier.context, expr);

            if !standalone_substeps.is_empty() && self.verbosity != Verbosity::Succinct {
                buf.line("Computation:");

                for sub in &standalone_substeps {
                    buf.line(format!("   → {}", sub.description));
                    if !sub.before_expr.is_empty() {
                        buf.line(format!(
                            "     {} → {}",
                            latex_to_text(&sub.before_expr),
                            latex_to_text(&sub.after_expr)
                        ));
                    }
                }
            } else if self.verbosity != Verbosity::Succinct {
                buf.line("No simplification steps needed.");
            }
        } else {
            if self.verbosity != Verbosity::Succinct {
                buf.line("Steps:");
            }

            // Enrich steps ONCE before iterating
            let enriched_steps = cas_didactic::enrich_steps(
                &self.core.engine.simplifier.context,
                expr,
                steps.to_vec(),
            );

            let mut current_root = expr;
            let mut step_count = 0;
            let mut sub_steps_shown = false;

            for (step_idx, step) in steps.iter().enumerate() {
                if should_show_step(step, self.verbosity) {
                    let before_disp = clean_display_string(&format!(
                        "{}",
                        DisplayExprStyled::new(
                            &self.core.engine.simplifier.context,
                            step.before,
                            &style_prefs
                        )
                    ));
                    let after_disp = clean_display_string(&format!(
                        "{}",
                        DisplayExprStyled::new(
                            &self.core.engine.simplifier.context,
                            step.after,
                            &style_prefs
                        )
                    ));

                    if before_disp == after_disp {
                        if let Some(global_after) = step.global_after {
                            current_root = global_after;
                        }
                        continue;
                    }

                    step_count += 1;

                    if self.verbosity == Verbosity::Succinct {
                        current_root = reconstruct_global_expr(
                            &mut self.core.engine.simplifier.context,
                            current_root,
                            step.path(),
                            step.after,
                        );
                        buf.line(format!(
                            "-> {}",
                            DisplayExprStyled::new(
                                &self.core.engine.simplifier.context,
                                current_root,
                                &style_prefs
                            )
                        ));
                    } else {
                        // Normal/Verbose
                        buf.line(format!(
                            "{}. {}  [{}]",
                            step_count, step.description, step.rule_name
                        ));

                        if self.verbosity == Verbosity::Verbose
                            || self.verbosity == Verbosity::Normal
                        {
                            // Show Before: global expression
                            if let Some(global_before) = step.global_before {
                                buf.line(format!(
                                    "   Before: {}",
                                    clean_display_string(&format!(
                                        "{}",
                                        DisplayExprStyled::new(
                                            &self.core.engine.simplifier.context,
                                            global_before,
                                            &style_prefs
                                        )
                                    ))
                                ));
                            } else {
                                buf.line(format!(
                                    "   Before: {}",
                                    clean_display_string(&format!(
                                        "{}",
                                        DisplayExprStyled::new(
                                            &self.core.engine.simplifier.context,
                                            current_root,
                                            &style_prefs
                                        )
                                    ))
                                ));
                            }

                            // Didactic sub-steps
                            if let Some(enriched_step) = enriched_steps.get(step_idx) {
                                if !enriched_step.sub_steps.is_empty() {
                                    let has_fraction_sum =
                                        enriched_step.sub_steps.iter().any(|s| {
                                            s.description.contains("common denominator")
                                                || s.description.contains("Sum the fractions")
                                        });
                                    let has_factorization =
                                        enriched_step.sub_steps.iter().any(|s| {
                                            s.description.contains("Cancel common factor")
                                                || s.description.contains("Factor")
                                        });
                                    let has_nested_fraction =
                                        enriched_step.sub_steps.iter().any(|s| {
                                            s.description.contains("Combinar términos")
                                                || s.description.contains("Invertir la fracción")
                                                || s.description.contains("denominadores internos")
                                        });
                                    let has_polynomial_identity =
                                        enriched_step.sub_steps.iter().any(|s| {
                                            s.description.contains("forma normal polinómica")
                                                || s.description
                                                    .contains("Cancelar términos semejantes")
                                        });

                                    let should_show = if has_fraction_sum
                                        && !has_nested_fraction
                                        && !has_factorization
                                        && !has_polynomial_identity
                                    {
                                        !sub_steps_shown
                                    } else {
                                        true
                                    };

                                    if should_show {
                                        if has_polynomial_identity {
                                            buf.line("   [Normalización polinómica]");
                                        } else if has_fraction_sum && !has_nested_fraction {
                                            sub_steps_shown = true;
                                            buf.line("   [Suma de fracciones en exponentes]");
                                        } else if has_factorization {
                                            buf.line("   [Factorización de polinomios]");
                                        } else if has_nested_fraction {
                                            buf.line("   [Simplificación de fracción compleja]");
                                        }

                                        for sub in &enriched_step.sub_steps {
                                            buf.line(format!("      → {}", sub.description));
                                            if !sub.before_expr.is_empty() {
                                                buf.line(format!(
                                                    "        {} → {}",
                                                    latex_to_text(&sub.before_expr),
                                                    latex_to_text(&sub.after_expr)
                                                ));
                                            }
                                        }
                                    }
                                }
                            }

                            // Show Rule: local transformation
                            let (rule_before_id, rule_after_id) =
                                match (step.before_local(), step.after_local()) {
                                    (Some(bl), Some(al)) => (bl, al),
                                    _ => (step.before, step.after),
                                };

                            let before_disp = clean_display_string(&format!(
                                "{}",
                                DisplayExprStyled::new(
                                    &self.core.engine.simplifier.context,
                                    rule_before_id,
                                    &style_prefs
                                )
                            ));
                            let after_disp = clean_display_string(&render_with_rule_scope(
                                &self.core.engine.simplifier.context,
                                rule_after_id,
                                &step.rule_name,
                                &style_prefs,
                            ));

                            if before_disp == after_disp {
                                if let Some(global_after) = step.global_after {
                                    current_root = global_after;
                                }
                                continue;
                            }

                            buf.line(format!("   Rule: {} -> {}", before_disp, after_disp));

                            // Rule-provided substeps
                            if !step.substeps().is_empty() {
                                for substep in step.substeps() {
                                    buf.line(format!("   [{}]", substep.title));
                                    for line in &substep.lines {
                                        buf.line(format!("      • {}", line));
                                    }
                                }
                            }
                        }

                        // Update current_root
                        if let Some(global_after) = step.global_after {
                            current_root = global_after;
                        } else {
                            current_root = reconstruct_global_expr(
                                &mut self.core.engine.simplifier.context,
                                current_root,
                                step.path(),
                                step.after,
                            );
                        }

                        // Show After
                        if self.verbosity == Verbosity::Normal
                            || self.verbosity == Verbosity::Verbose
                        {
                            buf.line(format!(
                                "   After: {}",
                                clean_display_string(&format!(
                                    "{}",
                                    DisplayExprStyled::new(
                                        &self.core.engine.simplifier.context,
                                        current_root,
                                        &style_prefs
                                    )
                                ))
                            ));

                            for event in step.assumption_events() {
                                if event.kind.should_display() {
                                    buf.line(format!(
                                        "   {} {}: {}",
                                        event.kind.icon(),
                                        event.kind.label(),
                                        event.message
                                    ));
                                }
                            }
                        }
                    }
                } else if let Some(global_after) = step.global_after {
                    current_root = global_after;
                } else {
                    current_root = reconstruct_global_expr(
                        &mut self.core.engine.simplifier.context,
                        current_root,
                        step.path(),
                        step.after,
                    );
                }
            }
        }

        buf.into_reply()
    }

    /// Wrapper that prints steps (for backwards compatibility)
    pub(crate) fn show_simplification_steps(
        &mut self,
        expr: cas_ast::ExprId,
        steps: &[cas_didactic::Step],
        style_signals: cas_ast::root_style::ParseStyleSignals,
    ) {
        let reply = self.show_simplification_steps_core(expr, steps, style_signals);
        self.print_reply(reply);
    }
}

// =============================================================================
// Helper functions (moved outside impl to avoid duplication)
// =============================================================================

/// Convert LaTeX notation to plain text for display
fn latex_to_text(s: &str) -> String {
    let mut result = s.to_string();

    // Replace \cdot with · (multiplication dot)
    result = result.replace("\\cdot", " · ");

    // Replace \text{...} with content without wrapper
    while let Some(start) = result.find("\\text{") {
        if let Some(end) = result[start + 6..].find('}') {
            let content = &result[start + 6..start + 6 + end];
            result = format!(
                "{}{}{}",
                &result[..start],
                content,
                &result[start + 7 + end..]
            );
        } else {
            break;
        }
    }

    // Recursively replace \frac{num}{den} with (num/den)
    let mut iterations = 0;
    while result.contains("\\frac{") && iterations < 10 {
        iterations += 1;
        if let Some(start) = result.rfind("\\frac{") {
            let rest = &result[start + 5..];
            if let Some((numer, numer_end)) = find_balanced_braces(rest) {
                let after_numer = &rest[numer_end + 1..];
                if after_numer.starts_with('{') {
                    if let Some((denom, denom_end)) = find_balanced_braces(after_numer) {
                        let total_end = start + 5 + numer_end + 1 + denom_end + 1;
                        let replacement = format!("({}/{})", numer, denom);
                        result = format!(
                            "{}{}{}",
                            &result[..start],
                            replacement,
                            &result[total_end..]
                        );
                        continue;
                    }
                }
            }
        }
        break;
    }

    // Clean remaining backslashes
    result = result.replace("\\", "");
    result
}

/// Find content within balanced braces
fn find_balanced_braces(s: &str) -> Option<(String, usize)> {
    let mut depth = 0;
    let mut content = String::new();
    for (i, c) in s.chars().enumerate() {
        match c {
            '{' => {
                if depth > 0 {
                    content.push(c);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((content, i));
                }
                content.push(c);
            }
            _ => {
                if depth > 0 {
                    content.push(c);
                }
            }
        }
    }
    None
}
