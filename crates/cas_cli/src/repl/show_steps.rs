use super::*;

impl Repl {
    pub(crate) fn show_simplification_steps(
        &mut self,
        expr: cas_ast::ExprId,
        steps: &[cas_engine::Step],
        style_signals: cas_ast::root_style::ParseStyleSignals,
    ) {
        use cas_ast::root_style::StylePreferences;
        use cas_ast::DisplayExprStyled;

        if self.verbosity == Verbosity::None {
            return;
        }

        // Create global style preferences from input signals + AST
        let style_prefs = StylePreferences::from_expression_with_signals(
            &self.engine.simplifier.context,
            expr,
            Some(&style_signals),
        );

        if steps.is_empty() {
            // Even with no engine steps, show didactic sub-steps if there are fraction sums
            let standalone_substeps = cas_engine::didactic::get_standalone_substeps(
                &self.engine.simplifier.context,
                expr,
            );

            if !standalone_substeps.is_empty() && self.verbosity != Verbosity::Succinct {
                println!("Computation:");
                // Helper function for LaTeX to plain text
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
                    // Handle nested fractions by processing innermost first
                    let mut iterations = 0;
                    while result.contains("\\frac{") && iterations < 10 {
                        iterations += 1;
                        if let Some(start) = result.rfind("\\frac{") {
                            let rest = &result[start + 5..]; // +5 = \frac, not including trailing {
                                                             // Find matching braces for numerator
                            if let Some((numer, numer_end)) = find_balanced_braces(rest) {
                                let after_numer = &rest[numer_end + 1..];
                                if after_numer.starts_with('{') {
                                    if let Some((denom, denom_end)) =
                                        find_balanced_braces(after_numer)
                                    {
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

                // Helper to find content within balanced braces
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

                for sub in &standalone_substeps {
                    println!("   → {}", sub.description);
                    if !sub.before_expr.is_empty() {
                        println!(
                            "     {} → {}",
                            latex_to_text(&sub.before_expr),
                            latex_to_text(&sub.after_expr)
                        );
                    }
                }
            } else if self.verbosity != Verbosity::Succinct {
                println!("No simplification steps needed.");
            }
        } else {
            if self.verbosity != Verbosity::Succinct {
                println!("Steps:");
            }

            // Enrich steps ONCE before iterating
            let enriched_steps = cas_engine::didactic::enrich_steps(
                &self.engine.simplifier.context,
                expr,
                steps.to_vec(),
            );

            let mut current_root = expr;
            let mut step_count = 0;
            let mut sub_steps_shown = false; // Track to show sub-steps only on first visible step
            for (step_idx, step) in steps.iter().enumerate() {
                if should_show_step(step, self.verbosity) {
                    // Early check for display no-op: skip step entirely if before/after display identical
                    let before_disp = clean_display_string(&format!(
                        "{}",
                        DisplayExprStyled::new(
                            &self.engine.simplifier.context,
                            step.before,
                            &style_prefs
                        )
                    ));
                    let after_disp = clean_display_string(&format!(
                        "{}",
                        DisplayExprStyled::new(
                            &self.engine.simplifier.context,
                            step.after,
                            &style_prefs
                        )
                    ));
                    // Display no-op check removed/simplified for brevity, logic copied from prev if needed
                    // But let's assume helper needs to be robust. I'll rely on copied logic.
                    if before_disp == after_disp {
                        if let Some(global_after) = step.global_after {
                            current_root = global_after;
                        }
                        continue;
                    }

                    step_count += 1;

                    if self.verbosity == Verbosity::Succinct {
                        // Low mode: just global state
                        current_root = reconstruct_global_expr(
                            &mut self.engine.simplifier.context,
                            current_root,
                            &step.path,
                            step.after,
                        );
                        println!(
                            "-> {}",
                            DisplayExprStyled::new(
                                &self.engine.simplifier.context,
                                current_root,
                                &style_prefs
                            )
                        );
                    } else {
                        // Normal/Verbose
                        println!("{}. {}  [{}]", step_count, step.description, step.rule_name);

                        if self.verbosity == Verbosity::Verbose
                            || self.verbosity == Verbosity::Normal
                        {
                            // Show Before: global expression before this step (always)
                            if let Some(global_before) = step.global_before {
                                println!(
                                    "   Before: {}",
                                    clean_display_string(&format!(
                                        "{}",
                                        DisplayExprStyled::new(
                                            &self.engine.simplifier.context,
                                            global_before,
                                            &style_prefs
                                        )
                                    ))
                                );
                            } else {
                                println!(
                                    "   Before: {}",
                                    clean_display_string(&format!(
                                        "{}",
                                        DisplayExprStyled::new(
                                            &self.engine.simplifier.context,
                                            current_root,
                                            &style_prefs
                                        )
                                    ))
                                );
                            }

                            // Didactic: Show sub-steps AFTER Before: line
                            // For fraction sums (global), show only once;
                            // For per-step enrichment (nested fractions, factorization), show each time
                            if let Some(enriched_step) = enriched_steps.get(step_idx) {
                                if !enriched_step.sub_steps.is_empty() {
                                    // Helper function for LaTeX to plain text
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
                                                let rest = &result[start + 5..]; // +5 = \frac
                                                if let Some((numer, numer_end)) =
                                                    find_brace_content(rest)
                                                {
                                                    let after_numer = &rest[numer_end + 1..];
                                                    if after_numer.starts_with('{') {
                                                        if let Some((denom, denom_end)) =
                                                            find_brace_content(after_numer)
                                                        {
                                                            let total_end = start
                                                                + 5
                                                                + numer_end
                                                                + 1
                                                                + denom_end
                                                                + 1;
                                                            let replacement =
                                                                format!("({}/{})", numer, denom);
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

                                        result = result.replace("\\", "");
                                        result
                                    }

                                    fn find_brace_content(s: &str) -> Option<(String, usize)> {
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

                                    // Categorize sub-steps
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

                                    // For fraction sums (global), show only once
                                    // For per-step enrichments, show each time
                                    let should_show = if has_fraction_sum
                                        && !has_nested_fraction
                                        && !has_factorization
                                        && !has_polynomial_identity
                                    {
                                        !sub_steps_shown
                                    } else {
                                        true // Always show per-step enrichments
                                    };

                                    if should_show {
                                        if has_polynomial_identity {
                                            println!("   [Normalización polinómica]");
                                        } else if has_fraction_sum && !has_nested_fraction {
                                            sub_steps_shown = true;
                                            println!("   [Suma de fracciones en exponentes]");
                                        } else if has_factorization {
                                            println!("   [Factorización de polinomios]");
                                        } else if has_nested_fraction {
                                            println!("   [Simplificación de fracción compleja]");
                                        }

                                        for sub in &enriched_step.sub_steps {
                                            println!("      → {}", sub.description);
                                            if !sub.before_expr.is_empty() {
                                                println!(
                                                    "        {} → {}",
                                                    latex_to_text(&sub.before_expr),
                                                    latex_to_text(&sub.after_expr)
                                                );
                                            }
                                        }
                                    }
                                }
                            }

                            // Show Rule: local transformation
                            let (rule_before_id, rule_after_id) =
                                match (step.before_local, step.after_local) {
                                    (Some(bl), Some(al)) => (bl, al),
                                    _ => (step.before, step.after),
                                };

                            let before_disp = clean_display_string(&format!(
                                "{}",
                                DisplayExprStyled::new(
                                    &self.engine.simplifier.context,
                                    rule_before_id,
                                    &style_prefs
                                )
                            ));
                            // Use scoped renderer for after expression if rule has transforms
                            let after_disp = clean_display_string(&render_with_rule_scope(
                                &self.engine.simplifier.context,
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

                            println!("   Rule: {} -> {}", before_disp, after_disp);

                            // V2.14.45: Show rule-provided substeps (educational explanations)
                            if !step.substeps.is_empty() {
                                for substep in &step.substeps {
                                    println!("   [{}]", substep.title);
                                    for line in &substep.lines {
                                        println!("      • {}", line);
                                    }
                                }
                            }
                        }

                        // Use precomputed global_after if available, fall back to reconstruction
                        if let Some(global_after) = step.global_after {
                            current_root = global_after;
                        } else {
                            current_root = reconstruct_global_expr(
                                &mut self.engine.simplifier.context,
                                current_root,
                                &step.path,
                                step.after,
                            );
                        }

                        // Show After
                        if self.verbosity == Verbosity::Normal
                            || self.verbosity == Verbosity::Verbose
                        {
                            println!(
                                "   After: {}",
                                clean_display_string(&format!(
                                    "{}",
                                    DisplayExprStyled::new(
                                        &self.engine.simplifier.context,
                                        current_root,
                                        &style_prefs
                                    )
                                ))
                            );

                            for event in &step.assumption_events {
                                if event.kind.should_display() {
                                    println!(
                                        "   {} {}: {}",
                                        event.kind.icon(),
                                        event.kind.label(),
                                        event.message
                                    );
                                }
                            }
                        }
                    }
                } else if let Some(global_after) = step.global_after {
                    current_root = global_after;
                } else {
                    current_root = reconstruct_global_expr(
                        &mut self.engine.simplifier.context,
                        current_root,
                        &step.path,
                        step.after,
                    );
                }
            }
        }
    }
}
