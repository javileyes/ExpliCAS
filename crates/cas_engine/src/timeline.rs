use crate::step::{PathStep, Step};
use cas_ast::{Context, DisplayExpr, Expr, ExprId, LaTeXExpr};

/// Timeline HTML generator - exports simplification steps to interactive HTML
pub struct TimelineHtml<'a> {
    context: &'a mut Context,
    steps: &'a [Step],
    original_expr: ExprId,
    title: String,
    verbosity_level: VerbosityLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum VerbosityLevel {
    Low,     // Only high-importance steps (Factor, Expand, Integrate, etc.)
    Normal,  // Medium+ importance steps (most transformations)
    Verbose, // All steps including trivial ones
}

impl VerbosityLevel {
    /// Returns minimum importance level to show for this verbosity
    fn min_importance(&self) -> crate::step::ImportanceLevel {
        use crate::step::ImportanceLevel;
        match self {
            VerbosityLevel::Low => ImportanceLevel::High,
            VerbosityLevel::Normal => ImportanceLevel::Medium,
            VerbosityLevel::Verbose => ImportanceLevel::Trivial, // Show all
        }
    }

    /// Check if a step should be shown at this verbosity level
    fn should_show_step(&self, step: &Step) -> bool {
        step.importance() >= self.min_importance()
    }
}

impl<'a> TimelineHtml<'a> {
    pub fn new(
        context: &'a mut Context,
        steps: &'a [Step],
        original_expr: ExprId,
        verbosity: VerbosityLevel,
    ) -> Self {
        let title = format!(
            "{}",
            DisplayExpr {
                context,
                id: original_expr
            }
        );
        Self {
            context,
            steps,
            original_expr,
            title,
            verbosity_level: verbosity,
        }
    }

    /// Generate complete HTML document
    pub fn to_html(&mut self) -> String {
        // Filter steps based on verbosity level
        let filtered_steps: Vec<&Step> = self
            .steps
            .iter()
            .filter(|step| self.verbosity_level.should_show_step(step))
            .collect();

        let mut html = Self::html_header(&self.title);
        html.push_str(&self.render_timeline_filtered(&filtered_steps));
        html.push_str(Self::html_footer());
        html
    }

    fn html_header(title: &str) -> String {
        let escaped_title = html_escape(title);
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAS Steps: {}</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            padding: 30px 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .original {{
            text-align: center;
            font-size: 1.2em;
            padding: 15px;
            background: #f0f4ff;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 2px solid #667eea;
        }}
        .timeline {{
            position: relative;
            padding: 20px 0;
        }}
        .timeline::before {{
            content: '';
            position: absolute;
            left: 30px;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(to bottom, #667eea, #764ba2);
        }}
        .step {{
            position: relative;
            margin-bottom: 30px;
            padding-left: 80px;
            animation: fadeIn 0.5s ease-in;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .step-number {{
            position: absolute;
            left: 0;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 20px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        .step-content {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .step-content:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }}
        .step-content h3 {{
            margin-top: 0;
            color: #667eea;
            font-size: 1.1em;
        }}
        .math-expr {{
            padding: 12px 15px;
            background: #fafafa;
            border-left: 4px solid #667eea;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 1.05em;
        }}
        .math-expr.before {{
            border-left-color: #ff9800;
            background: #fff8f0;
        }}
        .math-expr.after {{
            border-left-color: #4caf50;
            background: #f0fff4;
        }}
        .math-expr strong {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .rule-description {{
            text-align: center;
            padding: 12px 20px;
            margin: 15px 0;
            background: #f9f5ff;
            border-radius: 6px;
            font-size: 0.95em;
            color: #667eea;
            border: 2px dashed #667eea;
        }}
        .local-change {{
            font-size: 1.1em;
            margin: 8px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
            text-align: center;
        }}
        .rule-name {{
            font-size: 0.85em;
            color: #764ba2;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .final-result {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
            color: white;
            border-radius: 10px;
            margin-top: 30px;
            font-size: 1.2em;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        }}
        footer {{
            text-align: center;
            margin-top: 30px;
            color: white;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CAS Simplification Steps</h1>
        <p class="subtitle">Step-by-step visualization</p>
        <div class="original">
            \(\textbf{{Original Expression:}}\)
            \[{}\]
        </div>
"#,
            escaped_title,
            latex_escape(title)
        )
    }

    fn reconstruct_global_expr(
        &mut self,
        root: ExprId,
        path: &[PathStep],
        replacement: ExprId,
    ) -> ExprId {
        if path.is_empty() {
            return replacement;
        }

        let current_step = &path[0];
        let remaining_path = &path[1..];
        let expr = self.context.get(root).clone();

        match (expr, current_step) {
            (Expr::Add(l, r), PathStep::Left) => {
                let new_l = self.reconstruct_global_expr(l, remaining_path, replacement);
                self.context.add(Expr::Add(new_l, r))
            }
            (Expr::Add(l, r), PathStep::Right) => {
                let new_r = self.reconstruct_global_expr(r, remaining_path, replacement);
                self.context.add(Expr::Add(l, new_r))
            }
            (Expr::Sub(l, r), PathStep::Left) => {
                let new_l = self.reconstruct_global_expr(l, remaining_path, replacement);
                self.context.add(Expr::Sub(new_l, r))
            }
            (Expr::Sub(l, r), PathStep::Right) => {
                let new_r = self.reconstruct_global_expr(r, remaining_path, replacement);
                self.context.add(Expr::Sub(l, new_r))
            }
            (Expr::Mul(l, r), PathStep::Left) => {
                let new_l = self.reconstruct_global_expr(l, remaining_path, replacement);
                self.context.add(Expr::Mul(new_l, r))
            }
            (Expr::Mul(l, r), PathStep::Right) => {
                let new_r = self.reconstruct_global_expr(r, remaining_path, replacement);
                self.context.add(Expr::Mul(l, new_r))
            }
            (Expr::Div(l, r), PathStep::Left) => {
                let new_l = self.reconstruct_global_expr(l, remaining_path, replacement);
                self.context.add(Expr::Div(new_l, r))
            }
            (Expr::Div(l, r), PathStep::Right) => {
                let new_r = self.reconstruct_global_expr(r, remaining_path, replacement);
                self.context.add(Expr::Div(l, new_r))
            }
            (Expr::Pow(b, e), PathStep::Base) => {
                let new_b = self.reconstruct_global_expr(b, remaining_path, replacement);
                self.context.add(Expr::Pow(new_b, e))
            }
            (Expr::Pow(b, e), PathStep::Exponent) => {
                let new_e = self.reconstruct_global_expr(e, remaining_path, replacement);
                self.context.add(Expr::Pow(b, new_e))
            }
            (Expr::Neg(e), PathStep::Inner) => {
                let new_e = self.reconstruct_global_expr(e, remaining_path, replacement);
                self.context.add(Expr::Neg(new_e))
            }
            (Expr::Function(name, args), PathStep::Arg(idx)) => {
                let mut new_args = args.clone();
                if *idx < new_args.len() {
                    new_args[*idx] =
                        self.reconstruct_global_expr(new_args[*idx], remaining_path, replacement);
                    self.context.add(Expr::Function(name, new_args))
                } else {
                    root
                }
            }
            _ => root,
        }
    }

    fn render_timeline_filtered(&mut self, filtered_steps: &[&Step]) -> String {
        let mut html = String::from("        <div class=\"timeline\">\n");

        let mut step_number = 0;
        let mut current_global = self.original_expr;

        // Track which steps to display (create a set of their indices)
        let filtered_indices: std::collections::HashSet<_> = filtered_steps
            .iter()
            .map(|s| *s as *const Step) // Dereference `&Step` to `Step` before taking pointer
            .collect();

        // Iterate over ALL steps to correctly update the global state
        for step in self.steps.iter() {
            // Store global state BEFORE this step is applied
            let global_state_before_this_step = current_global;

            // Always update global state, regardless of whether it's filtered
            current_global = self.reconstruct_global_expr(current_global, &step.path, step.after);

            // Only render if this step is in filtered set
            let step_ptr = step as *const Step;
            if !filtered_indices.contains(&step_ptr) {
                continue; // Skip rendering for non-filtered steps
            }
            step_number += 1;

            // Global state BEFORE this step (this is the state *before* the current `step` was applied)
            let global_before = LaTeXExpr {
                context: self.context,
                id: global_state_before_this_step,
            }
            .to_latex();

            // Global AFTER is the current_global (already updated)
            let _global_after = LaTeXExpr {
                context: self.context,
                id: current_global,
            }
            .to_latex();

            // Local change (before -> after)
            // For certain rules, preserve exponent notation instead of converting to roots
            let should_preserve_exponents = step.rule_name.contains("Multiply exponents")
                || step.rule_name.contains("Power of a Power");

            let local_change_latex = if should_preserve_exponents {
                let local_before = cas_ast::LatexNoRoots {
                    context: self.context,
                    id: step.before,
                }
                .to_latex();
                let local_after = cas_ast::LatexNoRoots {
                    context: self.context,
                    id: step.after,
                }
                .to_latex();
                format!("{} \\rightarrow {}", local_before, local_after)
            } else {
                let local_before = LaTeXExpr {
                    context: self.context,
                    id: step.before,
                }
                .to_latex();
                let local_after = LaTeXExpr {
                    context: self.context,
                    id: step.after,
                }
                .to_latex();
                format!("{} \\rightarrow {}", local_before, local_after)
            };

            html.push_str(&format!(
                r#"            <div class="step">
                <div class="step-number">{}</div>
                <div class="step-content">
                    <h3>{}</h3>
                    <div class="math-expr before">
                        \(\textbf{{Expression:}}\)
                        \[{}\]
                    </div>
                    <div class="rule-description">
                        <div class="rule-name">\(\text{{{}}}\)</div>
                        <div class="local-change">
                            \[{}\]
                        </div>
                    </div>
                </div>
            </div>
"#,
                step_number,
                html_escape(&step.rule_name),
                global_before,
                step.description,
                local_change_latex
            ));
        }

        // Add final result
        let final_expr = LaTeXExpr {
            context: self.context,
            id: current_global,
        }
        .to_latex();
        html.push_str(
            r#"        </div>
        <div class="final-result">
            \(\textbf{Final Result:}\)
            \["#,
        );
        html.push_str(&final_expr);
        html.push_str(
            r#"\]
        </div>
    </div>
"#,
        );
        html
    }

    fn html_footer() -> &'static str {
        r#"    <footer>
        Generated by Rust CAS Engine
    </footer>
</body>
</html>"#
    }
}

/// Escape HTML special characters
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Prepare string for LaTeX rendering in MathJax
fn latex_escape(s: &str) -> String {
    // For MathJax, we mostly just need the string as-is
    // But escape backslashes that aren't part of LaTeX commands
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_generation() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let three = ctx.num(3);
        let add_expr = ctx.add(Expr::Add(two, three));
        let five = ctx.num(5);

        // Create a step for the simplification
        let steps = vec![Step::new(
            "2 + 3 = 5",
            "Combine Constants",
            add_expr,
            five,
            vec![],
            Some(&ctx),
        )];

        let mut timeline = TimelineHtml::new(&mut ctx, &steps, add_expr, VerbosityLevel::Verbose);
        let html = timeline.to_html();

        assert!(html.contains("<!DOCTYPE html"));
        assert!(html.contains("timeline"));
        assert!(html.contains("CAS Simplification"));
        // The HTML should contain our step (Combine Constants has ImportanceLevel::Low, so needs Verbose)
        assert!(html.contains("Combine Constants"));
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("x & y"), "x &amp; y");
    }
}
