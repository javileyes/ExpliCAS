use crate::step::Step;
use cas_ast::{Context, DisplayExpr, ExprId};

/// Timeline HTML generator - exports simplification steps to interactive HTML
pub struct TimelineHtml<'a> {
    context: &'a Context,
    steps: &'a [Step],
    title: String,
    verbosity_level: VerbosityLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum VerbosityLevel {
    Low,     // Only global state changes
    Normal,  // Filtered meaningful steps
    Verbose, // All steps
}

impl<'a> TimelineHtml<'a> {
    pub fn new(
        context: &'a Context,
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
            title,
            verbosity_level: verbosity,
        }
    }

    /// Generate complete HTML document
    pub fn to_html(&self) -> String {
        let mut html = Self::html_header(&self.title);
        html.push_str(&self.render_timeline());
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
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            padding: 30px;
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
            padding: 8px 15px;
            margin: 10px 0;
            background: #f5f5f5;
            border-radius: 4px;
            font-size: 0.85em;
            color: #667eea;
            font-style: italic;
            border: 1px dashed #667eea;
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
            <strong>Original Expression:</strong><br>
            \({}\)
        </div>
"#,
            escaped_title,
            latex_escape(title)
        )
    }

    fn should_show_step(&self, step: &Step) -> bool {
        match self.verbosity_level {
            VerbosityLevel::Verbose => true,
            VerbosityLevel::Normal => {
                // Filter out noisy canonicalization steps
                !step.rule_name.starts_with("Canonicalize")
                    && step.rule_name != "Add Zero"
                    && step.rule_name != "Multiply by One"
                    && step.rule_name != "Identity Power"
            }
            VerbosityLevel::Low => {
                // Only show major transformations
                !step.rule_name.starts_with("Canonicalize")
                    && !step.rule_name.starts_with("Pull")
                    && step.rule_name != "Add Zero"
                    && step.rule_name != "Multiply by One"
                    && step.rule_name != "Identity Power"
                    && step.rule_name != "Sort"
                    && step.rule_name != "Flatten"
            }
        }
    }

    fn render_timeline(&self) -> String {
        let mut html = String::from("        <div class=\"timeline\">\n");

        let mut step_number = 0;
        for step in self.steps.iter() {
            // Filter based on verbosity
            if !self.should_show_step(step) {
                continue;
            }

            step_number += 1;
            let before = format!(
                "{}",
                DisplayExpr {
                    context: self.context,
                    id: step.before
                }
            );
            let after = format!(
                "{}",
                DisplayExpr {
                    context: self.context,
                    id: step.after
                }
            );

            html.push_str(&format!(
                r#"            <div class="step">
                <div class="step-number">{}</div>
                <div class="step-content">
                    <h3>{}</h3>
                    <div class="math-expr before">
                        <strong>Before:</strong><br>
                        \({}\)
                    </div>
                    <div class="rule-description">
                        {} 
                    </div>
                    <div class="math-expr after">
                        <strong>After:</strong><br>
                        \({}\)
                    </div>
                </div>
            </div>
"#,
                step_number,
                html_escape(&step.rule_name),
                latex_escape(&before),
                html_escape(&step.description),
                latex_escape(&after)
            ));
        }

        // Add final result
        if let Some(last_step) = self.steps.last() {
            let final_expr = format!(
                "{}",
                DisplayExpr {
                    context: self.context,
                    id: last_step.after
                }
            );
            html.push_str(&format!(
                r#"        </div>
        <div class="final-result">
            <strong>Final Result:</strong><br>
            \({}\)
        </div>
"#,
                latex_escape(&final_expr)
            ));
        } else {
            html.push_str("        </div>\n");
        }

        html
    }

    fn html_footer() -> &'static str {
        r#"    </div>
    <footer>
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
        let ctx = Context::new();
        let steps = vec![];
        let expr = ctx.num(1);

        let timeline = TimelineHtml::new(&ctx, &steps, expr, VerbosityLevel::Normal);
        let html = timeline.to_html();

        assert!(html.contains("<!DOCTYPE html"));
        assert!(html.contains("MathJax"));
        assert!(html.contains("timeline"));
        assert!(html.contains("CAS Simplification"));
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("x & y"), "x &amp; y");
    }
}
