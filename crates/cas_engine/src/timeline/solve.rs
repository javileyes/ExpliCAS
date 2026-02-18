use super::escape::html_escape;
use super::latex_clean::clean_latex_identities;
use crate::solver::SolveStep;
use cas_ast::{Context, Equation, SolutionSet};
use cas_formatter::{DisplayExpr, LaTeXExpr};

/// Timeline HTML generator for equation solving steps
pub struct SolveTimelineHtml<'a> {
    context: &'a mut Context,
    steps: &'a [SolveStep],
    original_eq: &'a Equation,
    solution_set: &'a SolutionSet,
    var: String,
    title: String,
}

impl<'a> SolveTimelineHtml<'a> {
    pub fn new(
        context: &'a mut Context,
        steps: &'a [SolveStep],
        original_eq: &'a Equation,
        solution_set: &'a SolutionSet,
        var: &str,
    ) -> Self {
        let title = format!(
            "{} {} {}",
            DisplayExpr {
                context,
                id: original_eq.lhs
            },
            original_eq.op,
            DisplayExpr {
                context,
                id: original_eq.rhs
            }
        );
        Self {
            context,
            steps,
            original_eq,
            solution_set,
            var: var.to_string(),
            title,
        }
    }

    /// Generate complete HTML document for solve steps
    pub fn to_html(&mut self) -> String {
        let mut html = self.html_header_solve();
        html.push_str(&self.render_solve_timeline());
        html.push_str(Self::html_footer_solve());

        // Clean up identity patterns like "\cdot 1" for better display
        clean_latex_identities(&html)
    }

    fn html_header_solve(&self) -> String {
        let escaped_title = html_escape(&self.title);
        let original_latex = format!(
            "{} {} {}",
            LaTeXExpr {
                context: self.context,
                id: self.original_eq.lhs
            }
            .to_latex(),
            self.relop_to_latex(&self.original_eq.op),
            LaTeXExpr {
                context: self.context,
                id: self.original_eq.rhs
            }
            .to_latex()
        );

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Solve Steps: {}</title>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
    * {{
        box-sizing: border-box;
    }}
    /* Theme variables */
    :root {{
        --bg-gradient-start: #1a1a2e;
        --bg-gradient-end: #16213e;
        --container-bg: rgba(30, 40, 60, 0.95);
        --container-shadow: rgba(0, 0, 0, 0.3);
        --title-color: #64b5f6;
        --subtitle-color: #90caf9;
        --original-bg-start: #1565c0;
        --original-bg-end: #0d47a1;
        --original-shadow: rgba(21, 101, 192, 0.4);
        --timeline-line-start: #64b5f6;
        --timeline-line-end: #4caf50;
        --step-bg: rgba(40, 50, 70, 0.8);
        --step-border: #64b5f6;
        --step-hover-shadow: rgba(100, 181, 246, 0.3);
        --step-dot-bg: #64b5f6;
        --step-dot-border: #1a1a2e;
        --step-number-color: #64b5f6;
        --description-color: #b0bec5;
        --equation-bg: rgba(30, 40, 55, 0.9);
        --final-bg-start: #2e7d32;
        --final-bg-end: #1b5e20;
        --final-shadow: rgba(76, 175, 80, 0.3);
        --footer-color: white;
        --text-color: #e0e0e0;
    }}
    :root.light {{
        --bg-gradient-start: #667eea;
        --bg-gradient-end: #764ba2;
        --container-bg: white;
        --container-shadow: rgba(0, 0, 0, 0.2);
        --title-color: #333;
        --subtitle-color: #666;
        --original-bg-start: #f0f4ff;
        --original-bg-end: #f0f4ff;
        --original-shadow: rgba(102, 126, 234, 0.2);
        --timeline-line-start: #667eea;
        --timeline-line-end: #764ba2;
        --step-bg: white;
        --step-border: #667eea;
        --step-hover-shadow: rgba(102, 126, 234, 0.3);
        --step-dot-bg: #667eea;
        --step-dot-border: white;
        --step-number-color: #667eea;
        --description-color: #666;
        --equation-bg: #fafafa;
        --final-bg-start: #4caf50;
        --final-bg-end: #45a049;
        --final-shadow: rgba(76, 175, 80, 0.3);
        --footer-color: white;
        --text-color: #333;
    }}
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 95%;
        margin: 0 auto;
        padding: 20px 10px;
        background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
        min-height: 100vh;
        color: var(--text-color);
        transition: background 0.3s ease;
    }}
    .container {{
        background: var(--container-bg);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px var(--container-shadow);
        transition: background 0.3s ease;
    }}
    /* Theme toggle switch */
    .theme-toggle {{
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 8px;
        background: var(--container-bg);
        padding: 8px 12px;
        border-radius: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }}
    .theme-toggle span {{
        font-size: 1.2em;
    }}
    .toggle-switch {{
        position: relative;
        width: 50px;
        height: 26px;
    }}
    .toggle-switch input {{
        opacity: 0;
        width: 0;
        height: 0;
    }}
    .toggle-slider {{
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #333;
        transition: 0.3s;
        border-radius: 26px;
    }}
    .toggle-slider:before {{
        position: absolute;
        content: "";
        height: 20px;
        width: 20px;
        left: 3px;
        bottom: 3px;
        background-color: white;
        transition: 0.3s;
        border-radius: 50%;
    }}
    input:checked + .toggle-slider {{
        background-color: #64b5f6;
    }}
    input:checked + .toggle-slider:before {{
        transform: translateX(24px);
    }}
    h1 {{
        color: var(--title-color);
        text-align: center;
        margin-bottom: 10px;
        font-size: 1.8em;
        transition: color 0.3s ease;
    }}
    .subtitle {{
        text-align: center;
        color: var(--subtitle-color);
        margin-bottom: 25px;
        transition: color 0.3s ease;
    }}
    .original {{
        background: linear-gradient(135deg, var(--original-bg-start), var(--original-bg-end));
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 15px var(--original-shadow);
        transition: background 0.3s ease;
    }}
    .timeline {{
        position: relative;
        padding-left: 30px;
    }}
    .timeline::before {{
        content: '';
        position: absolute;
        left: 10px;
        top: 0;
        bottom: 0;
        width: 3px;
        background: linear-gradient(to bottom, var(--timeline-line-start), var(--timeline-line-end));
        transition: background 0.3s ease;
    }}
    .step {{
        background: var(--step-bg);
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 20px;
        position: relative;
        border-left: 4px solid var(--step-border);
        transition: transform 0.2s, box-shadow 0.2s, background 0.3s ease;
    }}
    .step:hover {{
        transform: translateX(5px);
        box-shadow: 0 4px 20px var(--step-hover-shadow);
    }}
    .step::before {{
        content: '';
        position: absolute;
        left: -23px;
        top: 20px;
        width: 12px;
        height: 12px;
        background: var(--step-dot-bg);
        border-radius: 50%;
        border: 3px solid var(--step-dot-border);
        transition: background 0.3s ease;
    }}
    .step-number {{
        color: var(--step-number-color);
        font-weight: bold;
        font-size: 0.9em;
        margin-bottom: 5px;
        transition: color 0.3s ease;
    }}
    .description {{
        color: var(--description-color);
        font-size: 1em;
        margin-bottom: 10px;
        font-style: italic;
        transition: color 0.3s ease;
    }}
    .equation {{
        background: var(--equation-bg);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2em;
        transition: background 0.3s ease;
        overflow-x: auto;
        max-width: 100%;
    }}
    .final-result {{
        background: linear-gradient(135deg, var(--final-bg-start), var(--final-bg-end));
        padding: 20px;
        text-align: center;
        color: white;
        border-radius: 10px;
        margin-top: 30px;
        font-size: 1.2em;
        box-shadow: 0 4px 12px var(--final-shadow);
        transition: background 0.3s ease;
    }}
    footer {{
        text-align: center;
        margin-top: 30px;
        color: var(--footer-color);
        font-size: 0.9em;
        transition: color 0.3s ease;
    }}
    /* Collapsible substeps styles */
    .substeps-toggle {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 12px;
        padding: 8px 12px;
        background: rgba(100, 181, 246, 0.15);
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.9em;
        color: var(--step-number-color);
        border: 1px solid rgba(100, 181, 246, 0.3);
        transition: all 0.2s ease;
    }}
    .substeps-toggle:hover {{
        background: rgba(100, 181, 246, 0.25);
    }}
    .substeps-toggle .arrow {{
        transition: transform 0.3s ease;
        font-size: 0.8em;
    }}
    .substeps-toggle.expanded .arrow {{
        transform: rotate(90deg);
    }}
    .substeps-container {{
        display: none;
        margin-top: 12px;
        padding-left: 20px;
        border-left: 2px solid rgba(100, 181, 246, 0.3);
    }}
    .substeps-container.visible {{
        display: block;
        animation: slideDown 0.3s ease;
    }}
    @keyframes slideDown {{
        from {{ opacity: 0; transform: translateY(-10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .substep {{
        background: rgba(30, 40, 55, 0.6);
        border-radius: 8px;
        padding: 12px 15px;
        margin-bottom: 10px;
        border-left: 3px solid #90caf9;
    }}
    .substep-number {{
        color: #90caf9;
        font-weight: bold;
        font-size: 0.85em;
        margin-bottom: 4px;
    }}
    .substep-description {{
        color: var(--description-color);
        font-size: 0.9em;
        margin-bottom: 8px;
        font-style: italic;
    }}
    .substep-equation {{
        background: rgba(20, 30, 45, 0.8);
        padding: 10px;
        border-radius: 6px;
        text-align: center;
        font-size: 1em;
    }}
</style>
</head>
<body>
<div class="theme-toggle">
    <span>🌙</span>
    <label class="toggle-switch">
        <input type="checkbox" id="themeToggle" onchange="toggleTheme()">
        <span class="toggle-slider"></span>
    </label>
    <span>☀️</span>
</div>
<script>
    function toggleTheme() {{
        document.documentElement.classList.toggle('light');
        localStorage.setItem('theme', document.documentElement.classList.contains('light') ? 'light' : 'dark');
    }}
    // Load saved theme preference
    if (localStorage.getItem('theme') === 'light') {{
        document.documentElement.classList.add('light');
        document.getElementById('themeToggle').checked = true;
    }}
</script>
<div class="container">
    <h1>Equation Solving Steps</h1>
    <p class="subtitle">Solving for <strong>{}</strong></p>
    <div class="original">
        \(\textbf{{Original Equation:}}\)
        \[{}\]
    </div>
"#,
            escaped_title,
            html_escape(&self.var),
            original_latex
        )
    }

    fn render_solve_timeline(&mut self) -> String {
        let mut html = String::from("        <div class=\"timeline\">\n");

        let mut _current_eq = self.original_eq.clone();

        for (i, step) in self.steps.iter().enumerate() {
            let step_number = i + 1;

            // Generate LaTeX for the equation after this step
            let eq_latex = format!(
                "{} {} {}",
                LaTeXExpr {
                    context: self.context,
                    id: step.equation_after.lhs
                }
                .to_latex(),
                self.relop_to_latex(&step.equation_after.op),
                LaTeXExpr {
                    context: self.context,
                    id: step.equation_after.rhs
                }
                .to_latex()
            );

            // Start the step div
            html.push_str(&format!(
                r#"        <div class="step">
            <div class="step-number">Step {}</div>
            <div class="description">{}</div>
            <div class="equation">
                \[{}\]
            </div>
"#,
                step_number,
                html_escape(&step.description),
                eq_latex
            ));

            // Add collapsible substeps if any
            if !step.substeps.is_empty() {
                let substep_id = format!("substeps-{}", step_number);
                html.push_str(&format!(
                    r#"            <div class="substeps-toggle" onclick="toggleSubsteps('{}')">
                <span class="arrow">▶</span>
                <span>Show derivation ({} steps)</span>
            </div>
            <div id="{}" class="substeps-container">
"#,
                    substep_id,
                    step.substeps.len(),
                    substep_id
                ));

                for (j, substep) in step.substeps.iter().enumerate() {
                    let sub_eq_latex = format!(
                        "{} {} {}",
                        LaTeXExpr {
                            context: self.context,
                            id: substep.equation_after.lhs
                        }
                        .to_latex(),
                        self.relop_to_latex(&substep.equation_after.op),
                        LaTeXExpr {
                            context: self.context,
                            id: substep.equation_after.rhs
                        }
                        .to_latex()
                    );

                    html.push_str(&format!(
                        r#"                <div class="substep">
                    <div class="substep-number">Step {}.{}</div>
                    <div class="substep-description">{}</div>
                    <div class="substep-equation">
                        \[{}\]
                    </div>
                </div>
"#,
                        step_number,
                        j + 1,
                        html_escape(&substep.description),
                        sub_eq_latex
                    ));
                }

                html.push_str("            </div>\n");
            }

            html.push_str("        </div>\n");

            _current_eq = step.equation_after.clone();
        }

        // Add final result showing the SOLUTION SET, not the last equation
        let solution_latex = self.solution_set_to_latex();

        html.push_str(&format!(
            r#"        </div>
        <div class="final-result">
            \(\textbf{{Solution: }} {} = \)
            \[{}\]
        </div>
    </div>
"#,
            html_escape(&self.var),
            solution_latex
        ));
        html
    }

    /// Convert SolutionSet to LaTeX representation
    fn solution_set_to_latex(&self) -> String {
        use cas_ast::SolutionSet;
        match self.solution_set {
            SolutionSet::Empty => r"\emptyset".to_string(),
            SolutionSet::AllReals => r"\mathbb{R}".to_string(),
            SolutionSet::Discrete(exprs) => {
                let elements: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        LaTeXExpr {
                            context: self.context,
                            id: *e,
                        }
                        .to_latex()
                    })
                    .collect();
                format!(r"\left\{{ {} \right\}}", elements.join(", "))
            }
            SolutionSet::Continuous(interval) => self.interval_to_latex(interval),
            SolutionSet::Union(intervals) => {
                let parts: Vec<String> = intervals
                    .iter()
                    .map(|i| self.interval_to_latex(i))
                    .collect();
                parts.join(r" \cup ")
            }
            SolutionSet::Residual(expr) => {
                // Show the residual expression as-is
                LaTeXExpr {
                    context: self.context,
                    id: *expr,
                }
                .to_latex()
            }
            SolutionSet::Conditional(cases) => {
                // V2.0 Phase 2C: Pretty-print conditional solutions as piecewise LaTeX
                // V2.1: Use "otherwise" without "if" prefix for natural reading
                // V2.x: Skip "otherwise" cases that only contain Residual (not useful info)
                let case_strs: Vec<String> = cases
                    .iter()
                    .filter_map(|case| {
                        // Skip "otherwise" cases that only contain Residual
                        if case.when.is_otherwise()
                            && matches!(&case.then.solutions, SolutionSet::Residual(_))
                        {
                            return None;
                        }
                        let sol_latex = self.solution_set_inner_to_latex(&case.then.solutions);
                        if case.when.is_otherwise() {
                            Some(format!("{} & \\text{{otherwise}}", sol_latex))
                        } else {
                            let cond_latex =
                                cas_formatter::condition_set_to_latex(&case.when, self.context);
                            Some(format!("{} & \\text{{if }} {}", sol_latex, cond_latex))
                        }
                    })
                    .collect();
                // If only one case remains after filtering, render without \begin{cases}
                if case_strs.len() == 1 {
                    // Extract just the solution part (before the " & \text{if}")
                    let single = &case_strs[0];
                    if let Some(idx) = single.find(r" & \text{if}") {
                        return single[..idx].to_string();
                    }
                }
                format!(r"\begin{{cases}} {} \end{{cases}}", case_strs.join(r" \\ "))
            }
        }
    }

    /// V2.0 Phase 2C: Render inner solution set to LaTeX (for Conditional cases)
    fn solution_set_inner_to_latex(&self, solution_set: &SolutionSet) -> String {
        match solution_set {
            SolutionSet::Empty => r"\emptyset".to_string(),
            SolutionSet::AllReals => r"\mathbb{R}".to_string(),
            SolutionSet::Discrete(exprs) => {
                let elements: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        LaTeXExpr {
                            context: self.context,
                            id: *e,
                        }
                        .to_latex()
                    })
                    .collect();
                format!(r"\left\{{ {} \right\}}", elements.join(", "))
            }
            SolutionSet::Continuous(interval) => self.interval_to_latex(interval),
            SolutionSet::Union(intervals) => {
                let parts: Vec<String> = intervals
                    .iter()
                    .map(|i| self.interval_to_latex(i))
                    .collect();
                parts.join(r" \cup ")
            }
            SolutionSet::Residual(expr) => LaTeXExpr {
                context: self.context,
                id: *expr,
            }
            .to_latex(),
            SolutionSet::Conditional(_) => r"\text{(nested conditional)}".to_string(),
        }
    }

    /// Convert Interval to LaTeX representation
    fn interval_to_latex(&self, interval: &cas_ast::Interval) -> String {
        use cas_ast::BoundType;
        let left = match interval.min_type {
            BoundType::Open => "(",
            BoundType::Closed => "[",
        };
        let right = match interval.max_type {
            BoundType::Open => ")",
            BoundType::Closed => "]",
        };
        let min_latex = LaTeXExpr {
            context: self.context,
            id: interval.min,
        }
        .to_latex();
        let max_latex = LaTeXExpr {
            context: self.context,
            id: interval.max,
        }
        .to_latex();
        format!(r"{}{}, {}{}", left, min_latex, max_latex, right)
    }

    fn relop_to_latex(&self, op: &cas_ast::RelOp) -> &'static str {
        use cas_ast::RelOp;
        match op {
            RelOp::Eq => "=",
            RelOp::Neq => "\\neq",
            RelOp::Lt => "<",
            RelOp::Gt => ">",
            RelOp::Leq => "\\leq",
            RelOp::Geq => "\\geq",
        }
    }

    fn html_footer_solve() -> &'static str {
        r#"    <script>
        function toggleSubsteps(id) {
            const container = document.getElementById(id);
            const toggle = document.querySelector(`[onclick*="${id}"]`);
            if (container.classList.contains('visible')) {
                container.classList.remove('visible');
                toggle.classList.remove('expanded');
                toggle.querySelector('span:last-child').textContent = 'Show derivation (' + container.children.length + ' steps)';
            } else {
                container.classList.add('visible');
                toggle.classList.add('expanded');
                toggle.querySelector('span:last-child').textContent = 'Hide derivation';
                // Re-render MathJax for the newly visible content
                if (window.MathJax) {
                    MathJax.typeset([container]);
                }
            }
        }
    </script>
    <footer>
        Generated by Rust CAS Engine - Equation Solver
    </footer>
</body>
</html>"#
    }
}
