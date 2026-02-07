use super::escape::{html_escape, latex_escape};
use super::latex_clean::clean_latex_identities;
use super::path::{
    diff_find_all_paths_to_expr, diff_find_path_to_expr, diff_find_paths_by_structure,
    extract_add_terms, find_path_to_expr, navigate_to_subexpr, pathstep_to_u8,
};
use crate::step::{pathsteps_to_expr_path, PathStep, Step};
use cas_ast::{
    Context, Expr, ExprId, ExprPath, HighlightColor, HighlightConfig, LaTeXExprHighlighted,
    PathHighlightConfig, PathHighlightedLatexRenderer,
};

/// Timeline HTML generator - exports simplification steps to interactive HTML
pub struct TimelineHtml<'a> {
    context: &'a mut Context,
    steps: &'a [Step],
    original_expr: ExprId,
    simplified_result: Option<ExprId>, // Optional: the final simplified result
    title: String,
    verbosity_level: VerbosityLevel,
    /// V2.12.13: Global requires inferred from input expression.
    /// Shown at the end of the timeline, after final result.
    global_requires: Vec<crate::implicit_domain::ImplicitCondition>,
    /// V2.14.40: Style preferences derived from input string for consistent root rendering
    style_prefs: cas_ast::StylePreferences,
}

#[derive(Debug, Clone, Copy)]
pub enum VerbosityLevel {
    Low,     // Only high-importance steps (Factor, Expand, Integrate, etc.)
    Normal,  // Medium+ importance steps (most transformations)
    Verbose, // All steps including trivial ones
}

impl VerbosityLevel {
    /// Check if a step should be shown at this verbosity level
    /// Uses step.get_importance() as the single source of truth
    fn should_show_step(&self, step: &Step) -> bool {
        use crate::step::ImportanceLevel;

        match self {
            VerbosityLevel::Verbose => true,
            VerbosityLevel::Low => step.get_importance() >= ImportanceLevel::High,
            VerbosityLevel::Normal => step.get_importance() >= ImportanceLevel::Medium,
        }
    }
}

impl<'a> TimelineHtml<'a> {
    pub fn new(
        context: &'a mut Context,
        steps: &'a [Step],
        original_expr: ExprId,
        verbosity: VerbosityLevel,
    ) -> Self {
        Self::new_with_result(context, steps, original_expr, None, verbosity)
    }

    /// Create a new TimelineHtml with a known simplified result
    pub fn new_with_result(
        context: &'a mut Context,
        steps: &'a [Step],
        original_expr: ExprId,
        simplified_result: Option<ExprId>,
        verbosity: VerbosityLevel,
    ) -> Self {
        Self::new_with_result_and_style(
            context,
            steps,
            original_expr,
            simplified_result,
            verbosity,
            None,
        )
    }

    /// Create a new TimelineHtml with style preferences derived from input string
    /// V2.14.40: Enables consistent root rendering (exponential vs radical)
    pub fn new_with_result_and_style(
        context: &'a mut Context,
        steps: &'a [Step],
        original_expr: ExprId,
        simplified_result: Option<ExprId>,
        verbosity: VerbosityLevel,
        input_string: Option<&str>,
    ) -> Self {
        use crate::implicit_domain::infer_implicit_domain;
        use crate::semantics::ValueDomain;

        // V2.14.40: Compute style preferences from input string
        let signals = input_string.map(cas_ast::ParseStyleSignals::from_input_string);
        let style_prefs = cas_ast::StylePreferences::from_expression_with_signals(
            context,
            original_expr,
            signals.as_ref(),
        );

        // V2.14.40: Generate title using styled renderer for consistent root rendering
        let empty_config = PathHighlightConfig::new();
        let title = PathHighlightedLatexRenderer {
            context,
            id: original_expr,
            path_highlights: &empty_config,
            hints: None,
            style_prefs: Some(&style_prefs),
        }
        .to_latex();

        // V2.12.13: Infer global requires from input expression
        // This ensures timeline shows the same requires as REPL
        let input_domain = infer_implicit_domain(context, original_expr, ValueDomain::RealOnly);
        let global_requires: Vec<_> = input_domain.conditions().iter().cloned().collect();

        Self {
            context,
            steps,
            original_expr,
            simplified_result,
            title,
            verbosity_level: verbosity,
            global_requires,
            style_prefs,
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

        // Enrich steps with didactic sub-steps
        let enriched_steps =
            crate::didactic::enrich_steps(self.context, self.original_expr, self.steps.to_vec());

        let mut html = Self::html_header(&self.title);
        html.push_str(&self.render_timeline_filtered_enriched(&filtered_steps, &enriched_steps));
        html.push_str(Self::html_footer());

        // Clean up identity patterns like "\cdot 1" for better display
        clean_latex_identities(&html)
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
            --content-bg: rgba(30, 40, 55, 0.9);
            --content-border: rgba(100, 181, 246, 0.2);
            --content-h3-color: #90caf9;
            --math-bg: rgba(30, 40, 55, 0.8);
            --math-border: #64b5f6;
            --math-before-border: #ff9800;
            --math-before-bg: rgba(255, 152, 0, 0.1);
            --math-after-border: #4caf50;
            --math-after-bg: rgba(76, 175, 80, 0.1);
            --math-strong-color: #b0bec5;
            --rule-bg: rgba(100, 181, 246, 0.1);
            --rule-color: #90caf9;
            --rule-border: rgba(100, 181, 246, 0.4);
            --rule-name-color: #bb86fc;
            --local-change-bg: rgba(30, 40, 55, 0.8);
            --final-bg-start: #2e7d32;
            --final-bg-end: #1b5e20;
            --final-shadow: rgba(76, 175, 80, 0.3);
            --footer-color: #90caf9;
            --substeps-bg: rgba(255, 152, 0, 0.1);
            --substeps-border: rgba(255, 152, 0, 0.3);
            --substeps-summary-color: #ffb74d;
            --substeps-summary-hover: #ffa726;
            --substeps-content-bg: rgba(30, 40, 55, 0.9);
            --substep-border: rgba(255, 152, 0, 0.2);
            --substep-desc-color: #b0bec5;
            --substep-math-bg: rgba(30, 40, 55, 0.8);
            --warning-bg: rgba(255, 193, 7, 0.15);
            --warning-border: rgba(255, 193, 7, 0.4);
            --warning-color: #ffd54f;
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
            --content-bg: #fafafa;
            --content-border: #e0e0e0;
            --content-h3-color: #667eea;
            --math-bg: #fafafa;
            --math-border: #667eea;
            --math-before-border: #ff9800;
            --math-before-bg: #fff8f0;
            --math-after-border: #4caf50;
            --math-after-bg: #f0fff4;
            --math-strong-color: #666;
            --rule-bg: #f9f5ff;
            --rule-color: #667eea;
            --rule-border: #667eea;
            --rule-name-color: #764ba2;
            --local-change-bg: white;
            --final-bg-start: #4caf50;
            --final-bg-end: #45a049;
            --final-shadow: rgba(76, 175, 80, 0.3);
            --footer-color: white;
            --substeps-bg: #fff8e1;
            --substeps-border: #ffcc80;
            --substeps-summary-color: #ef6c00;
            --substeps-summary-hover: #e65100;
            --substeps-content-bg: white;
            --substep-border: #ffe0b2;
            --substep-desc-color: #795548;
            --substep-math-bg: #fafafa;
            --warning-bg: #fff3cd;
            --warning-border: #ffc107;
            --warning-color: #856404;
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
        .step-content {{
            background: var(--content-bg);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid var(--content-border);
            transition: transform 0.2s, box-shadow 0.2s, background 0.3s ease;
        }}
        .step-content:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 16px var(--step-hover-shadow);
        }}
        .step-content h3 {{
            margin-top: 0;
            color: var(--content-h3-color);
            font-size: 1.1em;
            transition: color 0.3s ease;
        }}
        .math-expr {{
            padding: 12px 15px;
            background: var(--math-bg);
            border-left: 4px solid var(--math-border);
            margin: 10px 0;
            border-radius: 4px;
            font-size: 1.05em;
            transition: background 0.3s ease;
            overflow-x: auto;
            max-width: 100%;
        }}
        .math-expr.before {{
            border-left-color: var(--math-before-border);
            background: var(--math-before-bg);
        }}
        .math-expr.after {{
            border-left-color: var(--math-after-border);
            background: var(--math-after-bg);
        }}
        .math-expr strong {{
            color: var(--math-strong-color);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            transition: color 0.3s ease;
        }}
        .rule-description {{
            text-align: center;
            padding: 12px 20px;
            margin: 15px 0;
            background: var(--rule-bg);
            border-radius: 6px;
            font-size: 0.95em;
            color: var(--rule-color);
            border: 2px dashed var(--rule-border);
            transition: background 0.3s ease, color 0.3s ease;
        }}
        .local-change {{
            font-size: 1.1em;
            margin: 8px 0;
            padding: 10px;
            background: var(--local-change-bg);
            border-radius: 4px;
            text-align: center;
            transition: background 0.3s ease;
        }}
        .rule-name {{
            font-size: 0.85em;
            color: var(--rule-name-color);
            font-weight: bold;
            margin-bottom: 5px;
            transition: color 0.3s ease;
        }}
        .final-result {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, var(--final-bg-start), var(--final-bg-end));
            color: white;
            border-radius: 10px;
            margin-top: 30px;
            font-size: 1.2em;
            box-shadow: 0 4px 12px var(--final-shadow);
            transition: background 0.3s ease;
        }}
        .poly-badge {{
            background: rgba(255,255,255,0.25);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75em;
            margin-left: 10px;
            font-weight: normal;
        }}
        .poly-output {{
            text-align: left;
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.65em;
            max-height: 400px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-all;
            line-height: 1.4;
        }}
        footer {{
            text-align: center;
            margin-top: 30px;
            color: var(--footer-color);
            font-size: 0.9em;
            transition: color 0.3s ease;
        }}
        /* Expandable details for didactic sub-steps */
        .substeps-details {{
            margin: 10px 0;
            padding: 10px 15px;
            background: var(--substeps-bg);
            border: 1px solid var(--substeps-border);
            border-radius: 8px;
            font-size: 0.95em;
            transition: background 0.3s ease;
        }}
        .substeps-details summary {{
            cursor: pointer;
            font-weight: bold;
            color: var(--substeps-summary-color);
            padding: 5px 0;
            transition: color 0.3s ease;
        }}
        .substeps-details summary:hover {{
            color: var(--substeps-summary-hover);
        }}
        .substeps-content {{
            margin-top: 10px;
            padding: 10px;
            background: var(--substeps-content-bg);
            border-radius: 6px;
            transition: background 0.3s ease;
        }}
        .substep {{
            padding: 8px 0;
            border-bottom: 1px dashed var(--substep-border);
        }}
        .substep:last-child {{
            border-bottom: none;
        }}
        .substep-desc {{
            font-weight: 500;
            color: var(--substep-desc-color);
            display: block;
            margin-bottom: 5px;
            transition: color 0.3s ease;
        }}
        .substep-math {{
            padding: 5px 10px;
            background: var(--substep-math-bg);
            border-radius: 4px;
            text-align: center;
            transition: background 0.3s ease;
        }}
        .domain-warning {{
            margin-top: 10px;
            padding: 8px 12px;
            background: var(--warning-bg);
            border: 1px solid var(--warning-border);
            border-radius: 6px;
            color: var(--warning-color);
            font-size: 0.9em;
            transition: background 0.3s ease, color 0.3s ease;
        }}
        .domain-warning::before {{
            content: '⚠ ';
        }}
        .domain-requires {{
            margin-top: 10px;
            padding: 8px 12px;
            background: rgba(33, 150, 243, 0.15);
            border: 1px solid rgba(33, 150, 243, 0.4);
            border-radius: 6px;
            color: #64b5f6;
            font-size: 0.9em;
            transition: background 0.3s ease, color 0.3s ease;
        }}
        .domain-requires::before {{
            content: 'ℹ️ ';
        }}
        .global-requires {{
            margin-top: 15px;
            padding: 12px 16px;
            background: rgba(33, 150, 243, 0.1);
            border: 2px solid rgba(33, 150, 243, 0.5);
            border-radius: 8px;
            color: #90caf9;
            font-size: 1em;
        }}
        .global-requires strong {{
            color: #64b5f6;
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
                // Check if left side is Neg - if so, preserve the Neg wrapper
                if let Expr::Neg(inner) = self.context.get(l).clone() {
                    // Traverse into the Neg and wrap result back in Neg
                    let new_inner =
                        self.reconstruct_global_expr(inner, remaining_path, replacement);
                    let new_neg = self.context.add(Expr::Neg(new_inner));
                    self.context.add(Expr::Add(new_neg, r))
                } else {
                    // Normal case
                    let new_l = self.reconstruct_global_expr(l, remaining_path, replacement);
                    self.context.add(Expr::Add(new_l, r))
                }
            }
            // Special case: Sub(a,b) may have been canonicalized to Add(a, Neg(b))
            // When PathStep::Right expects to modify the original "b", we need to
            // traverse into the Neg wrapper and reconstruct there.
            (Expr::Add(l, r), PathStep::Right) => {
                // Check if right side is Neg - if so, this might be a canonicalized Sub
                if let Expr::Neg(inner) = self.context.get(r).clone() {
                    // Traverse into the Neg and wrap result back in Neg
                    let new_inner =
                        self.reconstruct_global_expr(inner, remaining_path, replacement);
                    let new_neg = self.context.add(Expr::Neg(new_inner));
                    self.context.add(Expr::Add(l, new_neg))
                } else {
                    // Normal case - not a canonicalized Sub
                    let new_r = self.reconstruct_global_expr(r, remaining_path, replacement);
                    self.context.add(Expr::Add(l, new_r))
                }
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
                let mut new_args = args;
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

    /// Render timeline with enriched sub-steps (expandable details)
    fn render_timeline_filtered_enriched(
        &mut self,
        filtered_steps: &[&Step],
        enriched_steps: &[crate::didactic::EnrichedStep],
    ) -> String {
        let mut html = String::from("        <div class=\"timeline\">\n");

        // Build display hints for consistent sqrt notation (including final result)
        let display_hints = crate::display_context::build_display_context_with_result(
            self.context,
            self.original_expr,
            self.steps,
            self.simplified_result,
        );

        let mut step_number = 0;
        let mut last_global_after = self.original_expr; // Track final result across all steps
                                                        // Track if substeps have been shown (show only once on first visible step)
        let mut sub_steps_shown = false;

        // Track which steps to display
        let filtered_indices: std::collections::HashSet<_> =
            filtered_steps.iter().map(|s| *s as *const Step).collect();

        // Iterate over ALL steps to correctly update the global state
        for (step_idx, step) in self.steps.iter().enumerate() {
            // Use step.global_before/global_after if available (pre-computed with exponent simplification)
            // Otherwise fall back to recalculated state
            let global_before_expr = step.global_before.unwrap_or_else(|| {
                if step_idx == 0 {
                    self.original_expr
                } else {
                    // Reconstruct from previous step's global_after
                    self.steps
                        .get(step_idx - 1)
                        .and_then(|prev| prev.global_after)
                        .unwrap_or(self.original_expr)
                }
            });
            let global_after_expr = step.global_after.unwrap_or_else(|| {
                self.reconstruct_global_expr(global_before_expr, &step.path, step.after)
            });
            last_global_after = global_after_expr; // Always update for final result

            let step_ptr = step as *const Step;
            if !filtered_indices.contains(&step_ptr) {
                continue;
            }
            step_number += 1;

            // Generate global BEFORE with red highlight on the transformed subtree
            // V2.9.16: Using PathHighlightedLatexRenderer to highlight by path, not ExprId
            // This ensures only the specific occurrence is highlighted, not all identical values
            //
            // V2.9.17: When before_local differs from step.before, try to extend path to focus area
            // If path cannot be found (before_local is dynamically constructed), use ExprId-based highlighting
            //
            // V2.9.25: When before_local is an Add node (representing multiple matched terms),
            // always use multi-term highlighting. Single-path highlighting would only highlight
            // one subtree when the matched terms may come from different parts of the expression.
            let (global_before, global_after) = if let Some(before_local) =
                step.before_local.filter(|&bl| bl != step.before)
            {
                // V2.9.25: Check if before_local is an Add node. If so, use multi-term highlighting
                // to ensure all terms are highlighted, not just the subtree that happens to match.
                let before_local_is_add = matches!(self.context.get(before_local), Expr::Add(_, _));

                // Try to find path from step.before to before_local
                let focus_path = if !before_local_is_add {
                    find_path_to_expr(self.context, step.before, before_local)
                } else {
                    // Skip single-path search for Add nodes - will use multi-term highlighting
                    Vec::new()
                };

                if !focus_path.is_empty() {
                    // Path found - extend step.path and use path-based highlighting
                    // This branch is only used for non-Add before_local nodes
                    let mut extended = pathsteps_to_expr_path(&step.path);
                    for ps in &focus_path {
                        extended.push(pathstep_to_u8(ps));
                    }
                    let mut before_config = PathHighlightConfig::new();
                    before_config.add(extended.clone(), HighlightColor::Red);
                    let before = PathHighlightedLatexRenderer {
                        context: self.context,
                        id: global_before_expr,
                        path_highlights: &before_config,
                        hints: Some(&display_hints),
                        style_prefs: Some(&self.style_prefs),
                    }
                    .to_latex();

                    let mut after_config = PathHighlightConfig::new();
                    after_config.add(extended, HighlightColor::Green);
                    let after = PathHighlightedLatexRenderer {
                        context: self.context,
                        id: global_after_expr,
                        path_highlights: &after_config,
                        hints: Some(&display_hints),
                        style_prefs: Some(&self.style_prefs),
                    }
                    .to_latex();

                    (before, after)
                } else {
                    // Path not found (before_local is dynamically constructed)
                    // V2.9.19: Use multi-path highlighting with paths to individual terms
                    // This fixes the regression where ExprId-based highlighting would mark
                    // all identical values (e.g., all 'x' symbols) instead of just those
                    // within the focus area.
                    let focus_before = before_local;
                    let focus_after = step.after_local.unwrap_or(step.after);

                    // BEFORE: Extract terms from focus_before and find paths to each within
                    // the subexpression at step.path (NOT the entire global_before_expr).
                    // This handles dynamically constructed expressions like Add(x, x) or Sub(frac1, frac2)
                    let focus_terms = extract_add_terms(self.context, focus_before);
                    let step_path_prefix = pathsteps_to_expr_path(&step.path);

                    // Navigate to the subexpression at step.path
                    let subexpr_at_path =
                        navigate_to_subexpr(self.context, global_before_expr, &step_path_prefix);

                    // V2.14.32: Find the path to before_local within subexpr_at_path
                    // This limits the search scope to only the part that's actually being transformed.
                    // For example, when transforming a numerator, we should not highlight
                    // occurrences in the denominator that happen to share an ExprId.
                    let before_local_path =
                        diff_find_path_to_expr(self.context, subexpr_at_path, focus_before);

                    // Determine the actual scope: either before_local subtree or full subexpr_at_path
                    let (search_scope, scope_path_prefix) =
                        if let Some(path_to_local) = &before_local_path {
                            // before_local exists in the tree - limit search to that subtree
                            let local_scope =
                                navigate_to_subexpr(self.context, subexpr_at_path, path_to_local);
                            let mut full_prefix = step_path_prefix.clone();
                            full_prefix.extend(path_to_local.clone());
                            (local_scope, full_prefix)
                        } else {
                            // before_local is dynamically constructed - use full subexpr_at_path
                            (subexpr_at_path, step_path_prefix.clone())
                        };

                    let mut found_paths: Vec<ExprPath> = Vec::new();
                    for term in &focus_terms {
                        let paths_before = found_paths.len();

                        // V2.14.32: Search within the scoped subtree only
                        // This limits highlighting to the focused area, not ALL occurrences globally
                        for sub_path in
                            diff_find_all_paths_to_expr(self.context, search_scope, *term)
                        {
                            // Prepend scope_path_prefix to get the full path from root
                            let mut full_path = scope_path_prefix.clone();
                            full_path.extend(sub_path.clone());
                            // Avoid duplicate paths
                            if !found_paths.contains(&full_path) {
                                found_paths.push(full_path);
                            }
                        }

                        // V2.9.24: If ExprId-based search found nothing for THIS term,
                        // try structural search. This handles dynamically constructed terms
                        // (e.g., from inverse_trig rules) where ExprIds differ but
                        // expressions are structurally equivalent.
                        if found_paths.len() == paths_before {
                            for sub_path in
                                diff_find_paths_by_structure(self.context, search_scope, *term)
                            {
                                let mut full_path = scope_path_prefix.clone();
                                full_path.extend(sub_path.clone());
                                if !found_paths.contains(&full_path) {
                                    found_paths.push(full_path);
                                }
                            }
                        }
                    }

                    let before = if !found_paths.is_empty() {
                        // Use path-based multi-term highlighting for accuracy
                        let mut before_config = PathHighlightConfig::new();
                        for path in found_paths {
                            before_config.add(path, HighlightColor::Red);
                        }
                        PathHighlightedLatexRenderer {
                            context: self.context,
                            id: global_before_expr,
                            path_highlights: &before_config,
                            hints: Some(&display_hints),
                            style_prefs: Some(&self.style_prefs),
                        }
                        .to_latex()
                    } else {
                        // Fallback: use step.path if no paths found to individual terms
                        let expr_path = pathsteps_to_expr_path(&step.path);
                        let mut before_config = PathHighlightConfig::new();
                        before_config.add(expr_path, HighlightColor::Red);
                        PathHighlightedLatexRenderer {
                            context: self.context,
                            id: global_before_expr,
                            path_highlights: &before_config,
                            hints: Some(&display_hints),
                            style_prefs: Some(&self.style_prefs),
                        }
                        .to_latex()
                    };
                    // AFTER: Try to find path to focus_after in global_after_expr
                    // This handles cases where tree reordering changes the structure
                    let after = if let Some(after_path) =
                        diff_find_path_to_expr(self.context, global_after_expr, focus_after)
                    {
                        // Found path - use path-based highlighting for accuracy
                        let mut after_config = PathHighlightConfig::new();
                        after_config.add(after_path, HighlightColor::Green);
                        PathHighlightedLatexRenderer {
                            context: self.context,
                            id: global_after_expr,
                            path_highlights: &after_config,
                            hints: Some(&display_hints),
                            style_prefs: Some(&self.style_prefs),
                        }
                        .to_latex()
                    } else {
                        // Path not found - fall back to ExprId-based highlighting
                        let mut after_config = HighlightConfig::new();
                        after_config.add(focus_after, HighlightColor::Green);
                        LaTeXExprHighlighted {
                            context: self.context,
                            id: global_after_expr,
                            highlights: &after_config,
                        }
                        .to_latex()
                    };

                    (before, after)
                }
            } else {
                // Standard case: use step.path for highlighting
                let expr_path = pathsteps_to_expr_path(&step.path);
                let mut before_config = PathHighlightConfig::new();
                before_config.add(expr_path.clone(), HighlightColor::Red);
                let before = PathHighlightedLatexRenderer {
                    context: self.context,
                    id: global_before_expr,
                    path_highlights: &before_config,
                    hints: Some(&display_hints),
                    style_prefs: Some(&self.style_prefs),
                }
                .to_latex();

                let mut after_config = PathHighlightConfig::new();
                after_config.add(expr_path, HighlightColor::Green);
                let after = PathHighlightedLatexRenderer {
                    context: self.context,
                    id: global_after_expr,
                    path_highlights: &after_config,
                    hints: Some(&display_hints),
                    style_prefs: Some(&self.style_prefs),
                }
                .to_latex();

                (before, after)
            };

            // Note: We intentionally do NOT skip steps where LaTeX renders identically.
            // The LaTeX renderer normalizes expressions (e.g., 1*x → x), which would
            // incorrectly filter Identity Property steps. Upstream to_display_steps
            // already removes structural no-ops (before == after ExprId).

            // Generate colored rule display: red antecedent → green consequent
            // Use before_local/after_local (Focus) if available, otherwise fall back to before/after
            let focus_before = step.before_local.unwrap_or(step.before);
            let focus_after = step.after_local.unwrap_or(step.after);

            let mut rule_before_config = HighlightConfig::new();
            rule_before_config.add(focus_before, HighlightColor::Red);
            let local_before_colored = cas_ast::LaTeXExprHighlightedWithHints {
                context: self.context,
                id: focus_before,
                highlights: &rule_before_config,
                hints: &display_hints,
                style_prefs: Some(&self.style_prefs),
            }
            .to_latex();

            let mut rule_after_config = HighlightConfig::new();
            rule_after_config.add(focus_after, HighlightColor::Green);
            let local_after_colored = cas_ast::LaTeXExprHighlightedWithHints {
                context: self.context,
                id: focus_after,
                highlights: &rule_after_config,
                hints: &display_hints,
                style_prefs: Some(&self.style_prefs),
            }
            .to_latex();

            let local_change_latex = format!(
                "{} \\rightarrow {}",
                local_before_colored, local_after_colored
            );

            // Get enriched sub-steps for this step
            // Detect enrichment type FIRST
            let sub_steps_html = if let Some(enriched) = enriched_steps.get(step_idx) {
                if !enriched.sub_steps.is_empty() {
                    // Detect type from sub-step descriptions
                    let has_fraction_sum = enriched.sub_steps.iter().any(|s| {
                        s.description.contains("common denominator")
                            || s.description.contains("Sum the fractions")
                    });
                    let has_factorization = enriched.sub_steps.iter().any(|s| {
                        s.description.contains("Cancel common factor")
                            || s.description.contains("Factor")
                    });
                    let has_nested_fraction = enriched.sub_steps.iter().any(|s| {
                        s.description.contains("Invertir") || s.description.contains("denominador")
                    });

                    // Per-step enrichments (nested fractions, factorization): always show
                    // Global enrichments (fraction sums): show only once
                    let should_show = if has_nested_fraction || has_factorization {
                        true // Per-step: always show for each relevant step
                    } else {
                        !sub_steps_shown // Global (fraction sums or default): only show once
                    };

                    if should_show {
                        // Mark as shown for global enrichments only
                        if has_fraction_sum && !has_nested_fraction && !has_factorization {
                            sub_steps_shown = true;
                        }

                        let header = if has_nested_fraction {
                            "Simplificación de fracción compleja"
                        } else if has_fraction_sum {
                            "Suma de fracciones"
                        } else if has_factorization {
                            "Factorización de polinomios"
                        } else {
                            "Pasos intermedios"
                        };

                        let mut details_html = format!(
                            r#"<details class="substeps-details">
                            <summary>{}</summary>
                            <div class="substeps-content">"#,
                            header
                        );
                        for sub in &enriched.sub_steps {
                            details_html.push_str(&format!(
                                r#"<div class="substep">
                                    <span class="substep-desc">{}</span>"#,
                                html_escape(&sub.description)
                            ));
                            if !sub.before_expr.is_empty() {
                                details_html.push_str(&format!(
                                    r#"<div class="substep-math">\[{} \rightarrow {}\]</div>"#,
                                    sub.before_expr, sub.after_expr
                                ));
                            }
                            details_html.push_str("</div>");
                        }
                        details_html.push_str("</div></details>");
                        details_html
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            // V2.14.45: Build HTML for rule-provided substeps (educational explanations)
            let rule_substeps_html = if !step.substeps.is_empty() {
                let mut details_html = String::from(
                    r#"<details class="substeps-details" open>
                    <summary>Pasos didácticos</summary>
                    <div class="substeps-content">"#,
                );
                for substep in &step.substeps {
                    details_html.push_str(&format!(
                        r#"<div class="substep">
                            <strong>[{}]</strong>"#,
                        html_escape(&substep.title)
                    ));
                    for line in &substep.lines {
                        details_html.push_str(&format!(
                            r#"<div class="substep-line">• {}</div>"#,
                            html_escape(line)
                        ));
                    }
                    details_html.push_str("</div>");
                }
                details_html.push_str("</div></details>");
                details_html
            } else {
                String::new()
            };

            // V2.12.13: Build assumption HTML from assumption_events, filtered and grouped by kind
            let domain_html = if !step.assumption_events.is_empty() {
                use crate::assumptions::AssumptionKind;

                // Filter to displayable events only
                let displayable: Vec<_> = step
                    .assumption_events
                    .iter()
                    .filter(|e| e.kind.should_display())
                    .collect();

                if displayable.is_empty() {
                    String::new()
                } else {
                    let mut parts = Vec::new();

                    // Group by kind and format with icons
                    let requires: Vec<_> = displayable
                        .iter()
                        .filter(|e| matches!(e.kind, AssumptionKind::RequiresIntroduced))
                        .map(|e| html_escape(&e.message))
                        .collect();
                    if !requires.is_empty() {
                        parts.push(format!("ℹ️ Requires: {}", requires.join(", ")));
                    }

                    let branches: Vec<_> = displayable
                        .iter()
                        .filter(|e| matches!(e.kind, AssumptionKind::BranchChoice))
                        .map(|e| html_escape(&e.message))
                        .collect();
                    if !branches.is_empty() {
                        parts.push(format!("🔀 Branch: {}", branches.join(", ")));
                    }

                    let domain_ext: Vec<_> = displayable
                        .iter()
                        .filter(|e| matches!(e.kind, AssumptionKind::DomainExtension))
                        .map(|e| html_escape(&e.message))
                        .collect();
                    if !domain_ext.is_empty() {
                        parts.push(format!("🧿 Domain: {}", domain_ext.join(", ")));
                    }

                    let assumes: Vec<_> = displayable
                        .iter()
                        .filter(|e| matches!(e.kind, AssumptionKind::HeuristicAssumption))
                        .map(|e| html_escape(&e.message))
                        .collect();
                    if !assumes.is_empty() {
                        parts.push(format!("⚠️ Assumes: {}", assumes.join(", ")));
                    }

                    if parts.is_empty() {
                        String::new()
                    } else {
                        format!(
                            r#"                    <div class="domain-assumptions">{}</div>
"#,
                            parts.join("<br/>")
                        )
                    }
                }
            } else {
                String::new()
            };

            // V2.12.13: Per-step requires removed - they are now shown once in the
            // global-requires section at the end of the timeline. This avoids redundancy
            // when the same conditions appear on multiple steps.
            let requires_html = String::new();

            html.push_str(&format!(
                r#"            <div class="step">
                <div class="step-number">{}</div>
                <div class="step-content">
                    <h3>{}</h3>
                    <div class="math-expr before">
                        \(\textbf{{Before:}}\)
                        \[{}\]
                    </div>
                    {}
                    <div class="rule-description">
                        <div class="rule-name">\(\text{{{}}}\)</div>
                        <div class="local-change">
                            \[{}\]
                        </div>
                    </div>
                    {}
                    <div class="math-expr after">
                        \(\textbf{{After:}}\)
                        \[{}\]
                    </div>
{}{}                </div>
            </div>
"#,
                step_number,
                html_escape(&step.rule_name),
                global_before,
                sub_steps_html,
                step.description,
                local_change_latex,
                rule_substeps_html, // Add rule-provided educational substeps
                global_after,
                requires_html,
                domain_html
            ));
        }

        // Add final result with display hints for consistent root notation
        // Use simplified_result if available (passed from simplifier), otherwise use last_global_after
        let final_result_expr = self.simplified_result.unwrap_or(last_global_after);

        // Check if result is a poly_result - render as text (not LaTeX) for large polynomials
        if let Some(poly_text) =
            crate::poly_store::try_render_poly_result(self.context, final_result_expr)
        {
            // Get term count for info badge
            let term_count = poly_text.matches('+').count() + 1;
            html.push_str(
                r#"        </div>
        <div class="final-result">
            <strong>🧮 Final Result</strong> <span class="poly-badge">"#,
            );
            html.push_str(&format!("Polynomial: {} terms", term_count));
            html.push_str(
                r#"</span>
            <pre class="poly-output">"#,
            );
            html.push_str(&html_escape(&poly_text));
            html.push_str(
                r#"</pre>
        </div>
"#,
            );
        } else {
            // Standard LaTeX rendering for normal expressions
            // V2.14.40: Use styled renderer for consistent root notation
            let empty_config = PathHighlightConfig::new();
            let final_expr = PathHighlightedLatexRenderer {
                context: self.context,
                id: final_result_expr,
                path_highlights: &empty_config,
                hints: Some(&display_hints),
                style_prefs: Some(&self.style_prefs),
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
"#,
            );
        }

        // V2.12.13: Add global requires section (inferred from input expression)
        // This ensures timeline shows the same requires as REPL
        if !self.global_requires.is_empty() {
            let requires_messages = crate::implicit_domain::render_conditions_normalized(
                self.context,
                &self.global_requires,
            );
            if !requires_messages.is_empty() {
                html.push_str(r#"        <div class="global-requires">"#);
                html.push_str("\n            <strong>ℹ️ Requires:</strong> ");
                let escaped: Vec<String> =
                    requires_messages.iter().map(|s| html_escape(s)).collect();
                html.push_str(&escaped.join(", "));
                html.push_str("\n        </div>\n");
            }
        }

        html.push_str(
            r#"    </div>
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
