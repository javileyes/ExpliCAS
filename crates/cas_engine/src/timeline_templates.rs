//! HTML/CSS templates for timeline visualization.
//!
//! This module contains the static HTML and CSS templates used by the
//! TimelineHtml generator. Extracted from timeline.rs for maintainability.

/// CSS styles for the timeline HTML visualization
pub static CSS_STYLES: &str = r#"
* {
    box-sizing: border-box;
}
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px 15px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    min-height: 100vh;
    color: #e0e0e0;
}
.container {
    background: rgba(30, 40, 60, 0.95);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
h1 {
    color: #64b5f6;
    text-align: center;
    margin-bottom: 10px;
    font-size: 1.8em;
}
.subtitle {
    text-align: center;
    color: #90caf9;
    margin-bottom: 25px;
}
.original {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(21, 101, 192, 0.4);
}
.timeline {
    position: relative;
    padding-left: 30px;
}
.timeline::before {
    content: '';
    position: absolute;
    left: 10px;
    top: 0;
    bottom: 0;
    width: 3px;
    background: linear-gradient(to bottom, #64b5f6, #4caf50);
}
.step {
    background: rgba(40, 50, 70, 0.8);
    border-radius: 10px;
    padding: 15px 20px;
    margin-bottom: 20px;
    position: relative;
    border-left: 4px solid #64b5f6;
    transition: transform 0.2s, box-shadow 0.2s;
}
.step:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 20px rgba(100, 181, 246, 0.3);
}
.step::before {
    content: '';
    position: absolute;
    left: -23px;
    top: 20px;
    width: 12px;
    height: 12px;
    background: #64b5f6;
    border-radius: 50%;
    border: 3px solid #1a1a2e;
}
.step-number {
    color: #64b5f6;
    font-weight: bold;
    font-size: 0.9em;
    margin-bottom: 5px;
}
.step-content {
    background: rgba(30, 40, 55, 0.9);
    border-radius: 8px;
    padding: 15px 20px;
    border: 1px solid rgba(100, 181, 246, 0.2);
}
.step-description {
    font-weight: 600;
    color: #90caf9;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.importance-badge {
    font-size: 0.75em;
    padding: 2px 6px;
    border-radius: 10px;
    font-weight: bold;
    text-transform: uppercase;
}
.importance-high {
    background: rgba(76, 175, 80, 0.2);
    color: #81c784;
}
.importance-medium {
    background: rgba(255, 152, 0, 0.2);
    color: #ffb74d;
}
.importance-low {
    background: rgba(233, 30, 99, 0.2);
    color: #f48fb1;
}
.step-math {
    text-align: center;
    padding: 10px;
    background: rgba(30, 40, 55, 0.9);
    border-radius: 6px;
    overflow-x: auto;
}
.step-final {
    border-left: 4px solid #4CAF50;
    background: rgba(76, 175, 80, 0.15);
}
.step-final .step-number {
    color: #4CAF50;
}
.result-section {
    background: linear-gradient(135deg, #2e7d32, #1b5e20);
    padding: 20px;
    text-align: center;
    color: white;
    border-radius: 10px;
    margin-top: 30px;
    font-size: 1.2em;
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
}
.result-section h2 {
    color: white;
    margin-bottom: 15px;
}
@media (max-width: 600px) {
    body { padding: 10px; }
    .container { padding: 15px; }
    .step { padding: 10px 15px; }
    .step-number { font-size: 0.8em; }
}
.hidden-steps-note {
    text-align: center;
    color: #90caf9;
    font-style: italic;
    margin: 20px 0;
    padding: 10px;
    background: rgba(40, 50, 70, 0.5);
    border-radius: 6px;
}
.substeps-details {
    margin-top: 10px;
    padding: 10px;
    background: rgba(255, 152, 0, 0.1);
    border: 1px solid rgba(255, 152, 0, 0.3);
    border-radius: 8px;
    font-size: 0.95em;
}
.substeps-details summary {
    cursor: pointer;
    font-weight: bold;
    color: #ffb74d;
    padding: 5px 0;
}
.substeps-details summary:hover {
    color: #ffa726;
}
.substeps-content {
    margin-top: 10px;
    padding: 10px;
    background: rgba(30, 40, 55, 0.9);
    border-radius: 6px;
}
.substep {
    padding: 8px 0;
    border-bottom: 1px dashed rgba(255, 152, 0, 0.2);
}
.substep:last-child {
    border-bottom: none;
}
.substep-desc {
    font-weight: 500;
    color: #b0bec5;
    display: block;
    margin-bottom: 5px;
}
.substep-math {
    padding: 5px 10px;
    background: rgba(30, 40, 55, 0.8);
    border-radius: 4px;
    text-align: center;
}
.domain-warning {
    margin-top: 10px;
    padding: 8px 12px;
    background: rgba(255, 193, 7, 0.15);
    border: 1px solid rgba(255, 193, 7, 0.4);
    border-radius: 6px;
    color: #ffd54f;
    font-size: 0.9em;
}
.domain-warning::before {
    content: '⚠ ';
}
"#;

/// Generate the HTML header with title
pub fn html_header(title: &str, original_latex: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAS Steps: {escaped_title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <h1>CAS Simplification Steps</h1>
        <p class="subtitle">Step-by-step visualization</p>
        <div class="original">
            \(\textbf{{Original Expression:}}\)
            \[{original}\]
        </div>
"#,
        escaped_title = crate::timeline::html_escape(title),
        css = CSS_STYLES,
        original = crate::timeline::latex_escape(original_latex)
    )
}

/// HTML footer closing tags
pub static HTML_FOOTER: &str = r#"
    </div>
</body>
</html>
"#;

/// Generate a step HTML block
pub fn step_html(
    step_number: usize,
    description: &str,
    importance_class: &str,
    importance_label: &str,
    math_content: &str,
    is_final: bool,
    domain_assumption: Option<&str>,
) -> String {
    let final_class = if is_final { " step-final" } else { "" };
    let domain_html = if let Some(assumption) = domain_assumption {
        format!(
            r#"        <div class="domain-warning">Domain: {}</div>
"#,
            crate::timeline::html_escape(assumption)
        )
    } else {
        String::new()
    };
    format!(
        r#"<div class="step{final_class}">
    <div class="step-number">{step_number}</div>
    <div class="step-content">
        <div class="step-description">
            <span class="importance-badge {importance_class}">{importance_label}</span>
            {description}
        </div>
        <div class="step-math">\[{math_content}\]</div>
{domain_html}    </div>
</div>
"#,
        final_class = final_class,
        step_number = step_number,
        importance_class = importance_class,
        importance_label = importance_label,
        description = crate::timeline::html_escape(description),
        math_content = math_content,
        domain_html = domain_html
    )
}

/// Generate the result section HTML
pub fn result_section_html(result_latex: &str) -> String {
    format!(
        r#"<div class="result-section">
    <h2>✓ Final Result</h2>
    \[{result}\]
</div>
"#,
        result = result_latex
    )
}

/// Generate hidden steps note
pub fn hidden_steps_note(hidden_count: usize, total_count: usize) -> String {
    format!(
        r#"<div class="hidden-steps-note">
    Showing simplified view. {hidden} of {total} steps hidden.
</div>
"#,
        hidden = hidden_count,
        total = total_count
    )
}
