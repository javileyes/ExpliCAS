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
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px 15px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}
.container {
    background: white;
    border-radius: 12px;
    padding: 30px 25px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}
h1 {
    text-align: center;
    color: #333;
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}
.original {
    text-align: center;
    font-size: 1.2em;
    padding: 15px;
    background: #f0f4ff;
    border-radius: 8px;
    margin-bottom: 30px;
    border: 2px solid #667eea;
}
.timeline {
    position: relative;
    padding: 20px 0;
}
.timeline::before {
    content: '';
    position: absolute;
    left: 30px;
    top: 0;
    bottom: 0;
    width: 3px;
    background: linear-gradient(to bottom, #667eea, #764ba2);
}
.step {
    position: relative;
    margin-bottom: 30px;
    padding-left: 80px;
    animation: fadeIn 0.5s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.step-number {
    position: absolute;
    left: 10px;
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 1.1em;
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4);
}
.step-content {
    background: #fafafa;
    border-radius: 8px;
    padding: 15px 20px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.step-description {
    font-weight: 600;
    color: #667eea;
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
    background: #e8f5e9;
    color: #2e7d32;
}
.importance-medium {
    background: #fff3e0;
    color: #ef6c00;
}
.importance-low {
    background: #fce4ec;
    color: #c2185b;
}
.step-math {
    text-align: center;
    padding: 10px;
    background: white;
    border-radius: 6px;
    overflow-x: auto;
}
.step-final {
    border: 2px solid #4CAF50;
    background: #e8f5e9;
}
.step-final .step-number {
    background: linear-gradient(135deg, #4CAF50, #2E7D32);
}
.result-section {
    text-align: center;
    margin-top: 40px;
    padding: 25px;
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-radius: 12px;
    border: 2px solid #4CAF50;
}
.result-section h2 {
    color: #2e7d32;
    margin-bottom: 15px;
}
@media (max-width: 600px) {
    body { padding: 10px; }
    .container { padding: 15px; }
    .step { padding-left: 60px; }
    .step-number { left: 5px; width: 35px; height: 35px; font-size: 0.9em; }
    .timeline::before { left: 22px; }
}
.hidden-steps-note {
    text-align: center;
    color: #888;
    font-style: italic;
    margin: 20px 0;
    padding: 10px;
    background: #f5f5f5;
    border-radius: 6px;
}
.substeps-details {
    margin-top: 10px;
    padding: 10px;
    background: #fff8e1;
    border: 1px solid #ffcc80;
    border-radius: 8px;
    font-size: 0.95em;
}
.substeps-details summary {
    cursor: pointer;
    font-weight: bold;
    color: #ef6c00;
    padding: 5px 0;
}
.substeps-details summary:hover {
    color: #e65100;
}
.substeps-content {
    margin-top: 10px;
    padding: 10px;
    background: white;
    border-radius: 6px;
}
.substep {
    padding: 8px 0;
    border-bottom: 1px dashed #ffe0b2;
}
.substep:last-child {
    border-bottom: none;
}
.substep-desc {
    font-weight: 500;
    color: #795548;
    display: block;
    margin-bottom: 5px;
}
.substep-math {
    padding: 5px 10px;
    background: #fafafa;
    border-radius: 4px;
    text-align: center;
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
) -> String {
    let final_class = if is_final { " step-final" } else { "" };
    format!(
        r#"<div class="step{final_class}">
    <div class="step-number">{step_number}</div>
    <div class="step-content">
        <div class="step-description">
            <span class="importance-badge {importance_class}">{importance_label}</span>
            {description}
        </div>
        <div class="step-math">\[{math_content}\]</div>
    </div>
</div>
"#,
        final_class = final_class,
        step_number = step_number,
        importance_class = importance_class,
        importance_label = importance_label,
        description = crate::timeline::html_escape(description),
        math_content = math_content
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
