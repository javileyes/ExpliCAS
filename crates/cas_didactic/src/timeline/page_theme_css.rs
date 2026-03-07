pub(super) const COMMON_TIMELINE_PAGE_CSS: &str = r#"
        * {
            box-sizing: border-box;
        }
        /* Theme variables */
        :root {
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
        }
        :root.light {
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
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 95%;
            margin: 0 auto;
            padding: 20px 10px;
            background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
            min-height: 100vh;
            color: var(--text-color);
            transition: background 0.3s ease;
        }
        .container {
            background: var(--container-bg);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px var(--container-shadow);
            transition: background 0.3s ease;
        }
        .theme-toggle {
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
        }
        .theme-toggle span {
            font-size: 1.2em;
        }
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #333;
            transition: 0.3s;
            border-radius: 26px;
        }
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: 0.3s;
            border-radius: 50%;
        }
        input:checked + .toggle-slider {
            background-color: #64b5f6;
        }
        input:checked + .toggle-slider:before {
            transform: translateX(24px);
        }
        h1 {
            color: var(--title-color);
            text-align: center;
            margin-bottom: 10px;
            font-size: 1.8em;
            transition: color 0.3s ease;
        }
        .subtitle {
            text-align: center;
            color: var(--subtitle-color);
            margin-bottom: 25px;
            transition: color 0.3s ease;
        }
        .original {
            background: linear-gradient(135deg, var(--original-bg-start), var(--original-bg-end));
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 15px var(--original-shadow);
            transition: background 0.3s ease;
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
            background: linear-gradient(to bottom, var(--timeline-line-start), var(--timeline-line-end));
            transition: background 0.3s ease;
        }
        .step {
            background: var(--step-bg);
            border-radius: 10px;
            padding: 15px 20px;
            margin-bottom: 20px;
            position: relative;
            border-left: 4px solid var(--step-border);
            transition: transform 0.2s, box-shadow 0.2s, background 0.3s ease;
        }
        .step:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 20px var(--step-hover-shadow);
        }
        .step::before {
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
        }
        .step-number {
            color: var(--step-number-color);
            font-weight: bold;
            font-size: 0.9em;
            margin-bottom: 5px;
            transition: color 0.3s ease;
        }
        .step-description {
            font-size: 1.1em;
            font-weight: 500;
            margin-bottom: 12px;
        }
        .rule-name {
            color: var(--rule-name-color);
            font-size: 0.9em;
            font-style: italic;
        }
        .math-block {
            background: var(--math-bg);
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            overflow-x: auto;
            border: 1px solid var(--math-border);
        }
        .math-before {
            border-left: 4px solid var(--math-before-border);
            background: var(--math-before-bg);
        }
        .math-after {
            border-left: 4px solid var(--math-after-border);
            background: var(--math-after-bg);
        }
        .local-change {
            background: var(--local-change-bg);
            border: 1px dashed var(--content-border);
        }
        .rule-box {
            margin-top: 12px;
            padding: 10px 12px;
            background: var(--rule-bg);
            color: var(--rule-color);
            border-left: 4px solid var(--rule-border);
            border-radius: 8px;
            font-size: 0.95em;
        }
        .substeps details {
            margin-top: 12px;
            background: var(--substeps-bg);
            border: 1px solid var(--substeps-border);
            border-radius: 8px;
            padding: 8px 10px;
        }
        .substeps summary {
            cursor: pointer;
            color: var(--substeps-summary-color);
            font-weight: 600;
        }
        .substeps summary:hover {
            color: var(--substeps-summary-hover);
        }
        .substeps .content {
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid var(--substep-border);
        }
        .substep {
            margin-bottom: 10px;
            padding-left: 8px;
            border-left: 3px solid var(--substep-border);
        }
        .substep-desc {
            color: var(--substep-desc-color);
            margin-bottom: 6px;
        }
        .substep-math {
            background: var(--substep-math-bg);
            padding: 8px;
            border-radius: 6px;
        }
        .warning-box {
            margin-top: 12px;
            padding: 12px;
            background: var(--warning-bg);
            border: 1px solid var(--warning-border);
            border-radius: 8px;
            color: var(--warning-color);
        }
        .final-result {
            background: linear-gradient(135deg, var(--final-bg-start), var(--final-bg-end));
            padding: 20px;
            text-align: center;
            color: white;
            border-radius: 10px;
            margin-top: 30px;
            font-size: 1.2em;
            box-shadow: 0 4px 12px var(--final-shadow);
            transition: background 0.3s ease;
        }
        footer {
            text-align: center;
            margin-top: 30px;
            color: var(--footer-color);
            font-size: 0.9em;
            transition: color 0.3s ease;
        }
"#;
