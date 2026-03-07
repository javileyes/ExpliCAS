pub(super) const SIMPLIFY_TIMELINE_EXTRA_CSS: &str = r#"
        .step-content {
            background: var(--content-bg);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid var(--content-border);
            transition: transform 0.2s, box-shadow 0.2s, background 0.3s ease;
        }
        .step-content:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 16px var(--step-hover-shadow);
        }
        .step-content h3 {
            margin-top: 0;
            color: var(--content-h3-color);
            font-size: 1.1em;
            transition: color 0.3s ease;
        }
        .math-expr {
            padding: 12px 15px;
            background: var(--math-bg);
            border-left: 4px solid var(--math-border);
            margin: 10px 0;
            border-radius: 4px;
            font-size: 1.05em;
            transition: background 0.3s ease;
            overflow-x: auto;
            max-width: 100%;
        }
        .math-expr.before {
            border-left-color: var(--math-before-border);
            background: var(--math-before-bg);
        }
        .math-expr.after {
            border-left-color: var(--math-after-border);
            background: var(--math-after-bg);
        }
        .math-expr strong {
            color: var(--math-strong-color);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            transition: color 0.3s ease;
        }
        .rule-description {
            text-align: center;
            padding: 12px 20px;
            margin: 15px 0;
            background: var(--rule-bg);
            border-radius: 6px;
            font-size: 0.95em;
            color: var(--rule-color);
            border: 2px dashed var(--rule-border);
            transition: background 0.3s ease, color 0.3s ease;
        }
        .local-change {
            font-size: 1.1em;
            margin: 8px 0;
            padding: 10px;
            background: var(--local-change-bg);
            border-radius: 4px;
            text-align: center;
            transition: background 0.3s ease;
        }
        .rule-name {
            font-size: 0.85em;
            color: var(--rule-name-color);
            font-weight: bold;
            margin-bottom: 5px;
            transition: color 0.3s ease;
        }
        .poly-badge {
            background: rgba(255,255,255,0.25);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75em;
            margin-left: 10px;
            font-weight: normal;
        }
        .poly-output {
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
        }
        .substeps-details {
            margin: 10px 0;
            padding: 10px 15px;
            background: var(--substeps-bg);
            border: 1px solid var(--substeps-border);
            border-radius: 8px;
            font-size: 0.95em;
            transition: background 0.3s ease;
        }
        .substeps-details summary {
            cursor: pointer;
            font-weight: bold;
            color: var(--substeps-summary-color);
            padding: 5px 0;
            transition: color 0.3s ease;
        }
        .substeps-details summary:hover {
            color: var(--substeps-summary-hover);
        }
        .substeps-content {
            margin-top: 10px;
            padding: 10px;
            background: var(--substeps-content-bg);
            border-radius: 6px;
            transition: background 0.3s ease;
        }
        .substep {
            padding: 8px 0;
            border-bottom: 1px dashed var(--substep-border);
        }
        .substep:last-child {
            border-bottom: none;
        }
        .substep-desc {
            font-weight: 500;
            color: var(--substep-desc-color);
            display: block;
            margin-bottom: 5px;
            transition: color 0.3s ease;
        }
        .substep-math {
            padding: 5px 10px;
            background: var(--substep-math-bg);
            border-radius: 4px;
            text-align: center;
            transition: background 0.3s ease;
        }
        .domain-warning {
            margin-top: 10px;
            padding: 8px 12px;
            background: var(--warning-bg);
            border: 1px solid var(--warning-border);
            border-radius: 6px;
            color: var(--warning-color);
            font-size: 0.9em;
            transition: background 0.3s ease, color 0.3s ease;
        }
        .domain-warning::before {
            content: '⚠ ';
        }
        .domain-requires {
            margin-top: 10px;
            padding: 8px 12px;
            background: rgba(33, 150, 243, 0.15);
            border: 1px solid rgba(33, 150, 243, 0.4);
            border-radius: 6px;
            color: #64b5f6;
            font-size: 0.9em;
            transition: background 0.3s ease, color 0.3s ease;
        }
        .domain-requires::before {
            content: 'ℹ️ ';
        }
        .global-requires {
            margin-top: 15px;
            padding: 12px 16px;
            background: rgba(33, 150, 243, 0.1);
            border: 2px solid rgba(33, 150, 243, 0.5);
            border-radius: 8px;
            color: #90caf9;
            font-size: 1em;
        }
        .global-requires strong {
            color: #64b5f6;
        }
"#;
