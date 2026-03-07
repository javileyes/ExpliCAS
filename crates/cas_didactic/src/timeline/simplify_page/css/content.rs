pub(super) const CONTENT_CSS: &str = r#"
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
"#;
