pub(super) const CONTENT_MATH_CSS: &str = r#"
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
"#;
