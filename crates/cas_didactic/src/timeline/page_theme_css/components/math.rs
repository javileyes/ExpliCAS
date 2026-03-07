pub(super) const MATH_CSS: &str = r#"
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
"#;
