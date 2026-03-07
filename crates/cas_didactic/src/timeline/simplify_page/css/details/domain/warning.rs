pub(super) const DETAILS_DOMAIN_WARNING_CSS: &str = r#"
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
"#;
