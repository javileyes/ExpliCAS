pub(super) const STEP_LABELS_CSS: &str = r#"
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
"#;
