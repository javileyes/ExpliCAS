pub(super) const STEP_CSS: &str = r#"
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
        .warning-box {
            margin-top: 12px;
            padding: 12px;
            background: var(--warning-bg);
            border: 1px solid var(--warning-border);
            border-radius: 8px;
            color: var(--warning-color);
        }
"#;
