pub(super) const DETAILS_CSS: &str = r#"
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
"#;
