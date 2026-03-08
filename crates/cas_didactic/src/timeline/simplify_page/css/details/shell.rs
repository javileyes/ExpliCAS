pub(super) const DETAILS_SHELL_CSS: &str = r#"
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
"#;
