pub(super) const SUBSTEPS_DETAILS_CSS: &str = r#"
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
"#;
