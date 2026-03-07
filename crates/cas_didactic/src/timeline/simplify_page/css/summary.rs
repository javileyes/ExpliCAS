pub(super) const SUMMARY_CSS: &str = r#"
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
