pub(super) const SUBSTEP_ITEM_CSS: &str = r#"
    .substep {
        background: rgba(30, 40, 55, 0.6);
        border-radius: 8px;
        padding: 12px 15px;
        margin-bottom: 10px;
        border-left: 3px solid #90caf9;
    }
    .substep-number {
        color: #90caf9;
        font-weight: bold;
        font-size: 0.85em;
        margin-bottom: 4px;
    }
    .substep-description {
        color: var(--description-color);
        font-size: 0.9em;
        margin-bottom: 8px;
        font-style: italic;
    }
    .substep-equation {
        background: rgba(20, 30, 45, 0.8);
        padding: 10px;
        border-radius: 6px;
        text-align: center;
        font-size: 1em;
    }
"#;
