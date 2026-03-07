pub(super) const SUBSTEPS_CSS: &str = r#"
    .substeps-toggle {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 12px;
        padding: 8px 12px;
        background: rgba(100, 181, 246, 0.15);
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.9em;
        color: var(--step-number-color);
        border: 1px solid rgba(100, 181, 246, 0.3);
        transition: all 0.2s ease;
    }
    .substeps-toggle:hover {
        background: rgba(100, 181, 246, 0.25);
    }
    .substeps-toggle .arrow {
        transition: transform 0.3s ease;
        font-size: 0.8em;
    }
    .substeps-toggle.expanded .arrow {
        transform: rotate(90deg);
    }
    .substeps-container {
        display: none;
        margin-top: 12px;
        padding-left: 20px;
        border-left: 2px solid rgba(100, 181, 246, 0.3);
    }
    .substeps-container.visible {
        display: block;
        animation: slideDown 0.3s ease;
    }
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
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
