pub(super) const SUBSTEPS_CONTAINER_CSS: &str = r#"
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
"#;
