pub(super) const THEME_TOGGLE_SHELL_CSS: &str = r#"
        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--container-bg);
            padding: 8px 12px;
            border-radius: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .theme-toggle span {
            font-size: 1.2em;
        }
"#;
