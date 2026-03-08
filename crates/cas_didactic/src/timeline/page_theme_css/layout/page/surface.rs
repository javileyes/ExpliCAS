pub(super) const SURFACE_CSS: &str = r#"
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 95%;
            margin: 0 auto;
            padding: 20px 10px;
            background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
            min-height: 100vh;
            color: var(--text-color);
            transition: background 0.3s ease;
        }
        .container {
            background: var(--container-bg);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px var(--container-shadow);
            transition: background 0.3s ease;
        }
"#;
