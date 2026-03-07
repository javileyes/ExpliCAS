pub(super) const LAYOUT_CSS: &str = r#"
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
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #333;
            transition: 0.3s;
            border-radius: 26px;
        }
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: 0.3s;
            border-radius: 50%;
        }
        input:checked + .toggle-slider {
            background-color: #64b5f6;
        }
        input:checked + .toggle-slider:before {
            transform: translateX(24px);
        }
        h1 {
            color: var(--title-color);
            text-align: center;
            margin-bottom: 10px;
            font-size: 1.8em;
            transition: color 0.3s ease;
        }
        .subtitle {
            text-align: center;
            color: var(--subtitle-color);
            margin-bottom: 25px;
            transition: color 0.3s ease;
        }
        .original {
            background: linear-gradient(135deg, var(--original-bg-start), var(--original-bg-end));
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 15px var(--original-shadow);
            transition: background 0.3s ease;
        }
        .timeline {
            position: relative;
            padding-left: 30px;
        }
        .timeline::before {
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(to bottom, var(--timeline-line-start), var(--timeline-line-end));
            transition: background 0.3s ease;
        }
        footer {
            text-align: center;
            margin-top: 30px;
            color: var(--footer-color);
            font-size: 0.9em;
            transition: color 0.3s ease;
        }
"#;
