pub(super) const THEME_TOGGLE_SCRIPT: &str = r#"<script>
        function toggleTheme() {
            document.documentElement.classList.toggle('light');
            localStorage.setItem('theme', document.documentElement.classList.contains('light') ? 'light' : 'dark');
        }
        if (localStorage.getItem('theme') === 'light') {
            document.documentElement.classList.add('light');
            document.getElementById('themeToggle').checked = true;
        }
    </script>"#;
