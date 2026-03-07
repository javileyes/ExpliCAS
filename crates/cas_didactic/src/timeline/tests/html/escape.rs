use super::super::super::*;

#[test]
fn test_html_escape() {
    assert_eq!(html_escape("<script>"), "&lt;script&gt;");
    assert_eq!(html_escape("x & y"), "x &amp; y");
}
