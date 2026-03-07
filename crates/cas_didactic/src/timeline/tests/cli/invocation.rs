use super::super::super::*;

#[test]
fn extract_timeline_invocation_input_strips_prefix() {
    assert_eq!(extract_timeline_invocation_input("timeline x+1"), "x+1");
    assert_eq!(
        extract_timeline_invocation_input("timeline solve x+1=2,x"),
        "solve x+1=2,x"
    );
    assert_eq!(extract_timeline_invocation_input("x+1"), "x+1");
}
