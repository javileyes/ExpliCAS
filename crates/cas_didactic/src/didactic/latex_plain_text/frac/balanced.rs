pub(super) fn find_balanced_braces(s: &str) -> Option<(String, usize)> {
    let mut depth = 0;
    let mut content = String::new();
    for (i, c) in s.chars().enumerate() {
        match c {
            '{' => {
                if depth > 0 {
                    content.push(c);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((content, i));
                }
                content.push(c);
            }
            _ => {
                if depth > 0 {
                    content.push(c);
                }
            }
        }
    }
    None
}
