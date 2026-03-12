const SUBSTITUTE_USAGE_MESSAGE: &str = "Usage: subst <expr>, <target>, <replacement>\n\n\
                     Examples:\n\
                       subst x^2 + x, x, 3              -> 12\n\
                       subst x^4 + x^2 + 1, x^2, y      -> y^2 + y + 1\n\
                       subst x^3, x^2, y                -> y*x";

pub(crate) fn substitute_usage_message() -> &'static str {
    SUBSTITUTE_USAGE_MESSAGE
}

pub(crate) fn split_by_comma_ignoring_parens(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut balance = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => balance += 1,
            ')' | ']' | '}' => balance -= 1,
            ',' if balance == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    parts.push(&s[start..]);
    parts
}
