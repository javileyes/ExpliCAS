pub(super) fn replace_last_sqrt(value: &str) -> Option<String> {
    let start = value.rfind("\\sqrt{")?;
    let rest = &value[start + 5..];
    let (radicand, radicand_end) = find_balanced_braces(rest)?;
    let total_end = start + 5 + radicand_end + 1;
    let replacement = format!("sqrt({})", radicand);
    Some(format!(
        "{}{}{}",
        &value[..start],
        replacement,
        &value[total_end..]
    ))
}

/// `\sqrt[n]{radicando}` -> `root(radicando, n)` (y `sqrt(radicando)` para n = 2).
///
/// Sin este paso la raíz con índice no casaba con `\sqrt{`, sobrevivía intacta
/// hasta el borrado ciego de llaves del final y salía como `sqrt[3]x + 1`: se
/// LEE como ∛x + 1 —otra expresión— y además no vuelve a parsear, así que el
/// botón de copiar de un paso producía texto que la propia app rechaza. La
/// forma elegida es la que el usuario ya escribe (`root(x, 3)`), no un prefijo
/// tipo `3√`, que el parser leería como 3 · √(…).
pub(super) fn replace_last_indexed_sqrt(value: &str) -> Option<String> {
    const PREFIX: &str = "\\sqrt[";
    let start = value.rfind(PREFIX)?;
    let index_start = start + PREFIX.len();
    let index_len = value[index_start..].find(']')?;
    let index = value[index_start..index_start + index_len]
        .trim()
        .to_string();

    let rest = &value[index_start + index_len + 1..];
    if !rest.starts_with('{') {
        return None;
    }
    let (radicand, radicand_end) = find_balanced_braces(rest)?;
    let total_end = index_start + index_len + 1 + radicand_end + 1;

    let replacement = if index == "2" {
        format!("sqrt({})", radicand)
    } else {
        format!("root({}, {})", radicand, index)
    };
    Some(format!(
        "{}{}{}",
        &value[..start],
        replacement,
        &value[total_end..]
    ))
}

fn find_balanced_braces(s: &str) -> Option<(String, usize)> {
    let mut depth = 0usize;
    let mut content = String::new();

    for (i, c) in s.char_indices() {
        match c {
            '{' => {
                if depth > 0 {
                    content.push(c);
                }
                depth += 1;
            }
            '}' => {
                depth = depth.checked_sub(1)?;
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
