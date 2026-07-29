mod frac;
mod sqrt;
mod text;

/// Convert LaTeX-like notation into plain-text form for CLI display.
pub(crate) fn latex_to_plain_text(s: &str) -> String {
    let mut result = strip_color_wrappers(s);

    result = text::strip_text_wrappers(result);

    // Convert LaTeX matrix environments to the engine's plain bracket form BEFORE the blind
    // brace/backslash stripping below, which would otherwise mangle `\begin{bmatrix}` into
    // `beginbmatrix` garbage in matrix/vector step before/after fields.
    result = convert_matrix_environments(result);

    result = convert_while_present(result, "\\frac{", frac::replace_last_fraction);
    result = convert_while_present(result, "\\sqrt{", sqrt::replace_last_sqrt);
    // La raíz CON índice tiene su propio prefijo (`\sqrt[n]{…}`): sin este paso no
    // la ve nadie y el borrado ciego de llaves de más abajo la parte en dos.
    result = convert_while_present(result, "\\sqrt[", sqrt::replace_last_indexed_sqrt);
    result = convert_while_present(result, "}^{", replace_next_parenthesized_power_base);
    result = convert_while_present(result, "^{", replace_next_grouped_exponent);

    result = result.replace("\\cdot", " · ");
    result = result.replace("\\left", "");
    result = result.replace("\\right", "");
    // Drop the \operatorname command but keep its `{name}` argument, which the brace strip below
    // then unwraps (e.g. `\operatorname{asinh}(x)` -> `asinh(x)`, not `operatornameasinh(x)`).
    result = result.replace("\\operatorname", "");
    result = result.replace('{', "");
    result = result.replace('}', "");
    result = humanize_even_literal_squares(&result);

    result.replace("\\", "")
}

/// Aplica `convert` mientras quede una ocurrencia de `marker`.
///
/// Cada aplicación con éxito consume una ocurrencia, así que el bucle termina; el
/// tope existe solo como red por si un `convert` futuro deja de progresar sin
/// devolver `None`, y se calcula a partir de las ocurrencias REALES.
///
/// El tope era 10 FIJO, y eso no es una red: es un truncamiento silencioso. Con
/// `cbrt` renderizando `\sqrt[3]{…}` (2026-07-29), `integrate(1/(x^3-2), x, 2, 3)`
/// pasó a tener una docena de raíces en un solo paso; las diez primeras se
/// convertían y **el resto lo destrozaba el borrado ciego de llaves**, dejando
/// `sqrt[3]2` —texto que no re-parsea— dentro de una expresión donde sus hermanas
/// decían `cbrt(2)`. El defecto llevaba latente desde siempre para cualquier
/// expresión con más de diez fracciones o raíces.
fn convert_while_present(
    mut value: String,
    marker: &str,
    convert: impl Fn(&str) -> Option<String>,
) -> String {
    let mut remaining = value.matches(marker).count() + 1;
    while value.contains(marker) && remaining > 0 {
        remaining -= 1;
        let Some(next) = convert(&value) else {
            break;
        };
        value = next;
    }
    value
}

/// Rewrite `\begin{bmatrix}…\end{bmatrix}` (and `pmatrix`/`vmatrix`) into the engine's plain
/// `[[a, b], [c, d]]` bracket form: rows split on `\\`, cells on `&`. Cell contents (e.g. `\frac`)
/// are left intact so the later fraction/sqrt/power passes still humanize them.
fn convert_matrix_environments(mut s: String) -> String {
    for env in ["bmatrix", "pmatrix", "vmatrix"] {
        let begin = format!("\\begin{{{env}}}");
        let end = format!("\\end{{{env}}}");
        while let Some(begin_at) = s.find(&begin) {
            let inner_start = begin_at + begin.len();
            let Some(rel_end) = s[inner_start..].find(&end) else {
                break;
            };
            let inner = &s[inner_start..inner_start + rel_end];
            let rows: Vec<String> = inner
                .split("\\\\")
                .map(|row| {
                    let cells: Vec<String> =
                        row.split('&').map(|cell| cell.trim().to_string()).collect();
                    format!("[{}]", cells.join(", "))
                })
                .filter(|row| row != "[]")
                .collect();
            // A single row is a vector/list -> flat `[a, b, c]` (matching the engine's own result
            // display, e.g. divisors); multiple rows stay nested `[[..], [..]]`.
            let replacement = if rows.len() == 1 {
                rows.into_iter().next().unwrap_or_else(|| "[]".to_string())
            } else {
                format!("[{}]", rows.join(", "))
            };
            let end_at = inner_start + rel_end + end.len();
            s.replace_range(begin_at..end_at, &replacement);
        }
    }
    s
}

fn strip_color_wrappers(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let chars: Vec<(usize, char)> = input.char_indices().collect();
    let mut index = 0usize;

    while index < chars.len() {
        let byte_index = chars[index].0;
        if let Some((content, consumed)) = parse_color_wrapper(&input[byte_index..]) {
            out.push_str(&strip_color_wrappers(&content));
            let target = byte_index + consumed;
            while index < chars.len() && chars[index].0 < target {
                index += 1;
            }
            continue;
        }

        out.push(chars[index].1);
        index += 1;
    }

    out
}

fn parse_color_wrapper(input: &str) -> Option<(String, usize)> {
    const RED_PREFIX: &str = "{\\color{red}{";
    const GREEN_PREFIX: &str = "{\\color{green}{";

    let prefix = if input.starts_with(RED_PREFIX) {
        RED_PREFIX
    } else if input.starts_with(GREEN_PREFIX) {
        GREEN_PREFIX
    } else {
        return None;
    };

    let remainder = &input[prefix.len()..];
    let mut depth = 1usize;

    for (offset, ch) in remainder.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    let content = remainder[..offset].to_string();
                    let outer_close = remainder[offset + 1..].chars().next()?;
                    if outer_close != '}' {
                        return None;
                    }
                    return Some((content, prefix.len() + offset + 2));
                }
            }
            _ => {}
        }
    }

    None
}

fn humanize_even_literal_squares(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let chars: Vec<(usize, char)> = input.char_indices().collect();
    let mut index = 0usize;

    while index < chars.len() {
        let byte_index = chars[index].0;
        if let Some((replacement, consumed)) = parse_even_literal_square(&input[byte_index..]) {
            out.push_str(&replacement);
            let target = byte_index + consumed;
            while index < chars.len() && chars[index].0 < target {
                index += 1;
            }
            continue;
        }

        out.push(chars[index].1);
        index += 1;
    }

    out
}

fn parse_even_literal_square(input: &str) -> Option<(String, usize)> {
    let bytes = input.as_bytes();
    let mut index = 0usize;
    let mut open_parens = 0usize;

    while index < bytes.len() && bytes[index] == b'(' && open_parens < 2 {
        open_parens += 1;
        index += 1;
    }
    if !(1..=2).contains(&open_parens) || *bytes.get(index)? != b'-' {
        return None;
    }
    index += 1;

    let digits_start = index;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    if index == digits_start {
        return None;
    }

    for _ in 0..open_parens {
        if *bytes.get(index)? != b')' {
            return None;
        }
        index += 1;
    }

    if bytes.get(index..index + 2) != Some(b"^2".as_slice()) {
        return None;
    }
    index += 2;

    Some((
        format!("{}^2", &input[digits_start..index - (open_parens + 2)]),
        index,
    ))
}

fn replace_next_parenthesized_power_base(value: &str) -> Option<String> {
    let mut open_braces = Vec::new();
    let chars: Vec<(usize, char)> = value.char_indices().collect();
    let mut index = 0usize;

    while index < chars.len() {
        let (byte_index, ch) = chars[index];
        match ch {
            '{' => open_braces.push(byte_index),
            '}' => {
                let Some(base_start) = open_braces.pop() else {
                    index += 1;
                    continue;
                };
                let after_close = &value[byte_index + 1..];
                if !after_close.starts_with("^{") {
                    index += 1;
                    continue;
                }

                let exponent_rest = &value[byte_index + 2..];
                let (_, exponent_end) = find_balanced_braces(exponent_rest)?;

                let base = &value[base_start + 1..byte_index];
                if !needs_parenthesized_power_base(base) {
                    index += 1;
                    continue;
                }

                let exponent_close = byte_index + 2 + exponent_end;
                let replacement = if has_single_outer_parentheses(base) {
                    format!("{base}^{}", &value[byte_index + 2..=exponent_close])
                } else {
                    format!("({base})^{}", &value[byte_index + 2..=exponent_close])
                };
                return Some(format!(
                    "{}{}{}",
                    &value[..base_start],
                    replacement,
                    &value[exponent_close + 1..]
                ));
            }
            _ => {}
        }
        index += 1;
    }

    None
}

fn replace_next_grouped_exponent(value: &str) -> Option<String> {
    let mut search_start = 0usize;

    while let Some(relative_start) = value[search_start..].find("^{") {
        let start = search_start + relative_start;
        let exponent_rest = &value[start + 1..];
        let (exponent, exponent_end) = find_balanced_braces(exponent_rest)?;
        let exponent_close = start + 1 + exponent_end;

        if needs_parenthesized_exponent(&exponent) {
            let replacement = format!("^({exponent})");
            return Some(format!(
                "{}{}{}",
                &value[..start],
                replacement,
                &value[exponent_close + 1..]
            ));
        }

        search_start = exponent_close + 1;
    }

    None
}

fn needs_parenthesized_power_base(base: &str) -> bool {
    base.chars()
        .any(|ch| matches!(ch, ' ' | '+' | '-' | '·' | '/'))
}

fn has_single_outer_parentheses(value: &str) -> bool {
    let Some(inner) = value
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
    else {
        return false;
    };
    if inner.is_empty() {
        return false;
    }

    let mut depth = 0usize;
    for (index, ch) in value.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = match depth.checked_sub(1) {
                    Some(depth) => depth,
                    None => return false,
                };
                if depth == 0 && index + ch.len_utf8() != value.len() {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

fn needs_parenthesized_exponent(exponent: &str) -> bool {
    exponent
        .chars()
        .any(|ch| matches!(ch, ' ' | '+' | '-' | '·' | '/' | '^'))
}

fn find_balanced_braces(s: &str) -> Option<(String, usize)> {
    let mut depth = 0usize;
    let mut content = String::new();

    for (i, ch) in s.char_indices() {
        match ch {
            '{' => {
                if depth > 0 {
                    content.push(ch);
                }
                depth += 1;
            }
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some((content, i));
                }
                content.push(ch);
            }
            _ => {
                if depth > 0 {
                    content.push(ch);
                }
            }
        }
    }

    None
}
