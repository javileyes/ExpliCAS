use cas_ast::{Constant, Context, Equation, Expr, ExprId, RelOp};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::multispace0,
    combinator::map,
    multi::{fold_many0, separated_list0},
    sequence::{delimited, pair, preceded},
    IResult,
};
use num_rational::BigRational;

use num_bigint::BigInt;

// ============================================================================
// Unicode Math Helpers
// ============================================================================

/// Convert a superscript digit character to its numeric value
/// Returns None if the character is not a superscript digit
fn superscript_to_digit(c: char) -> Option<u32> {
    match c {
        '⁰' => Some(0),
        '¹' => Some(1),
        '²' => Some(2),
        '³' => Some(3),
        '⁴' => Some(4),
        '⁵' => Some(5),
        '⁶' => Some(6),
        '⁷' => Some(7),
        '⁸' => Some(8),
        '⁹' => Some(9),
        _ => None,
    }
}

/// Parse a sequence of superscript digits into a number
/// Returns the number and the remaining string
fn parse_superscript_number(input: &str) -> Option<(u64, &str)> {
    let mut chars = input.chars().peekable();
    let mut value: u64 = 0;
    let mut count = 0;
    let mut byte_len = 0;

    while let Some(&c) = chars.peek() {
        if let Some(digit) = superscript_to_digit(c) {
            value = value * 10 + digit as u64;
            count += 1;
            byte_len += c.len_utf8();
            chars.next();
        } else {
            break;
        }
    }

    if count > 0 {
        Some((value, &input[byte_len..]))
    } else {
        None
    }
}

/// Get the root index from a Unicode root symbol
/// Returns (index, remaining_input) or None
fn parse_unicode_root_prefix(input: &str) -> Option<(u64, &str)> {
    // Check for direct Unicode root symbols
    if input.starts_with('∛') {
        return Some((3, &input['∛'.len_utf8()..]));
    }
    if input.starts_with('∜') {
        return Some((4, &input['∜'.len_utf8()..]));
    }
    if input.starts_with('√') {
        return Some((2, &input['√'.len_utf8()..]));
    }

    // Check for superscript number followed by √ (e.g., ³√, ⁴√, ⁵√)
    if let Some((index, after_num)) = parse_superscript_number(input) {
        if after_num.starts_with('√') {
            return Some((index, &after_num['√'.len_utf8()..]));
        }
    }

    None
}

// Intermediate AST for parsing
#[derive(Debug, Clone)]
enum ParseNode {
    Number(BigRational),
    Constant(Constant),
    Variable(String),
    Add(Box<ParseNode>, Box<ParseNode>),
    Sub(Box<ParseNode>, Box<ParseNode>),
    Mul(Box<ParseNode>, Box<ParseNode>),
    Div(Box<ParseNode>, Box<ParseNode>),
    Pow(Box<ParseNode>, Box<ParseNode>),
    Neg(Box<ParseNode>),
    Function(String, Vec<ParseNode>),
    Matrix(Vec<Vec<ParseNode>>), // 2D structure for validation during parsing
    SessionRef(u64),             // Reference to session history #id
}

impl ParseNode {
    fn lower(self, ctx: &mut Context) -> ExprId {
        match self {
            ParseNode::Number(n) => ctx.add(Expr::Number(n)),
            ParseNode::Constant(c) => ctx.add(Expr::Constant(c)),
            ParseNode::Variable(s) => ctx.add(Expr::Variable(s)),
            ParseNode::Add(l, r) => {
                let lid = l.lower(ctx);
                let rid = r.lower(ctx);
                ctx.add(Expr::Add(lid, rid))
            }
            ParseNode::Sub(l, r) => {
                let lid = l.lower(ctx);
                let rid = r.lower(ctx);
                ctx.add(Expr::Sub(lid, rid))
            }
            ParseNode::Mul(l, r) => {
                let lid = l.lower(ctx);
                let rid = r.lower(ctx);
                ctx.add(Expr::Mul(lid, rid))
            }
            ParseNode::Div(l, r) => {
                let lid = l.lower(ctx);
                let rid = r.lower(ctx);
                ctx.add(Expr::Div(lid, rid))
            }
            ParseNode::Pow(b, e) => {
                let bid = b.lower(ctx);
                let eid = e.lower(ctx);
                ctx.add(Expr::Pow(bid, eid))
            }
            ParseNode::Neg(e) => {
                let eid = e.lower(ctx);
                ctx.add(Expr::Neg(eid))
            }
            ParseNode::Function(name, args) => {
                let arg_ids = args.into_iter().map(|a| a.lower(ctx)).collect();
                ctx.add(Expr::Function(name, arg_ids))
            }
            ParseNode::Matrix(rows) => {
                // Flatten 2D structure to 1D for storage
                let num_rows = rows.len();
                let num_cols = if num_rows > 0 { rows[0].len() } else { 0 };

                // Collect all elements in row-major order
                let mut data = Vec::new();
                for row in rows {
                    for elem in row {
                        data.push(elem.lower(ctx));
                    }
                }

                ctx.matrix(num_rows, num_cols, data)
            }
            ParseNode::SessionRef(id) => ctx.add(Expr::SessionRef(id)),
        }
    }
}

/// Convert a decimal string to BigRational.
/// Supports: "8.2" → 41/5, ".5" → 1/2, "8." → 8, "123" → 123
/// Algorithm: For "A.B", num = A*10^k + B, den = 10^k (where k = len(B))
fn decimal_to_rational(integer_part: &str, fractional_part: &str) -> BigRational {
    let k = fractional_part.len();

    if k == 0 {
        // No fractional part: just an integer
        let n: BigInt = integer_part.parse().unwrap_or_else(|_| BigInt::from(0));
        return BigRational::from_integer(n);
    }

    // Calculate 10^k
    let ten = BigInt::from(10);
    let mut denominator = BigInt::from(1);
    for _ in 0..k {
        denominator *= &ten;
    }

    // Parse integer part (may be empty for ".5")
    let int_val: BigInt = if integer_part.is_empty() {
        BigInt::from(0)
    } else {
        integer_part.parse().unwrap_or_else(|_| BigInt::from(0))
    };

    // Parse fractional part
    let frac_val: BigInt = fractional_part.parse().unwrap_or_else(|_| BigInt::from(0));

    // numerator = integer_part * 10^k + fractional_part
    let numerator = int_val * &denominator + frac_val;

    // BigRational::new automatically reduces the fraction (gcd)
    BigRational::new(numerator, denominator)
}

// Parser for numeric literals (integers and decimals)
// Supports: 123, 8.2, .5, 8.
fn parse_number(input: &str) -> IResult<&str, ParseNode> {
    // Try to match: [digits] "." [digits] OR just digits
    // Pattern 1: digits "." [digits] (e.g., "8.2", "8.")
    // Pattern 2: "." digits (e.g., ".5")
    // Pattern 3: just digits (e.g., "123")

    use nom::bytes::complete::take_while;
    use nom::combinator::opt;
    use nom::sequence::pair;

    fn is_digit(c: char) -> bool {
        c.is_ascii_digit()
    }

    // Parse optional integer part, then optional (dot + fractional part)
    let (remaining, (int_part, maybe_frac)) = pair(
        take_while(is_digit),
        opt(pair(tag("."), take_while(is_digit))),
    )(input)?;

    let (int_str, frac_str) = match maybe_frac {
        Some((_, frac)) => (int_part, frac),
        None => (int_part, ""),
    };

    // Must have at least some digits somewhere
    if int_str.is_empty() && frac_str.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Digit,
        )));
    }

    // Edge case: just "." with no digits is not a valid number
    if int_str.is_empty() && maybe_frac.is_some() && frac_str.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Digit,
        )));
    }

    let rational = decimal_to_rational(int_str, frac_str);
    Ok((remaining, ParseNode::Number(rational)))
}

// Parser for session references (#1, #42, etc.)
// Syntax: # followed by one or more digits
fn parse_session_ref(input: &str) -> IResult<&str, ParseNode> {
    // Must start with '#'
    if !input.starts_with('#') {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    let after_hash = &input[1..];

    // Collect digits
    let digit_count = after_hash
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .count();

    if digit_count == 0 {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Digit,
        )));
    }

    let digit_str = &after_hash[..digit_count];
    let remaining = &after_hash[digit_count..];

    // Parse the id
    let id: u64 = digit_str.parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit))
    })?;

    Ok((remaining, ParseNode::SessionRef(id)))
}

// Parser for constants with word boundary check
// 'e' and 'pi' should not match prefixes of longer identifiers (e.g., 'exact', 'pivot')
fn parse_constant(input: &str) -> IResult<&str, ParseNode> {
    // Helper: check if next char is alphanumeric (would indicate identifier, not constant)
    fn is_word_boundary(remaining: &str) -> bool {
        remaining
            .chars()
            .next()
            .map_or(true, |c| !c.is_ascii_alphanumeric() && c != '_')
    }

    // Try 'pi' first (longer prefix)
    if input.starts_with("pi") && is_word_boundary(&input[2..]) {
        return Ok((&input[2..], ParseNode::Constant(Constant::Pi)));
    }

    // Try 'e' (must not be followed by alphanumeric)
    if input.starts_with('e') && is_word_boundary(&input[1..]) {
        return Ok((&input[1..], ParseNode::Constant(Constant::E)));
    }

    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Tag,
    )))
}

// Parser for identifiers (variable/function names)
// Identifiers start with letter or underscore, then allow letters, digits, underscores
// Examples: x, x1, theta3, _tmp, x_1
fn parse_identifier(input: &str) -> IResult<&str, &str> {
    // Check first char is valid start (letter or underscore)
    let mut chars = input.chars();
    let first = chars.next();
    if !matches!(first, Some(c) if c.is_ascii_alphabetic() || c == '_') {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Alpha,
        )));
    }

    // Count valid continuation chars (alphanumeric or underscore)
    let mut len = first.unwrap().len_utf8();
    for c in chars {
        if c.is_ascii_alphanumeric() || c == '_' {
            len += c.len_utf8();
        } else {
            break;
        }
    }

    Ok((&input[len..], &input[..len]))
}

// Parser for variables
// Note: "i" (imaginary unit) is recognized as Constant::I, not a variable
fn parse_variable(input: &str) -> IResult<&str, ParseNode> {
    map(parse_identifier, |s: &str| {
        // Recognize imaginary unit as constant
        if s == "i" {
            ParseNode::Constant(Constant::I)
        } else {
            ParseNode::Variable(s.to_string())
        }
    })(input)
}

// Parser for parentheses
fn parse_parens(input: &str) -> IResult<&str, ParseNode> {
    delimited(
        preceded(multispace0, tag("(")),
        parse_expr,
        preceded(multispace0, tag(")")),
    )(input)
}

// Parser for function calls
fn parse_function(input: &str) -> IResult<&str, ParseNode> {
    let (input, name) = parse_identifier(input)?;
    let (input, _) = preceded(multispace0, tag("("))(input)?;
    let (input, args) = separated_list0(preceded(multispace0, tag(",")), parse_expr)(input)?;
    let (input, _) = preceded(multispace0, tag(")"))(input)?;

    if name == "ln" && args.len() == 1 {
        // ln(x) -> log(e, x)
        return Ok((
            input,
            ParseNode::Function(
                "log".to_string(),
                vec![ParseNode::Constant(Constant::E), args[0].clone()],
            ),
        ));
    }

    if name == "exp" && args.len() == 1 {
        // exp(x) -> e^x
        return Ok((
            input,
            ParseNode::Pow(
                Box::new(ParseNode::Constant(Constant::E)),
                Box::new(args[0].clone()),
            ),
        ));
    }

    Ok((input, ParseNode::Function(name.to_string(), args)))
}

fn parse_abs(input: &str) -> IResult<&str, ParseNode> {
    delimited(
        preceded(multispace0, tag("|")),
        parse_expr,
        preceded(multispace0, tag("|")),
    )(input)
    .map(|(next_input, expr)| {
        (
            next_input,
            ParseNode::Function("abs".to_string(), vec![expr]),
        )
    })
}

// Parser for matrices and vectors
// Matrices: [[a, b], [c, d]]
// Vectors: [x, y, z] (default: column vector, nx1)
fn parse_matrix(input: &str) -> IResult<&str, ParseNode> {
    let (input, _) = preceded(multispace0, tag("["))(input)?;

    // Try to parse first element
    let (input, first_elem) = preceded(
        multispace0,
        alt((
            // Nested array for multi-row matrix
            |inp| {
                let (inp, _) = tag("[")(inp)?;
                let (inp, row) = separated_list0(preceded(multispace0, tag(",")), parse_expr)(inp)?;
                let (inp, _) = preceded(multispace0, tag("]"))(inp)?;
                Ok((inp, ParseNode::Matrix(vec![row])))
            },
            // Single expression for vector
            |inp| {
                let (inp, expr) = parse_expr(inp)?;
                Ok((inp, ParseNode::Matrix(vec![vec![expr]])))
            },
        )),
    )(input)?;

    // Extract first row structure
    let first_row = match first_elem {
        ParseNode::Matrix(ref rows) => rows[0].clone(),
        _ => unreachable!(),
    };

    // Try to parse remaining rows/elements
    let (input, remaining) = fold_many0(
        preceded(
            preceded(multispace0, tag(",")),
            preceded(
                multispace0,
                alt((
                    // Nested array for matrix row
                    |inp| {
                        let (inp, _) = tag("[")(inp)?;
                        let (inp, row) =
                            separated_list0(preceded(multispace0, tag(",")), parse_expr)(inp)?;
                        let (inp, _) = preceded(multispace0, tag("]"))(inp)?;
                        Ok((inp, row))
                    },
                    // Single expression for vector
                    |inp| {
                        let (inp, expr) = parse_expr(inp)?;
                        Ok((inp, vec![expr]))
                    },
                )),
            ),
        ),
        Vec::new,
        |mut acc, row| {
            acc.push(row);
            acc
        },
    )(input)?;

    let (input, _) = preceded(multispace0, tag("]"))(input)?;

    // Build final matrix structure
    let mut all_rows = vec![first_row];
    all_rows.extend(remaining);

    // Validate: all rows must have same length
    let cols = all_rows[0].len();
    for row in all_rows.iter() {
        if row.len() != cols {
            // Return error via nom - inconsistent row lengths
            return Err(nom::Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Verify,
            )));
        }
    }

    Ok((input, ParseNode::Matrix(all_rows)))
}

// Parser for Unicode root symbols: √, ∛, ∜, or ⁿ√ followed by expression
// Examples: √(x), ∛8, ³√(x+1), ⁵√32
fn parse_unicode_root(input: &str) -> IResult<&str, ParseNode> {
    let input = input.trim_start();

    // Try to parse a Unicode root prefix
    let (index, after_prefix) = parse_unicode_root_prefix(input).ok_or_else(|| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag))
    })?;

    // Parse the argument (either parenthesized or simple expression)
    // Try parentheses first for expressions like √(x+1)
    let (remaining, arg) = alt((
        parse_parens,
        parse_factorial, // For simple bases like √x or ∛8
    ))(after_prefix)?;

    // Create sqrt(arg, index) function call
    let index_node = ParseNode::Number(BigRational::from_integer(BigInt::from(index)));

    Ok((
        remaining,
        ParseNode::Function("sqrt".to_string(), vec![arg, index_node]),
    ))
}

// Atom
fn parse_atom(input: &str) -> IResult<&str, ParseNode> {
    preceded(
        multispace0,
        alt((
            parse_session_ref,  // #id references - try early to avoid # confusion
            parse_unicode_root, // Unicode roots: √, ∛, ∜, ⁿ√
            parse_number,
            parse_function,
            parse_constant,
            parse_variable,
            parse_matrix, // Try matrix before parens (since [ ] syntax)
            parse_parens,
            parse_abs,
        )),
    )(input)
}

// Factorial (Postfix) - Higher precedence than power?
// Actually, usually ! binds very tightly.
// x^y! -> x^(y!)
// So parse_factorial should be called by parse_power for the base?
// No, parse_power calls parse_factorial.
// parse_factorial calls parse_atom.
// Also handles superscript exponents: x⁶ → x^6

fn parse_factorial(input: &str) -> IResult<&str, ParseNode> {
    let (input, atom) = parse_atom(input)?;

    // First handle factorials
    let (input, with_factorial) = fold_many0(
        preceded(multispace0, tag("!")),
        move || atom.clone(),
        |acc, _| ParseNode::Function("fact".to_string(), vec![acc]),
    )(input)?;

    // Then check for superscript exponents (x⁶ → x^6)
    // Note: no whitespace allowed before superscript (it's attached to the base)
    if let Some((exp_value, remaining)) = parse_superscript_number(input) {
        let exp_node = ParseNode::Number(BigRational::from_integer(BigInt::from(exp_value)));
        return Ok((
            remaining,
            ParseNode::Pow(Box::new(with_factorial), Box::new(exp_node)),
        ));
    }

    Ok((input, with_factorial))
}

// Power - right associative: 2^3^4 = 2^(3^4), not (2^3)^4
// Also allows negative exponents: x^-2, x^-(a+b)
fn parse_power(input: &str) -> IResult<&str, ParseNode> {
    let (input, base) = parse_factorial(input)?;

    // Try to parse "^" followed by exponent
    let try_caret = preceded::<_, _, _, nom::error::Error<&str>, _, _>(
        multispace0::<_, nom::error::Error<&str>>,
        tag::<_, _, nom::error::Error<&str>>("^"),
    )(input);

    if let Ok((input, _)) = try_caret {
        // Parse exponent - allow unary minus/plus, then recurse for right-associativity
        let (input, exp) = parse_power_exponent(input)?;
        Ok((input, ParseNode::Pow(Box::new(base), Box::new(exp))))
    } else {
        Ok((input, base))
    }
}

// Parser for exponents: allows sign prefix (-2, +3) then recurses for chained powers
fn parse_power_exponent(input: &str) -> IResult<&str, ParseNode> {
    preceded(
        multispace0,
        alt((
            // Case: negative exponent -expr (e.g., x^-2, x^-(a+b))
            map(pair(tag("-"), parse_power_exponent), |(_, expr)| {
                ParseNode::Neg(Box::new(expr))
            }),
            // Case: positive sign +expr (rarely used, but valid)
            map(pair(tag("+"), parse_power_exponent), |(_, expr)| expr),
            // Case: normal power expression (recurse for 2^3^4)
            parse_power,
        )),
    )(input)
}

// Unary
fn parse_unary(input: &str) -> IResult<&str, ParseNode> {
    alt((
        map(
            pair(preceded(multispace0, tag("-")), parse_unary),
            |(_, expr)| ParseNode::Neg(Box::new(expr)),
        ),
        parse_power,
    ))(input)
}

// Term - handles explicit * and / operators
// Also handles implicit multiplication: 2x → 2*x, 3(x+y) → 3*(x+y)
fn parse_term(input: &str) -> IResult<&str, ParseNode> {
    let (input, init) = parse_unary(input)?;

    // First, handle explicit operators (* · / mod)
    let (input, result) = fold_many0(
        pair(
            preceded(multispace0, alt((tag("*"), tag("·"), tag("/"), tag("mod")))),
            parse_unary,
        ),
        move || init.clone(),
        |acc, (op, val)| match op {
            "*" | "·" => ParseNode::Mul(Box::new(acc), Box::new(val)),
            "/" => ParseNode::Div(Box::new(acc), Box::new(val)),
            "mod" => ParseNode::Function("mod".to_string(), vec![acc, val]),
            _ => unreachable!(),
        },
    )(input)?;

    // Now handle implicit multiplication (number followed by variable/parentheses/function)
    // Examples: 2x, 3(x+y), 2sin(x), 2pi
    // Note: We only do implicit mul when there's NO whitespace between number and next term
    parse_implicit_mul_chain(input, result)
}

// Parse implicit multiplication chain: 2xy → 2*x*y, 2x → 2*x
fn parse_implicit_mul_chain(input: &str, acc: ParseNode) -> IResult<&str, ParseNode> {
    // Only apply implicit multiplication if:
    // 1. Previous term could end with a number (or is a complete term)
    // 2. Next character suggests a multiplicand (letter for variable, '(' for parens)

    // Check if input starts with something that could be implicitly multiplied
    let first_char = input.chars().next();

    match first_char {
        // Variable or function start: 2x, 2sin(x), 2pi
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {
            // Only if the current accumulator could allow implicit mul
            // (basically, if last token was a number, power, or factored expression)
            if can_implicit_mul(&acc) {
                if let Ok((remaining, next_factor)) = parse_unary(input) {
                    let new_acc = ParseNode::Mul(Box::new(acc), Box::new(next_factor));
                    return parse_implicit_mul_chain(remaining, new_acc);
                }
            }
            Ok((input, acc))
        }
        // Parentheses start: 2(x+y)
        Some('(') => {
            if can_implicit_mul(&acc) {
                if let Ok((remaining, next_factor)) = parse_unary(input) {
                    let new_acc = ParseNode::Mul(Box::new(acc), Box::new(next_factor));
                    return parse_implicit_mul_chain(remaining, new_acc);
                }
            }
            Ok((input, acc))
        }
        _ => Ok((input, acc)),
    }
}

// Check if a ParseNode can be followed by implicit multiplication
fn can_implicit_mul(node: &ParseNode) -> bool {
    match node {
        // Numbers can be followed by implicit mul: 2x
        ParseNode::Number(_) => true,
        // Powers can be followed: 2^2x (though unusual)
        ParseNode::Pow(_, _) => true,
        // Factorials can be followed: n!x (unusual but valid)
        ParseNode::Function(name, args) if name == "fact" && args.len() == 1 => true,
        // Parenthesized expressions: (2+3)x is more like explicit, but (2)x → 2*x is valid
        // Actually, we don't want to match this for arbitrary expressions
        // Only allow after numbers essentially
        // Mul/Div: chain continues 2*3x
        ParseNode::Mul(_, right) | ParseNode::Div(_, right) => can_implicit_mul(right),
        _ => false,
    }
}

// Expr
fn parse_expr(input: &str) -> IResult<&str, ParseNode> {
    let (input, init) = parse_term(input)?;
    fold_many0(
        pair(preceded(multispace0, alt((tag("+"), tag("-")))), parse_term),
        move || init.clone(),
        |acc, (op, val)| match op {
            "+" => ParseNode::Add(Box::new(acc), Box::new(val)),
            "-" => ParseNode::Sub(Box::new(acc), Box::new(val)),
            _ => unreachable!(),
        },
    )(input)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Expression(ExprId),
    Equation(Equation),
}

// Parser for relational operators
fn parse_relop(input: &str) -> IResult<&str, RelOp> {
    preceded(
        multispace0,
        alt((
            map(tag("="), |_| RelOp::Eq),
            map(tag("!="), |_| RelOp::Neq),
            map(tag("<="), |_| RelOp::Leq),
            map(tag(">="), |_| RelOp::Geq),
            map(tag("<"), |_| RelOp::Lt),
            map(tag(">"), |_| RelOp::Gt),
        )),
    )(input)
}

// Parser for equations
fn parse_equation(input: &str) -> IResult<&str, (ParseNode, RelOp, ParseNode)> {
    let (input, lhs) = parse_expr(input)?;
    let (input, op) = parse_relop(input)?;
    let (input, rhs) = parse_expr(input)?;
    Ok((input, (lhs, op, rhs)))
}

use crate::error::ParseError;

pub fn parse(input: &str, ctx: &mut Context) -> Result<ExprId, ParseError> {
    let (remaining, expr_node) =
        parse_expr(input).map_err(|e| ParseError::NomError(format!("{}", e)))?;

    let remaining = remaining.trim();
    if !remaining.is_empty() {
        return Err(ParseError::UnconsumedInput(remaining.to_string()));
    }

    Ok(expr_node.lower(ctx))
}

pub fn parse_statement(input: &str, ctx: &mut Context) -> Result<Statement, ParseError> {
    // Try parsing as equation first
    if let Ok((remaining, (lhs, op, rhs))) = parse_equation(input) {
        if remaining.trim().is_empty() {
            let lhs_id = lhs.lower(ctx);
            let rhs_id = rhs.lower(ctx);
            return Ok(Statement::Equation(Equation {
                lhs: lhs_id,
                rhs: rhs_id,
                op,
            }));
        }
    }

    // Fallback to expression
    match parse_expr(input) {
        Ok((remaining, expr_node)) => {
            if remaining.trim().is_empty() {
                Ok(Statement::Expression(expr_node.lower(ctx)))
            } else {
                Err(ParseError::UnconsumedInput(remaining.to_string()))
            }
        }
        Err(e) => Err(ParseError::NomError(format!("{}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::DisplayExpr;

    #[test]
    fn test_parse_number() {
        let mut ctx = Context::new();
        let e = parse("123", &mut ctx).unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: e
                }
            ),
            "123"
        );
    }

    #[test]
    fn test_parse_decimal_literals() {
        // Test various decimal formats
        let cases = [
            ("8.2", "41/5"),            // Standard decimal
            ("0.5", "1/2"),             // Leading zero
            (".5", "1/2"),              // No leading zero
            ("8.", "8"),                // Trailing dot (integer)
            ("0.125", "1/8"),           // Eighth
            ("1.25", "5/4"),            // Mixed
            ("100.001", "100001/1000"), // Many decimals
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let e = parse(input, &mut ctx).expect(&format!("Failed to parse: {}", input));
            let result = format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: e
                }
            );
            assert_eq!(
                result, expected,
                "Input {} expected {} but got {}",
                input, expected, result
            );
        }
    }

    #[test]
    fn test_parse_negative_decimal() {
        // Negative decimals
        let mut ctx = Context::new();
        let e = parse("-0.125", &mut ctx).unwrap();
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: e
            }
        );
        assert_eq!(result, "-1/8");
    }

    #[test]
    fn test_parse_variable() {
        let mut ctx = Context::new();
        let e = parse("x", &mut ctx).unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: e
                }
            ),
            "x"
        );
    }

    #[test]
    fn test_parse_arithmetic() {
        let mut ctx = Context::new();
        let e = parse("1 + 2 * x", &mut ctx).unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: e
                }
            ),
            "1 + 2 * x"
        );
    }

    #[test]
    fn test_parse_parens() {
        let mut ctx = Context::new();
        let e = parse("(1 + 2) * x", &mut ctx).unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: e
                }
            ),
            // Canonical ordering: numbers before variables in multiplication
            "x * (1 + 2)"
        );
    }

    #[test]
    fn test_parse_power() {
        let mut ctx = Context::new();
        let e = parse("x^2", &mut ctx).unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: e
                }
            ),
            "x^2"
        );

        let e2 = parse("x^2 * y", &mut ctx).unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: e2
                }
            ),
            // Canonical ordering: y (Variable, rank=2) before x^2 (Pow, rank=5)
            "y * x^2"
        );
    }

    #[test]
    fn test_parse_vector() {
        let mut ctx = Context::new();
        // Column vector (nx1)
        let e = parse("[1, 2, 3]", &mut ctx).unwrap();
        if let Expr::Matrix { rows, cols, .. } = ctx.get(e) {
            assert_eq!(*rows, 3);
            assert_eq!(*cols, 1);
        } else {
            panic!("Expected Matrix variant");
        }
    }

    #[test]
    fn test_parse_row_matrix() {
        let mut ctx = Context::new();
        // Single row matrix (1xn) - needs double brackets
        let e = parse("[[1, 2, 3]]", &mut ctx).unwrap();
        if let Expr::Matrix { rows, cols, .. } = ctx.get(e) {
            assert_eq!(*rows, 1);
            assert_eq!(*cols, 3);
        } else {
            panic!("Expected Matrix variant");
        }
    }

    #[test]
    fn test_parse_matrix_2x2() {
        let mut ctx = Context::new();
        let e = parse("[[1, 2], [3, 4]]", &mut ctx).unwrap();
        // Verify it's a matrix
        if let Expr::Matrix { rows, cols, .. } = ctx.get(e) {
            assert_eq!(*rows, 2);
            assert_eq!(*cols, 2);
        } else {
            panic!("Expected Matrix variant");
        }
    }

    #[test]
    fn test_parse_matrix_with_expressions() {
        let mut ctx = Context::new();
        let e = parse("[[x + 1, y], [2 * z, 0]]", &mut ctx).unwrap();
        if let Expr::Matrix { rows, cols, data } = ctx.get(e) {
            assert_eq!(*rows, 2);
            assert_eq!(*cols, 2);
            assert_eq!(data.len(), 4);
        } else {
            panic!("Expected Matrix variant");
        }
    }

    #[test]
    fn test_parse_vector_with_variables() {
        let mut ctx = Context::new();
        let e = parse("[x, y, z]", &mut ctx).unwrap();
        if let Expr::Matrix { rows, cols, data } = ctx.get(e) {
            assert_eq!(*rows, 3);
            assert_eq!(*cols, 1);
            assert_eq!(data.len(), 3);
        } else {
            panic!("Expected Matrix variant");
        }
    }

    #[test]
    fn test_power_right_associativity() {
        let mut ctx = Context::new();
        // 2^3^4 should be 2^(3^4) = 2^81, NOT (2^3)^4 = 4096
        let e = parse("2^3^4", &mut ctx).unwrap();
        // Verify structure: should be Pow(2, Pow(3, 4))
        if let Expr::Pow(base, exp) = ctx.get(e) {
            // base should be 2
            if let Expr::Number(n) = ctx.get(*base) {
                assert!(n.is_integer());
                assert_eq!(n.to_integer(), 2.into());
            } else {
                panic!("Expected base to be Number(2)");
            }
            // exp should be Pow(3, 4)
            if let Expr::Pow(exp_base, exp_exp) = ctx.get(*exp) {
                if let Expr::Number(n) = ctx.get(*exp_base) {
                    assert_eq!(n.to_integer(), 3.into());
                }
                if let Expr::Number(n) = ctx.get(*exp_exp) {
                    assert_eq!(n.to_integer(), 4.into());
                }
            } else {
                panic!("Expected exponent to be Pow(3, 4)");
            }
        } else {
            panic!("Expected Pow");
        }
    }

    #[test]
    fn test_negative_exponent() {
        let mut ctx = Context::new();
        // x^-2 should parse as Pow(x, -2)
        // Note: Neg(Number(2)) is canonicalized to Number(-2) by Context::add
        let e = parse("x^-2", &mut ctx).unwrap();
        if let Expr::Pow(base, exp) = ctx.get(e) {
            // base should be x
            if let Expr::Variable(v) = ctx.get(*base) {
                assert_eq!(v, "x");
            } else {
                panic!("Expected base to be Variable(x)");
            }
            // exp should be Number(-2) due to canonicalization
            if let Expr::Number(n) = ctx.get(*exp) {
                assert_eq!(n.to_integer(), (-2).into(), "Expected exponent to be -2");
            } else {
                panic!(
                    "Expected exponent to be Number(-2), got {:?}",
                    ctx.get(*exp)
                );
            }
        } else {
            panic!("Expected Pow");
        }
    }

    #[test]
    fn test_negative_exponent_expression() {
        let mut ctx = Context::new();
        // x^-(a+b) should parse as Pow(x, Neg(Add(a, b)))
        let e = parse("x^-(a+b)", &mut ctx).unwrap();
        if let Expr::Pow(_, exp) = ctx.get(e) {
            if let Expr::Neg(_) = ctx.get(*exp) {
                // Successfully parsed as Neg
            } else {
                panic!("Expected exponent to be Neg");
            }
        } else {
            panic!("Expected Pow");
        }
    }

    #[test]
    fn test_parse_alphanumeric_variables() {
        let mut ctx = Context::new();

        // x1 + x2 should parse as Add(Variable("x1"), Variable("x2"))
        let e = parse("x1 + x2", &mut ctx).unwrap();
        if let Expr::Add(l, r) = ctx.get(e) {
            if let Expr::Variable(v) = ctx.get(*l) {
                assert_eq!(v, "x1");
            } else {
                panic!("Expected Variable(x1)");
            }
            if let Expr::Variable(v) = ctx.get(*r) {
                assert_eq!(v, "x2");
            } else {
                panic!("Expected Variable(x2)");
            }
        } else {
            panic!("Expected Add");
        }
    }

    #[test]
    fn test_parse_variable_with_power() {
        let mut ctx = Context::new();

        // x1^2 should be Pow(Variable("x1"), 2), NOT x * 1^2
        let e = parse("x1^2", &mut ctx).unwrap();
        if let Expr::Pow(base, exp) = ctx.get(e) {
            if let Expr::Variable(v) = ctx.get(*base) {
                assert_eq!(v, "x1");
            } else {
                panic!("Expected base to be Variable(x1)");
            }
            if let Expr::Number(n) = ctx.get(*exp) {
                assert_eq!(n.to_integer(), 2.into());
            } else {
                panic!("Expected exponent to be Number(2)");
            }
        } else {
            panic!("Expected Pow");
        }
    }

    #[test]
    fn test_parse_underscore_variables() {
        let mut ctx = Context::new();

        // _tmp and x_1 should be valid identifiers
        let e = parse("_tmp + x_1", &mut ctx).unwrap();
        if let Expr::Add(l, r) = ctx.get(e) {
            if let Expr::Variable(v) = ctx.get(*l) {
                assert!(v == "_tmp" || v == "x_1");
            }
            if let Expr::Variable(v) = ctx.get(*r) {
                assert!(v == "_tmp" || v == "x_1");
            }
        }
    }

    #[test]
    fn test_parse_longer_alphanumeric() {
        let mut ctx = Context::new();

        // theta3 and phi123 should work
        let e = parse("theta3 * phi123", &mut ctx).unwrap();
        if let Expr::Mul(l, r) = ctx.get(e) {
            // One should be theta3, other phi123
            let l_var = if let Expr::Variable(v) = ctx.get(*l) {
                v.clone()
            } else {
                panic!("Expected Variable")
            };
            let r_var = if let Expr::Variable(v) = ctx.get(*r) {
                v.clone()
            } else {
                panic!("Expected Variable")
            };
            assert!(l_var == "theta3" || l_var == "phi123");
            assert!(r_var == "theta3" || r_var == "phi123");
        } else {
            panic!("Expected Mul");
        }
    }

    // ========== Session Reference Tests ==========

    #[test]
    fn test_parse_session_ref_simple() {
        let mut ctx = Context::new();
        let e = parse("#1", &mut ctx).unwrap();
        if let Expr::SessionRef(id) = ctx.get(e) {
            assert_eq!(*id, 1);
        } else {
            panic!("Expected SessionRef, got {:?}", ctx.get(e));
        }
    }

    #[test]
    fn test_parse_session_ref_larger_id() {
        let mut ctx = Context::new();
        let e = parse("#42", &mut ctx).unwrap();
        if let Expr::SessionRef(id) = ctx.get(e) {
            assert_eq!(*id, 42);
        } else {
            panic!("Expected SessionRef(42)");
        }
    }

    #[test]
    fn test_parse_session_ref_in_expression() {
        let mut ctx = Context::new();
        // #1 + 2
        let e = parse("#1 + 2", &mut ctx).unwrap();
        if let Expr::Add(l, r) = ctx.get(e) {
            // One of them should be SessionRef(1), other Number(2)
            let has_ref = matches!(ctx.get(*l), Expr::SessionRef(1))
                || matches!(ctx.get(*r), Expr::SessionRef(1));
            let has_num =
                matches!(ctx.get(*l), Expr::Number(_)) || matches!(ctx.get(*r), Expr::Number(_));
            assert!(has_ref && has_num, "Expected SessionRef and Number in Add");
        } else {
            panic!("Expected Add");
        }
    }

    #[test]
    fn test_parse_session_ref_in_function() {
        let mut ctx = Context::new();
        // sin(#12)
        let e = parse("sin(#12)", &mut ctx).unwrap();
        if let Expr::Function(name, args) = ctx.get(e) {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            if let Expr::SessionRef(id) = ctx.get(args[0]) {
                assert_eq!(*id, 12);
            } else {
                panic!("Expected SessionRef(12) as function arg");
            }
        } else {
            panic!("Expected Function");
        }
    }

    #[test]
    fn test_parse_session_ref_with_power() {
        let mut ctx = Context::new();
        // #3^2
        let e = parse("#3^2", &mut ctx).unwrap();
        if let Expr::Pow(base, exp) = ctx.get(e) {
            if let Expr::SessionRef(id) = ctx.get(*base) {
                assert_eq!(*id, 3);
            } else {
                panic!("Expected SessionRef(3) as base");
            }
            if let Expr::Number(n) = ctx.get(*exp) {
                assert_eq!(n.to_integer(), 2.into());
            } else {
                panic!("Expected Number(2) as exponent");
            }
        } else {
            panic!("Expected Pow");
        }
    }

    #[test]
    fn test_parse_session_ref_in_parens() {
        let mut ctx = Context::new();
        let e = parse("(#5)", &mut ctx).unwrap();
        if let Expr::SessionRef(id) = ctx.get(e) {
            assert_eq!(*id, 5);
        } else {
            panic!("Expected SessionRef(5)");
        }
    }

    #[test]
    fn test_parse_session_ref_zero() {
        let mut ctx = Context::new();
        let e = parse("#0", &mut ctx).unwrap();
        if let Expr::SessionRef(id) = ctx.get(e) {
            assert_eq!(*id, 0);
        } else {
            panic!("Expected SessionRef(0)");
        }
    }

    #[test]
    fn test_implicit_multiplication() {
        let mut ctx = Context::new();

        // 2x should parse as 2 * x
        let e = parse("2x", &mut ctx).unwrap();
        if let Expr::Mul(l, r) = ctx.get(e) {
            if let Expr::Number(n) = ctx.get(*l) {
                assert_eq!(n.to_integer(), 2.into());
            } else {
                panic!("Expected Number(2) on left");
            }
            if let Expr::Variable(v) = ctx.get(*r) {
                assert_eq!(v, "x");
            } else {
                panic!("Expected Variable(x) on right");
            }
        } else {
            panic!("Expected Mul for '2x'");
        }

        // x1 should remain as a single variable (not 'x' * '1')
        let e2 = parse("x1", &mut ctx).unwrap();
        if let Expr::Variable(v) = ctx.get(e2) {
            assert_eq!(v, "x1", "x1 should be a single variable");
        } else {
            panic!("Expected Variable(x1)");
        }

        // 3(a+b) should parse as 3 * (a + b)
        let e3 = parse("3(a+b)", &mut ctx).unwrap();
        if let Expr::Mul(l, _) = ctx.get(e3) {
            if let Expr::Number(n) = ctx.get(*l) {
                assert_eq!(n.to_integer(), 3.into());
            } else {
                panic!("Expected Number(3) on left");
            }
        } else {
            panic!("Expected Mul for '3(a+b)'");
        }

        // 2pi should parse as 2 * pi
        let e4 = parse("2pi", &mut ctx).unwrap();
        if let Expr::Mul(l, r) = ctx.get(e4) {
            if let Expr::Number(n) = ctx.get(*l) {
                assert_eq!(n.to_integer(), 2.into());
            }
            assert!(matches!(ctx.get(*r), Expr::Constant(_)));
        } else {
            panic!("Expected Mul for '2pi'");
        }
    }
}
