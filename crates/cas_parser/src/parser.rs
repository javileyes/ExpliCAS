use cas_ast::Expr;
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, digit1, multispace0},
    combinator::{map, map_res},
    multi::{fold_many0, separated_list0},
    sequence::{delimited, pair, preceded},
    IResult,
};
use std::rc::Rc;

// Parser for integers
fn parse_i64(input: &str) -> IResult<&str, i64> {
    map_res(digit1, |s: &str| s.parse::<i64>())(input)
}

// Parser for numbers -> Expr::Number
fn parse_number(input: &str) -> IResult<&str, Rc<Expr>> {
    map(parse_i64, Expr::num)(input)
}

// Parser for constants -> Expr::Constant
fn parse_constant(input: &str) -> IResult<&str, Rc<Expr>> {
    alt((
        map(tag("pi"), |_| Expr::pi()),
        map(tag("e"), |_| Expr::e()),
    ))(input)
}

// Parser for variables -> Expr::Variable
fn parse_variable(input: &str) -> IResult<&str, Rc<Expr>> {
    map(alpha1, |s: &str| Expr::var(s))(input)
}

// Parser for parentheses: ( expr )
fn parse_parens(input: &str) -> IResult<&str, Rc<Expr>> {
    delimited(
        preceded(multispace0, tag("(")),
        parse_expr,
        preceded(multispace0, tag(")")),
    )(input)
}

// Parser for function calls: name(arg1, arg2, ...)
fn parse_function(input: &str) -> IResult<&str, Rc<Expr>> {
    let (input, name) = alpha1(input)?;
    let (input, _) = preceded(multispace0, tag("("))(input)?;
    let (input, args) = separated_list0(
        preceded(multispace0, tag(",")),
        parse_expr,
    )(input)?;
    let (input, _) = preceded(multispace0, tag(")"))(input)?;
    
    if name == "ln" && args.len() == 1 {
        return Ok((input, Expr::ln(args[0].clone())));
    }
    
    Ok((input, Rc::new(Expr::Function(name.to_string(), args))))
}

// Atom: number, variable, function, or (expr)
fn parse_atom(input: &str) -> IResult<&str, Rc<Expr>> {
    preceded(
        multispace0,
        alt((
            parse_number,
            // Try parse_function before parse_variable because "sqrt" matches alpha1 too
            // However, parse_function requires '(', so we can try it first.
            // Actually, alpha1 matches the name in parse_function.
            // Let's rely on the fact that parse_function expects '(' after name.
            // But parse_variable just takes the name.
            // We need to be careful. parse_variable consumes "sqrt" and leaves "(".
            // So we should try parse_function first.
            parse_function,
            parse_constant,
            parse_variable,
            parse_parens,
        )),
    )(input)
}

// Power: atom ^ atom
fn parse_power(input: &str) -> IResult<&str, Rc<Expr>> {
    let (input, init) = parse_atom(input)?;
    fold_many0(
        pair(
            preceded(multispace0, tag("^")),
            parse_atom,
        ),
        move || init.clone(),
        |acc, (_, val)| Expr::pow(acc, val),
    )(input)
}

// Unary: - Unary or Power
fn parse_unary(input: &str) -> IResult<&str, Rc<Expr>> {
    alt((
        map(
            pair(preceded(multispace0, tag("-")), parse_unary),
            |(_, expr)| Expr::neg(expr),
        ),
        parse_power,
    ))(input)
}

// Term: unary * unary or unary / unary
fn parse_term(input: &str) -> IResult<&str, Rc<Expr>> {
    let (input, init) = parse_unary(input)?;
    fold_many0(
        pair(
            preceded(multispace0, alt((tag("*"), tag("/")))),
            parse_unary,
        ),
        move || init.clone(),
        |acc, (op, val)| match op {
            "*" => Expr::mul(acc, val),
            "/" => Expr::div(acc, val),
            _ => unreachable!(),
        },
    )(input)
}

// Expr: term + term or term - term
fn parse_expr(input: &str) -> IResult<&str, Rc<Expr>> {
    let (input, init) = parse_term(input)?;
    fold_many0(
        pair(
            preceded(multispace0, alt((tag("+"), tag("-")))),
            parse_term,
        ),
        move || init.clone(),
        |acc, (op, val)| match op {
            "+" => Expr::add(acc, val),
            "-" => Expr::sub(acc, val),
            _ => unreachable!(),
        },
    )(input)
}

use cas_ast::{Equation, RelOp};

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Expression(Rc<Expr>),
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

// Parser for equations: expr op expr
fn parse_equation(input: &str) -> IResult<&str, Equation> {
    let (input, lhs) = parse_expr(input)?;
    let (input, op) = parse_relop(input)?;
    let (input, rhs) = parse_expr(input)?;
    Ok((input, Equation { lhs, rhs, op }))
}

pub fn parse(input: &str) -> Result<Rc<Expr>, String> {
    // Legacy support for just expressions
    match parse_expr(input) {
        Ok((remaining, expr)) => {
            if remaining.trim().is_empty() {
                Ok(expr)
            } else {
                Err(format!("Unconsumed input: {}", remaining))
            }
        }
        Err(e) => Err(format!("Parse error: {}", e)),
    }
}

pub fn parse_statement(input: &str) -> Result<Statement, String> {
    // Try parsing as equation first
    if let Ok((remaining, eq)) = parse_equation(input) {
        if remaining.trim().is_empty() {
            return Ok(Statement::Equation(eq));
        }
    }

    // Fallback to expression
    match parse_expr(input) {
        Ok((remaining, expr)) => {
            if remaining.trim().is_empty() {
                Ok(Statement::Expression(expr))
            } else {
                Err(format!("Unconsumed input: {}", remaining))
            }
        }
        Err(e) => Err(format!("Parse error: {}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_number() {
        let e = parse("123").unwrap();
        assert_eq!(format!("{}", e), "123");
    }

    #[test]
    fn test_parse_variable() {
        let e = parse("x").unwrap();
        assert_eq!(format!("{}", e), "x");
    }

    #[test]
    fn test_parse_arithmetic() {
        let e = parse("1 + 2 * x").unwrap();
        // Note: Operator precedence is handled by the parser structure (Term before Expr)
        // Term handles * and /, Expr handles + and -
        // So 1 + 2 * x should be 1 + (2 * x)
        assert_eq!(format!("{}", e), "1 + 2 * x");
    }

    #[test]
    fn test_parse_parens() {
        let e = parse("(1 + 2) * x").unwrap();
        assert_eq!(format!("{}", e), "(1 + 2) * x");
    }

    #[test]
    fn test_parse_power() {
        let e = parse("x^2").unwrap();
        assert_eq!(format!("{}", e), "x^2");
        
        let e2 = parse("x^2 * y").unwrap();
        assert_eq!(format!("{}", e2), "x^2 * y");
    }
}
