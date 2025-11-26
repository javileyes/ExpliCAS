use cas_ast::Expr;
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, digit1, multispace0},
    combinator::{map, map_res},
    multi::fold_many0,
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

// Atom: number, variable, or (expr)
fn parse_atom(input: &str) -> IResult<&str, Rc<Expr>> {
    preceded(
        multispace0,
        alt((parse_number, parse_variable, parse_parens)),
    )(input)
}

// Factor: atom ^ atom (Power)
fn parse_factor(input: &str) -> IResult<&str, Rc<Expr>> {
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

// Term: factor * factor or factor / factor
fn parse_term(input: &str) -> IResult<&str, Rc<Expr>> {
    let (input, init) = parse_factor(input)?;
    fold_many0(
        pair(
            preceded(multispace0, alt((tag("*"), tag("/")))),
            parse_factor,
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

pub fn parse(input: &str) -> Result<Rc<Expr>, String> {
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
