use cas_ast::{Expr, Constant, ExprId, Context, Equation, RelOp};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, digit1, multispace0},
    combinator::{map, map_res},
    multi::{fold_many0, separated_list0},
    sequence::{delimited, pair, preceded},
    IResult,
};
use num_rational::BigRational;

use num_bigint::BigInt;

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
            },
            ParseNode::Sub(l, r) => {
                let lid = l.lower(ctx);
                let rid = r.lower(ctx);
                ctx.add(Expr::Sub(lid, rid))
            },
            ParseNode::Mul(l, r) => {
                let lid = l.lower(ctx);
                let rid = r.lower(ctx);
                ctx.add(Expr::Mul(lid, rid))
            },
            ParseNode::Div(l, r) => {
                let lid = l.lower(ctx);
                let rid = r.lower(ctx);
                ctx.add(Expr::Div(lid, rid))
            },
            ParseNode::Pow(b, e) => {
                let bid = b.lower(ctx);
                let eid = e.lower(ctx);
                ctx.add(Expr::Pow(bid, eid))
            },
            ParseNode::Neg(e) => {
                let eid = e.lower(ctx);
                ctx.add(Expr::Neg(eid))
            },
            ParseNode::Function(name, args) => {
                let arg_ids = args.into_iter().map(|a| a.lower(ctx)).collect();
                ctx.add(Expr::Function(name, arg_ids))
            },
        }
    }
}

// Parser for integers
fn parse_i64(input: &str) -> IResult<&str, i64> {
    map_res(digit1, |s: &str| s.parse::<i64>())(input)
}

// Parser for numbers
fn parse_number(input: &str) -> IResult<&str, ParseNode> {
    map(parse_i64, |n| ParseNode::Number(BigRational::from_integer(BigInt::from(n))))(input)
}

// Parser for constants
fn parse_constant(input: &str) -> IResult<&str, ParseNode> {
    alt((
        map(tag("pi"), |_| ParseNode::Constant(Constant::Pi)),
        map(tag("e"), |_| ParseNode::Constant(Constant::E)),
    ))(input)
}

// Parser for variables
fn parse_variable(input: &str) -> IResult<&str, ParseNode> {
    map(alpha1, |s: &str| ParseNode::Variable(s.to_string()))(input)
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
    let (input, name) = alpha1(input)?;
    let (input, _) = preceded(multispace0, tag("("))(input)?;
    let (input, args) = separated_list0(
        preceded(multispace0, tag(",")),
        parse_expr,
    )(input)?;
    let (input, _) = preceded(multispace0, tag(")"))(input)?;
    
    if name == "ln" && args.len() == 1 {
        // ln(x) -> log(e, x)
        return Ok((input, ParseNode::Function("log".to_string(), vec![ParseNode::Constant(Constant::E), args[0].clone()])));
    }
    
    if name == "exp" && args.len() == 1 {
        // exp(x) -> e^x
        return Ok((input, ParseNode::Pow(Box::new(ParseNode::Constant(Constant::E)), Box::new(args[0].clone()))));
    }
    
    Ok((input, ParseNode::Function(name.to_string(), args)))
}

// Parser for absolute value
fn parse_abs(input: &str) -> IResult<&str, ParseNode> {
    delimited(
        preceded(multispace0, tag("|")),
        parse_expr,
        preceded(multispace0, tag("|")),
    )(input).map(|(next_input, expr)| (next_input, ParseNode::Function("abs".to_string(), vec![expr])))
}

// Atom
fn parse_atom(input: &str) -> IResult<&str, ParseNode> {
    preceded(
        multispace0,
        alt((
            parse_number,
            parse_function,
            parse_constant,
            parse_variable,
            parse_parens,
            parse_abs,
        )),
    )(input)
}

// Power
fn parse_power(input: &str) -> IResult<&str, ParseNode> {
    let (input, init) = parse_atom(input)?;
    fold_many0(
        pair(
            preceded(multispace0, tag("^")),
            parse_atom,
        ),
        move || init.clone(),
        |acc, (_, val)| ParseNode::Pow(Box::new(acc), Box::new(val)),
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

// Term
fn parse_term(input: &str) -> IResult<&str, ParseNode> {
    let (input, init) = parse_unary(input)?;
    fold_many0(
        pair(
            preceded(multispace0, alt((tag("*"), tag("/")))),
            parse_unary,
        ),
        move || init.clone(),
        |acc, (op, val)| match op {
            "*" => ParseNode::Mul(Box::new(acc), Box::new(val)),
            "/" => ParseNode::Div(Box::new(acc), Box::new(val)),
            _ => unreachable!(),
        },
    )(input)
}

// Expr
fn parse_expr(input: &str) -> IResult<&str, ParseNode> {
    let (input, init) = parse_term(input)?;
    fold_many0(
        pair(
            preceded(multispace0, alt((tag("+"), tag("-")))),
            parse_term,
        ),
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
    let (remaining, expr_node) = parse_expr(input)
        .map_err(|e| ParseError::NomError(format!("{}", e)))?;
    
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
            return Ok(Statement::Equation(Equation { lhs: lhs_id, rhs: rhs_id, op }));
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
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: e }), "123");
    }

    #[test]
    fn test_parse_variable() {
        let mut ctx = Context::new();
        let e = parse("x", &mut ctx).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: e }), "x");
    }

    #[test]
    fn test_parse_arithmetic() {
        let mut ctx = Context::new();
        let e = parse("1 + 2 * x", &mut ctx).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: e }), "1 + 2 * x");
    }

    #[test]
    fn test_parse_parens() {
        let mut ctx = Context::new();
        let e = parse("(1 + 2) * x", &mut ctx).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: e }), "(1 + 2) * x");
    }

    #[test]
    fn test_parse_power() {
        let mut ctx = Context::new();
        let e = parse("x^2", &mut ctx).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: e }), "x^2");
        
        let e2 = parse("x^2 * y", &mut ctx).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: e2 }), "x^2 * y");
    }
}
