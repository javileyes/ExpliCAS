use cas_ast::{Constant, Context, Equation, Expr, ExprId, RelOp};
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
    Matrix(Vec<Vec<ParseNode>>), // 2D structure for validation during parsing
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
        }
    }
}

// Parser for integers
fn parse_i64(input: &str) -> IResult<&str, i64> {
    map_res(digit1, |s: &str| s.parse::<i64>())(input)
}

// Parser for numbers
fn parse_number(input: &str) -> IResult<&str, ParseNode> {
    map(parse_i64, |n| {
        ParseNode::Number(BigRational::from_integer(BigInt::from(n)))
    })(input)
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

// Atom
fn parse_atom(input: &str) -> IResult<&str, ParseNode> {
    preceded(
        multispace0,
        alt((
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

fn parse_factorial(input: &str) -> IResult<&str, ParseNode> {
    let (input, atom) = parse_atom(input)?;
    fold_many0(
        preceded(multispace0, tag("!")),
        move || atom.clone(),
        |acc, _| ParseNode::Function("fact".to_string(), vec![acc]),
    )(input)
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

// Term
fn parse_term(input: &str) -> IResult<&str, ParseNode> {
    let (input, init) = parse_unary(input)?;
    fold_many0(
        pair(
            preceded(multispace0, alt((tag("*"), tag("/"), tag("mod")))),
            parse_unary,
        ),
        move || init.clone(),
        |acc, (op, val)| match op {
            "*" => ParseNode::Mul(Box::new(acc), Box::new(val)),
            "/" => ParseNode::Div(Box::new(acc), Box::new(val)),
            "mod" => ParseNode::Function("mod".to_string(), vec![acc, val]),
            _ => unreachable!(),
        },
    )(input)
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
        // x^-2 should parse as Pow(x, Neg(2))
        let e = parse("x^-2", &mut ctx).unwrap();
        if let Expr::Pow(base, exp) = ctx.get(e) {
            // base should be x
            if let Expr::Variable(v) = ctx.get(*base) {
                assert_eq!(v, "x");
            } else {
                panic!("Expected base to be Variable(x)");
            }
            // exp should be Neg(2)
            if let Expr::Neg(inner) = ctx.get(*exp) {
                if let Expr::Number(n) = ctx.get(*inner) {
                    assert_eq!(n.to_integer(), 2.into());
                } else {
                    panic!("Expected Neg inner to be Number(2)");
                }
            } else {
                panic!("Expected exponent to be Neg");
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
}
