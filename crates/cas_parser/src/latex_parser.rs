//! LaTeX to Expression Parser
//!
//! Parses LaTeX strings back into Expr trees for reversibility testing.
//! Uses a hybrid approach: nom-based tokenizer + recursive descent parser.
//!
//! # Example
//! ```ignore
//! let ctx = &mut Context::new();
//! let expr_id = parse_latex(ctx, "\\frac{1}{2}")?;
//! // expr_id now represents Div(1, 2) or Number(1/2)
//! ```

use cas_ast::{Constant, Context, Expr, ExprId};
use nom::{
    branch::alt,
    bytes::complete::take_while1,
    character::complete::{char, digit1, multispace0},
    combinator::map,
    IResult,
};
use num_bigint::BigInt;
use num_rational::BigRational;

// ============================================================================
// Token Definition
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Number(String),   // "123", "45"
    Variable(String), // "x", "y", "abc"

    // Operators
    Plus,  // +
    Minus, // -
    Cdot,  // \cdot (explicit multiplication)
    Caret, // ^

    // Grouping
    LBrace,   // {
    RBrace,   // }
    LParen,   // (
    RParen,   // )
    LBracket, // [
    RBracket, // ]

    // LaTeX Commands
    Frac,  // \frac
    Sqrt,  // \sqrt
    Sin,   // \sin
    Cos,   // \cos
    Tan,   // \tan
    Cot,   // \cot
    Sec,   // \sec
    Csc,   // \csc
    Ln,    // \ln
    Log,   // \log
    Pi,    // \pi
    Infty, // \infty
    Text,  // \text (followed by content in braces)

    // Arrow (for transformations, we'll skip these)
    RightArrow, // \rightarrow
}

// ============================================================================
// Tokenizer (using nom)
// ============================================================================

fn is_alpha(c: char) -> bool {
    c.is_ascii_alphabetic()
}

/// Parse a LaTeX command like \frac, \sqrt, \sin, etc.
fn latex_command(input: &str) -> IResult<&str, Token> {
    let (input, _) = char('\\')(input)?;
    let (input, cmd) = take_while1(is_alpha)(input)?;

    let token = match cmd {
        "frac" => Token::Frac,
        "sqrt" => Token::Sqrt,
        "sin" => Token::Sin,
        "cos" => Token::Cos,
        "tan" => Token::Tan,
        "cot" => Token::Cot,
        "sec" => Token::Sec,
        "csc" => Token::Csc,
        "ln" => Token::Ln,
        "log" => Token::Log,
        "pi" => Token::Pi,
        "infty" => Token::Infty,
        "cdot" => Token::Cdot,
        "rightarrow" => Token::RightArrow,
        "text" => Token::Text,
        "textbf" => Token::Text,               // Treat \textbf same as \text
        _ => Token::Variable(cmd.to_string()), // Unknown commands become variables
    };

    Ok((input, token))
}

/// Parse a number (integer)
fn number(input: &str) -> IResult<&str, Token> {
    let (input, digits) = digit1(input)?;
    Ok((input, Token::Number(digits.to_string())))
}

/// Parse a variable (single letter or multi-letter identifier)
fn variable(input: &str) -> IResult<&str, Token> {
    let (input, name) = take_while1(is_alpha)(input)?;
    Ok((input, Token::Variable(name.to_string())))
}

/// Parse a single-character operator or grouping symbol
fn operator_or_grouping(input: &str) -> IResult<&str, Token> {
    alt((
        map(char('+'), |_| Token::Plus),
        map(char('-'), |_| Token::Minus),
        map(char('^'), |_| Token::Caret),
        map(char('{'), |_| Token::LBrace),
        map(char('}'), |_| Token::RBrace),
        map(char('('), |_| Token::LParen),
        map(char(')'), |_| Token::RParen),
        map(char('['), |_| Token::LBracket),
        map(char(']'), |_| Token::RBracket),
    ))(input)
}

/// Parse a single token (with leading whitespace consumed)
fn token(input: &str) -> IResult<&str, Token> {
    let (input, _) = multispace0(input)?;
    alt((latex_command, number, operator_or_grouping, variable))(input)
}

/// Tokenize entire LaTeX string
pub fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut remaining = input;

    while !remaining.is_empty() {
        // Skip whitespace
        let trimmed = remaining.trim_start();
        if trimmed.is_empty() {
            break;
        }
        remaining = trimmed;

        match token(remaining) {
            Ok((rest, tok)) => {
                // Skip Text tokens and their content (they're labels, not math)
                if matches!(tok, Token::Text) {
                    // Skip the { content } after \text
                    if let Some(start) = rest.find('{') {
                        if let Some(end) = rest[start..].find('}') {
                            remaining = &rest[start + end + 1..];
                            continue;
                        }
                    }
                }
                // Skip RightArrow (transformation separator)
                if !matches!(tok, Token::RightArrow) {
                    tokens.push(tok);
                }
                remaining = rest;
            }
            Err(_) => {
                return Err(format!(
                    "Failed to tokenize at: '{}'",
                    &remaining[..remaining.len().min(20)]
                ));
            }
        }
    }

    Ok(tokens)
}

// ============================================================================
// Parser (Recursive Descent)
// ============================================================================

pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    ctx: &'a mut Context,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token], ctx: &'a mut Context) -> Self {
        Self {
            tokens,
            pos: 0,
            ctx,
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        match self.advance() {
            Some(tok) if tok == expected => Ok(()),
            Some(tok) => Err(format!("Expected {:?}, got {:?}", expected, tok)),
            None => Err(format!("Expected {:?}, got EOF", expected)),
        }
    }

    /// Parse expression (lowest precedence: addition/subtraction)
    pub fn parse_expr(&mut self) -> Result<ExprId, String> {
        let mut left = self.parse_term()?;

        while let Some(tok) = self.peek() {
            match tok {
                Token::Plus => {
                    self.advance();
                    let right = self.parse_term()?;
                    left = self.ctx.add(Expr::Add(left, right));
                }
                Token::Minus => {
                    self.advance();
                    let right = self.parse_term()?;
                    left = self.ctx.add(Expr::Sub(left, right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Parse term (multiplication - implicit or explicit with \cdot)
    fn parse_term(&mut self) -> Result<ExprId, String> {
        let mut left = self.parse_power()?;

        // Handle implicit multiplication: two atoms next to each other
        while let Some(tok) = self.peek() {
            match tok {
                Token::Cdot => {
                    self.advance();
                    let right = self.parse_power()?;
                    left = self.ctx.add(Expr::Mul(left, right));
                }
                // Implicit multiplication: variable, number, frac, sqrt, function, or lparen
                Token::Variable(_)
                | Token::Number(_)
                | Token::Frac
                | Token::Sqrt
                | Token::LBrace
                | Token::LParen
                | Token::Sin
                | Token::Cos
                | Token::Tan
                | Token::Cot
                | Token::Sec
                | Token::Csc
                | Token::Ln
                | Token::Log => {
                    let right = self.parse_power()?;
                    left = self.ctx.add(Expr::Mul(left, right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Parse power (exponentiation with ^)
    fn parse_power(&mut self) -> Result<ExprId, String> {
        let base = self.parse_unary()?;

        if let Some(Token::Caret) = self.peek() {
            self.advance();
            let exp = self.parse_atom()?; // Exponent is typically in braces
            Ok(self.ctx.add(Expr::Pow(base, exp)))
        } else {
            Ok(base)
        }
    }

    /// Parse unary (negation)
    fn parse_unary(&mut self) -> Result<ExprId, String> {
        if let Some(Token::Minus) = self.peek() {
            self.advance();
            let inner = self.parse_atom()?;
            Ok(self.ctx.add(Expr::Neg(inner)))
        } else {
            self.parse_atom()
        }
    }

    /// Parse atom (numbers, variables, braced groups, fractions, etc.)
    fn parse_atom(&mut self) -> Result<ExprId, String> {
        match self.peek().cloned() {
            Some(Token::Number(s)) => {
                self.advance();
                let n: BigInt = s.parse().map_err(|e| format!("Invalid number: {}", e))?;
                Ok(self.ctx.add(Expr::Number(BigRational::from_integer(n))))
            }
            Some(Token::Variable(name)) => {
                self.advance();
                Ok(self.ctx.var(&name))
            }
            Some(Token::Pi) => {
                self.advance();
                Ok(self.ctx.add(Expr::Constant(Constant::Pi)))
            }
            Some(Token::Infty) => {
                self.advance();
                Ok(self.ctx.add(Expr::Constant(Constant::Infinity)))
            }
            Some(Token::LBrace) => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&Token::RBrace)?;
                Ok(expr)
            }
            Some(Token::LParen) => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Some(Token::Frac) => {
                self.advance();
                self.expect(&Token::LBrace)?;
                let numer = self.parse_expr()?;
                self.expect(&Token::RBrace)?;
                self.expect(&Token::LBrace)?;
                let denom = self.parse_expr()?;
                self.expect(&Token::RBrace)?;
                Ok(self.ctx.add(Expr::Div(numer, denom)))
            }
            Some(Token::Sqrt) => {
                self.advance();
                // Check for optional index: \sqrt[n]{...}
                if let Some(Token::LBracket) = self.peek() {
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect(&Token::RBracket)?;
                    self.expect(&Token::LBrace)?;
                    let radicand = self.parse_expr()?;
                    self.expect(&Token::RBrace)?;
                    // sqrt(radicand, index) - root with index
                    Ok(self.ctx.call("sqrt", vec![radicand, index]))
                } else {
                    self.expect(&Token::LBrace)?;
                    let radicand = self.parse_expr()?;
                    self.expect(&Token::RBrace)?;
                    // sqrt(radicand) - square root
                    Ok(self.ctx.call("sqrt", vec![radicand]))
                }
            }
            Some(Token::Sin) | Some(Token::Cos) | Some(Token::Tan) | Some(Token::Cot)
            | Some(Token::Sec) | Some(Token::Csc) => {
                let name = match self.advance() {
                    Some(Token::Sin) => "sin",
                    Some(Token::Cos) => "cos",
                    Some(Token::Tan) => "tan",
                    Some(Token::Cot) => "cot",
                    Some(Token::Sec) => "sec",
                    Some(Token::Csc) => "csc",
                    _ => unreachable!(),
                };
                self.expect(&Token::LParen)?;
                let arg = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(self.ctx.call(name, vec![arg]))
            }
            Some(Token::Ln) => {
                self.advance();
                self.expect(&Token::LParen)?;
                let arg = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(self.ctx.call("ln", vec![arg]))
            }
            Some(tok) => Err(format!("Unexpected token in atom: {:?}", tok)),
            None => Err("Unexpected end of input".to_string()),
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Parse a LaTeX string into an expression
pub fn parse_latex(ctx: &mut Context, latex: &str) -> Result<ExprId, String> {
    let tokens = tokenize(latex)?;
    if tokens.is_empty() {
        return Err("Empty input".to_string());
    }

    let mut parser = Parser::new(&tokens, ctx);
    let result = parser.parse_expr()?;

    // Ensure all tokens were consumed
    if parser.pos < parser.tokens.len() {
        return Err(format!(
            "Unconsumed tokens starting at: {:?}",
            &parser.tokens[parser.pos..]
        ));
    }

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::LaTeXExpr;

    /// Helper: test parsing works and return LaTeX output
    fn test_parse(latex: &str) -> String {
        let mut ctx = Context::new();
        let expr_id =
            parse_latex(&mut ctx, latex).unwrap_or_else(|_| panic!("Failed to parse: {}", latex));
        LaTeXExpr {
            context: &ctx,
            id: expr_id,
        }
        .to_latex()
    }

    // Level 1: Numbers and Variables
    #[test]
    fn test_parse_number() {
        assert_eq!(test_parse("5"), "5");
        assert_eq!(test_parse("123"), "123");
    }

    #[test]
    fn test_parse_variable() {
        assert_eq!(test_parse("x"), "x");
        assert_eq!(test_parse("y"), "y");
    }

    #[test]
    fn test_parse_negative() {
        assert_eq!(test_parse("-3"), "-3");
        assert_eq!(test_parse("-x"), "-x");
    }

    // Level 2: Basic Operations
    #[test]
    fn test_parse_addition() {
        assert_eq!(test_parse("x + y"), "x + y");
    }

    #[test]
    fn test_parse_subtraction() {
        assert_eq!(test_parse("x - y"), "x - y");
    }

    #[test]
    fn test_parse_fraction() {
        assert_eq!(test_parse("\\frac{1}{2}"), "\\frac{1}{2}");
        assert_eq!(test_parse("\\frac{a}{b}"), "\\frac{a}{b}");
    }

    // Level 3: Powers
    #[test]
    fn test_parse_power() {
        assert_eq!(test_parse("{x}^{2}"), "{x}^{2}");
    }

    #[test]
    fn test_parse_sqrt() {
        assert_eq!(test_parse("\\sqrt{x}"), "\\sqrt{x}");
    }

    // Level 4: Constants and Functions
    #[test]
    fn test_parse_pi() {
        assert_eq!(test_parse("\\pi"), "\\pi");
    }

    #[test]
    fn test_parse_trig() {
        assert_eq!(test_parse("\\sin(x)"), "\\sin(x)");
        assert_eq!(test_parse("\\cos(x)"), "\\cos(x)");
    }

    #[test]
    fn test_parse_implicit_mult() {
        // Note: Variable * Variable is rendered as implicit multiplication (xy) for readability
        // while Number * Variable uses explicit cdot (2\cdot x)
        assert_eq!(test_parse("xy"), "xy");
        assert_eq!(test_parse("2x"), "2\\cdot x");
    }

    #[test]
    fn test_parse_explicit_mult() {
        assert_eq!(test_parse("2\\cdot 3"), "2\\cdot 3");
    }

    // Level 5: Complex Expressions
    #[test]
    fn test_parse_nested_power() {
        assert_eq!(test_parse("{{x}^{2}}^{3}"), "{{x}^{2}}^{3}");
    }

    #[test]
    fn test_parse_power_with_frac_exponent() {
        // V2.14.40: Fractional powers now render as sqrt by default
        assert_eq!(test_parse("{x}^{\\frac{1}{2}}"), "\\sqrt{x}");
    }

    #[test]
    fn test_parse_sqrt_of_power() {
        assert_eq!(test_parse("\\sqrt{{x}^{2}}"), "\\sqrt{{x}^{2}}");
    }

    #[test]
    fn test_parse_subtraction_complex() {
        // V2.14.40: Fractional powers now render as sqrt by default
        // This tests the problematic pattern from timeline
        assert_eq!(
            test_parse("{x}^{\\frac{17}{24}} - {x}^{\\frac{17}{24}}"),
            "\\sqrt[24]{{x}^{17}} - \\sqrt[24]{{x}^{17}}"
        );
    }

    #[test]
    fn test_parse_sqrt_power_minus() {
        // V2.14.40: Fractional powers now render as sqrt by default
        // Simplified version of timeline step 5
        assert_eq!(
            test_parse("\\sqrt{{x}^{\\frac{17}{12}}} - {x}^{\\frac{17}{24}}"),
            "\\sqrt{\\sqrt[12]{{x}^{17}}} - \\sqrt[24]{{x}^{17}}"
        );
    }

    // Tokenizer tests
    #[test]
    fn test_tokenize_simple() {
        let tokens = tokenize("x + y").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], Token::Variable("x".to_string()));
        assert_eq!(tokens[1], Token::Plus);
        assert_eq!(tokens[2], Token::Variable("y".to_string()));
    }

    #[test]
    fn test_tokenize_frac() {
        let tokens = tokenize("\\frac{1}{2}").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Frac,
                Token::LBrace,
                Token::Number("1".to_string()),
                Token::RBrace,
                Token::LBrace,
                Token::Number("2".to_string()),
                Token::RBrace,
            ]
        );
    }
}
