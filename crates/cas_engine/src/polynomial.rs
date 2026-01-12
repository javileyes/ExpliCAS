use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::cmp::max;

#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    // Coefficients ordered by power: coeffs[i] is the coefficient of x^i
    pub coeffs: Vec<BigRational>,
    pub var: String,
}

impl Polynomial {
    pub fn new(coeffs: Vec<BigRational>, var: String) -> Self {
        let mut poly = Polynomial { coeffs, var };
        poly.trim();
        poly
    }

    pub fn zero(var: String) -> Self {
        Polynomial {
            coeffs: vec![],
            var,
        }
    }

    pub fn one(var: String) -> Self {
        Polynomial {
            coeffs: vec![BigRational::one()],
            var,
        }
    }

    fn trim(&mut self) {
        while let Some(c) = self.coeffs.last() {
            if c.is_zero() {
                self.coeffs.pop();
            } else {
                break;
            }
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub fn degree(&self) -> usize {
        if self.is_zero() {
            0 // Technically -infinity, but 0 for implementation convenience often works if handled carefully
        } else {
            self.coeffs.len() - 1
        }
    }

    pub fn leading_coeff(&self) -> BigRational {
        if self.is_zero() {
            BigRational::zero()
        } else {
            self.coeffs
                .last()
                .cloned()
                .unwrap_or_else(BigRational::zero)
        }
    }

    // Convert Expr to Polynomial. Returns CasError if not a polynomial in `var`.
    pub fn from_expr(
        context: &Context,
        expr: ExprId,
        var: &str,
    ) -> Result<Self, crate::error::CasError> {
        use crate::error::CasError;
        let expr_data = context.get(expr);
        match expr_data {
            Expr::Number(n) => Ok(Polynomial::new(vec![n.clone()], var.to_string())),
            Expr::Variable(v) => {
                if v == var {
                    // x = 0 + 1*x
                    Ok(Polynomial::new(
                        vec![BigRational::zero(), BigRational::one()],
                        var.to_string(),
                    ))
                } else {
                    // Treat other variables as constants? For now, fail.
                    Err(CasError::PolynomialError(format!(
                        "Variable mismatch: expected '{}', found '{}'",
                        var, v
                    )))
                }
            }
            Expr::Add(l, r) => {
                let p1 = Polynomial::from_expr(context, *l, var)?;
                let p2 = Polynomial::from_expr(context, *r, var)?;
                Ok(p1.add(&p2))
            }
            Expr::Sub(l, r) => {
                let p1 = Polynomial::from_expr(context, *l, var)?;
                let p2 = Polynomial::from_expr(context, *r, var)?;
                Ok(p1.sub(&p2))
            }
            Expr::Mul(l, r) => {
                let p1 = Polynomial::from_expr(context, *l, var)?;
                let p2 = Polynomial::from_expr(context, *r, var)?;
                Ok(p1.mul(&p2))
            }
            Expr::Pow(base, exp) => {
                // Only handle x^n where n is non-negative integer
                if let Expr::Number(n) = context.get(*exp) {
                    if n.is_integer() && *n >= BigRational::zero() {
                        let p_base = Polynomial::from_expr(context, *base, var)?;
                        // Naive power
                        let exp_usize = n.to_integer().try_into().map_err(|_| {
                            CasError::PolynomialError("Exponent too large".to_string())
                        })?;
                        let mut res = Polynomial::one(var.to_string());
                        for _ in 0..exp_usize {
                            res = res.mul(&p_base);
                        }
                        return Ok(res);
                    }
                }
                Err(CasError::PolynomialError(format!(
                    "Unsupported power: {:?}",
                    context.get(*exp)
                )))
            }
            Expr::Neg(e) => {
                let p = Polynomial::from_expr(context, *e, var)?;
                Ok(p.neg())
            }
            _ => Err(CasError::PolynomialError(format!(
                "Unsupported expression type for polynomial: {:?}",
                expr_data
            ))),
        }
    }

    pub fn to_expr(&self, context: &mut Context) -> ExprId {
        if self.is_zero() {
            return context.num(0);
        }

        let mut terms = Vec::new();
        for (i, coeff) in self.coeffs.iter().enumerate().rev() {
            if !coeff.is_zero() {
                let term = if i == 0 {
                    context.add(Expr::Number(coeff.clone()))
                } else {
                    let var_expr = context.var(&self.var);
                    let var_part = if i == 1 {
                        var_expr
                    } else {
                        let exp = context.num(i as i64);
                        context.add(Expr::Pow(var_expr, exp))
                    };

                    if coeff.is_one() {
                        var_part
                    } else if *coeff == -BigRational::one() {
                        context.add(Expr::Neg(var_part))
                    } else {
                        let coeff_expr = context.add(Expr::Number(coeff.clone()));
                        context.add(Expr::Mul(coeff_expr, var_part))
                    }
                };
                terms.push(term);
            }
        }

        // Combine terms with Add
        if terms.is_empty() {
            return context.num(0);
        }

        let mut res = terms[0];
        for term in terms.into_iter().skip(1) {
            res = context.add(Expr::Add(res, term));
        }
        res
    }

    pub fn add(&self, other: &Self) -> Self {
        use std::iter;
        let len = max(self.coeffs.len(), other.coeffs.len());

        let it1 = self
            .coeffs
            .iter()
            .cloned()
            .chain(iter::repeat(BigRational::zero()));
        let it2 = other
            .coeffs
            .iter()
            .cloned()
            .chain(iter::repeat(BigRational::zero()));

        let new_coeffs: Vec<BigRational> = it1.zip(it2).take(len).map(|(c1, c2)| c1 + c2).collect();

        Polynomial::new(new_coeffs, self.var.clone())
    }

    pub fn sub(&self, other: &Self) -> Self {
        use std::iter;
        let len = max(self.coeffs.len(), other.coeffs.len());

        let it1 = self
            .coeffs
            .iter()
            .cloned()
            .chain(iter::repeat(BigRational::zero()));
        let it2 = other
            .coeffs
            .iter()
            .cloned()
            .chain(iter::repeat(BigRational::zero()));

        let new_coeffs: Vec<BigRational> = it1.zip(it2).take(len).map(|(c1, c2)| c1 - c2).collect();

        Polynomial::new(new_coeffs, self.var.clone())
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Polynomial::zero(self.var.clone());
        }
        let new_len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut new_coeffs = vec![BigRational::zero(); new_len];

        for (i, c1) in self.coeffs.iter().enumerate() {
            for (j, c2) in other.coeffs.iter().enumerate() {
                new_coeffs[i + j] += c1 * c2;
            }
        }

        Polynomial::new(new_coeffs, self.var.clone())
    }

    pub fn neg(&self) -> Self {
        let new_coeffs = self.coeffs.iter().map(|c| -c).collect();
        Polynomial::new(new_coeffs, self.var.clone())
    }

    /// Divide all coefficients by a scalar (exact division)
    pub fn div_scalar(&self, k: &BigRational) -> Self {
        if k.is_zero() {
            // Return zero poly to avoid panic (caller should check)
            return Polynomial::zero(self.var.clone());
        }
        let new_coeffs = self.coeffs.iter().map(|c| c / k).collect();
        Polynomial::new(new_coeffs, self.var.clone())
    }

    // Returns (quotient, remainder) or error if divisor is zero
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), crate::error::CasError> {
        if divisor.is_zero() {
            return Err(crate::error::CasError::DivisionByZero);
        }

        if self.degree() < divisor.degree() {
            return Ok((Polynomial::zero(self.var.clone()), self.clone()));
        }

        let mut quotient = Polynomial::zero(self.var.clone());
        let mut remainder = self.clone();
        let divisor_deg = divisor.degree();
        let divisor_lc = divisor.leading_coeff();

        while !remainder.is_zero() && remainder.degree() >= divisor_deg {
            let degree_diff = remainder.degree() - divisor_deg;
            let coeff_quot = remainder.leading_coeff() / &divisor_lc;

            // Create term: coeff * x^degree_diff
            let mut term_coeffs = vec![BigRational::zero(); degree_diff + 1];
            term_coeffs[degree_diff] = coeff_quot.clone();
            let term = Polynomial::new(term_coeffs, self.var.clone());

            quotient = quotient.add(&term);
            let to_subtract = divisor.mul(&term);
            remainder = remainder.sub(&to_subtract);
        }

        Ok((quotient, remainder))
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.clone();
        let mut b = other.clone();

        while !b.is_zero() {
            // div_rem cannot fail here since b is not zero
            let (_, r) = a
                .div_rem(&b)
                .expect("div_rem should not fail: b is not zero");
            a = b;
            b = r;
        }

        // Make the GCD primitive (integer coefficients with GCD=1)
        // Instead of monic (leading coefficient=1)
        if !a.is_zero() {
            let content = a.content();
            if !content.is_zero() && content != BigRational::one() {
                // Divide by content to get primitive polynomial
                let inv_content = BigRational::one() / content;
                let scalar = Polynomial::new(vec![inv_content], a.var.clone());
                a = a.mul(&scalar);
            }

            // Ensure leading coefficient is positive
            let lc = a.leading_coeff();
            if lc < BigRational::zero() {
                a = a.neg();
            }
        }

        a
    }

    pub fn content(&self) -> BigRational {
        if self.is_zero() {
            return BigRational::zero();
        }
        // GCD of all coefficients
        // Since BigRational doesn't have gcd directly exposed easily for all versions,
        // and we are using rational numbers, "content" usually refers to integer content
        // or we can just look for common integer factor if they are integers?
        // For Rationals, it's tricky.
        // Let's simplify: if all coeffs are integers, find integer GCD.
        // If not, return 1.

        // Actually, let's just return 1 for now unless we implement a proper Rational GCD.
        // But we can at least handle the leading coefficient to make it monic?
        // Or extract the leading coefficient?
        // The user asked for "factor common term".
        // e.g. 2x^2 + 4x -> 2x(x+2).
        // Coeffs: [0, 4, 2].
        // GCD(4, 2) = 2.
        // Min power = 1.

        // Let's implement a simple integer GCD for numerators if denominators are 1.
        let mut g = BigRational::zero();
        for c in &self.coeffs {
            if c.is_zero() {
                continue;
            }
            if g.is_zero() {
                g = c.abs();
            } else {
                g = gcd_rational(g, c.abs());
            }
        }
        g
    }

    pub fn min_degree(&self) -> usize {
        for (i, c) in self.coeffs.iter().enumerate() {
            if !c.is_zero() {
                return i;
            }
        }
        0
    }

    pub fn derivative(&self) -> Self {
        if self.degree() == 0 {
            return Polynomial::zero(self.var.clone());
        }
        let mut new_coeffs = Vec::with_capacity(self.degree());
        for (i, c) in self.coeffs.iter().enumerate().skip(1) {
            let power = BigRational::from_integer(i.into());
            new_coeffs.push(c * power);
        }
        Polynomial::new(new_coeffs, self.var.clone())
    }

    pub fn eval(&self, x: &BigRational) -> BigRational {
        let mut res = BigRational::zero();
        // Horner's method
        for c in self.coeffs.iter().rev() {
            res = res * x + c;
        }
        res
    }

    // Returns a list of factors. The product of these factors (times content) equals the polynomial.
    // Factors are not necessarily irreducible if they have no rational roots (e.g. x^2 + 1).
    pub fn factor_rational_roots(&self) -> Vec<Polynomial> {
        if self.is_zero() {
            return vec![self.clone()];
        }
        if self.degree() <= 1 {
            return vec![self.clone()];
        }

        let mut factors = Vec::new();
        let mut current_poly = self.clone();

        // 1. Extract content (to make integer coeffs easier to handle)
        // Actually, Rational Root Theorem works best on integer polynomials.
        // Let's just work with what we have. If coeffs are fractions, we can multiply by LCM of denominators?
        // For MVP, let's assume integer coeffs or simple rationals.
        // We'll try to find roots p/q.

        // Optimization: Make monic first? No, RRT needs a_0 and a_n.

        loop {
            if current_poly.degree() <= 1 {
                factors.push(current_poly);
                break;
            }

            if let Some(root) = current_poly.find_one_rational_root() {
                // Found root r. Factor is (x - r).
                // To keep things clean, we might prefer integer factors like (qx - p).
                // root = p/q.
                let p = root.numer();
                let q = root.denom();

                // Factor: q*x - p
                let factor_coeffs = vec![
                    -BigRational::from_integer(p.clone()),
                    BigRational::from_integer(q.clone()),
                ];
                let factor = Polynomial::new(factor_coeffs, self.var.clone());

                factors.push(factor.clone());
                // factor is never zero (q is non-zero from rational root theorem)
                let (quotient, _) = current_poly
                    .div_rem(&factor)
                    .expect("div_rem should not fail: factor is non-zero");
                current_poly = quotient;
            } else {
                // No more rational roots.
                factors.push(current_poly);
                break;
            }
        }
        factors
    }

    fn find_one_rational_root(&self) -> Option<BigRational> {
        // 1. Get a_0 and a_n
        let a_0 = self.coeffs.first()?;
        let a_n = self.coeffs.last()?;

        if a_0.is_zero() {
            return Some(BigRational::zero()); // 0 is a root
        }

        // Convert to integers if possible. If not, RRT is hard.
        // For now, only support if a_0 and a_n are integers (or we scale).
        // Let's try to scale by LCM of all denominators to get integer poly.
        // But finding root of scaled poly is same as original.

        // Let's just check integer factors of numerator of a_0 and a_n?
        // Simplified approach: Assume integer coefficients for now.
        if !self.are_coeffs_integers() {
            use num_integer::Integer;
            let lcm = self
                .coeffs
                .iter()
                .fold(num_bigint::BigInt::one(), |acc, c| acc.lcm(c.denom()));
            let scale_factor =
                Polynomial::new(vec![BigRational::from_integer(lcm)], self.var.clone());
            let scaled_poly = self.mul(&scale_factor);
            return scaled_poly.find_one_rational_root();
        }

        let p_candidates = get_factors(&a_0.to_integer());
        let q_candidates = get_factors(&a_n.to_integer());

        for p in &p_candidates {
            for q in &q_candidates {
                if q.is_zero() {
                    continue;
                } // Should not happen

                // Test +p/q and -p/q
                let root = BigRational::new(p.clone(), q.clone());
                if self.eval(&root).is_zero() {
                    return Some(root);
                }
                let neg_root = -root;
                if self.eval(&neg_root).is_zero() {
                    return Some(neg_root);
                }
            }
        }
        None
    }

    fn are_coeffs_integers(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_integer())
    }
}

fn get_factors(n: &num_bigint::BigInt) -> Vec<num_bigint::BigInt> {
    use num_integer::Integer;
    use num_traits::Signed;

    let n = n.abs();
    if n.is_zero() {
        return vec![];
    }

    let mut factors = Vec::new();
    let one = num_bigint::BigInt::one();
    let mut i = num_bigint::BigInt::one();

    // Naive factorization up to sqrt(n)
    // Since we are dealing with BigInt, this could be slow for huge numbers.
    // But for textbook CAS problems, coeffs are small.
    // Let's limit search space?

    while &i * &i <= n {
        if n.is_multiple_of(&i) {
            factors.push(i.clone());
            let div = &n / &i;
            if div != i {
                factors.push(div);
            }
        }
        i += &one;
    }
    factors
}

fn gcd_rational(a: BigRational, b: BigRational) -> BigRational {
    // Simple placeholder: if both are integers, use integer GCD.
    if a.is_integer() && b.is_integer() {
        use num_integer::Integer;
        let num_a = a.to_integer();
        let num_b = b.to_integer();
        let g = num_a.gcd(&num_b);
        return BigRational::from_integer(g);
    }
    // If not integers, just return 1 (or smaller of the two? No, 1 is safe).
    BigRational::one()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn test_poly_ops() {
        let x = Polynomial::new(
            vec![BigRational::zero(), BigRational::one()],
            "x".to_string(),
        ); // x
        let one = Polynomial::one("x".to_string());

        // x + 1
        let p1 = x.add(&one);
        assert_eq!(p1.coeffs.len(), 2); // 1, 1

        // (x+1)(x-1) = x^2 - 1
        let p2 = x.sub(&one);
        let prod = p1.mul(&p2);
        // coeffs: -1, 0, 1
        assert_eq!(prod.coeffs[0], -BigRational::one());
        assert_eq!(prod.coeffs[1], BigRational::zero());
        assert_eq!(prod.coeffs[2], BigRational::one());
    }

    #[test]
    fn test_div_rem() {
        let mut ctx = Context::new();
        // (x^2 + 2x + 1) / (x + 1) = x + 1, rem 0
        let expr_num = parse("x^2 + 2*x + 1", &mut ctx).unwrap();
        let p_num = Polynomial::from_expr(&ctx, expr_num, "x").unwrap();
        let expr_den = parse("x + 1", &mut ctx).unwrap();
        let p_den = Polynomial::from_expr(&ctx, expr_den, "x").unwrap();

        let (q, r) = p_num.div_rem(&p_den).unwrap();
        assert!(r.is_zero());
        let q_expr = q.to_expr(&mut ctx);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: q_expr
                }
            ),
            "x + 1" // Canonical: polynomial order (variables before constants)
        );
    }

    #[test]
    fn test_gcd() {
        let mut ctx = Context::new();
        // gcd(x^2 - 1, x^2 + 2x + 1) = x + 1
        let expr1 = parse("x^2 - 1", &mut ctx).unwrap();
        let p1 = Polynomial::from_expr(&ctx, expr1, "x").unwrap();
        let expr2 = parse("x^2 + 2*x + 1", &mut ctx).unwrap();
        let p2 = Polynomial::from_expr(&ctx, expr2, "x").unwrap();

        let g = p1.gcd(&p2);
        // Should be x + 1 (normalized)
        // coeffs: 1, 1
        assert_eq!(g.coeffs.len(), 2);
        assert_eq!(g.coeffs[0], BigRational::one());
        assert_eq!(g.coeffs[1], BigRational::one());
    }

    #[test]
    fn test_div_rem_division_by_zero() {
        use crate::error::CasError;

        let p = Polynomial::new(
            vec![BigRational::one(), BigRational::one()], // x + 1
            "x".to_string(),
        );
        let zero = Polynomial::zero("x".to_string());

        let result = p.div_rem(&zero);
        assert!(matches!(result, Err(CasError::DivisionByZero)));
    }
}
