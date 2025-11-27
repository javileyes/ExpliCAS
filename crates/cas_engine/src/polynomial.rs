use cas_ast::Expr;
use num_rational::BigRational;
use num_traits::{Zero, One};
use std::rc::Rc;
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
        Polynomial { coeffs: vec![], var }
    }

    pub fn one(var: String) -> Self {
        Polynomial { coeffs: vec![BigRational::one()], var }
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
            self.coeffs.last().unwrap().clone()
        }
    }

    // Convert Expr to Polynomial. Returns None if not a polynomial in `var`.
    pub fn from_expr(expr: &Expr, var: &str) -> Option<Self> {
        match expr {
            Expr::Number(n) => Some(Polynomial::new(vec![n.clone()], var.to_string())),
            Expr::Variable(v) => {
                if v == var {
                    // x = 0 + 1*x
                    Some(Polynomial::new(vec![BigRational::zero(), BigRational::one()], var.to_string()))
                } else {
                    // Treat other variables as constants? For now, fail.
                    // Or maybe treat as constant polynomial?
                    // Let's stick to strict univariate for now.
                    None
                }
            },
            Expr::Add(l, r) => {
                let p1 = Polynomial::from_expr(l, var)?;
                let p2 = Polynomial::from_expr(r, var)?;
                Some(p1.add(&p2))
            },
            Expr::Sub(l, r) => {
                let p1 = Polynomial::from_expr(l, var)?;
                let p2 = Polynomial::from_expr(r, var)?;
                Some(p1.sub(&p2))
            },
            Expr::Mul(l, r) => {
                let p1 = Polynomial::from_expr(l, var)?;
                let p2 = Polynomial::from_expr(r, var)?;
                Some(p1.mul(&p2))
            },
            Expr::Pow(base, exp) => {
                // Only handle x^n where n is non-negative integer
                if let Expr::Number(n) = exp.as_ref() {
                    if n.is_integer() && *n >= BigRational::zero() {
                        let p_base = Polynomial::from_expr(base, var)?;
                        // Naive power
                        let exp_usize = n.to_integer().try_into().ok()?;
                        let mut res = Polynomial::one(var.to_string());
                        for _ in 0..exp_usize {
                            res = res.mul(&p_base);
                        }
                        return Some(res);
                    }
                }
                None
            },
            Expr::Neg(e) => {
                let p = Polynomial::from_expr(e, var)?;
                Some(p.neg())
            },
            _ => None,
        }
    }

    pub fn to_expr(&self) -> Rc<Expr> {
        if self.is_zero() {
            return Expr::num(0);
        }
        
        let mut terms = Vec::new();
        for (i, coeff) in self.coeffs.iter().enumerate().rev() {
            if !coeff.is_zero() {
                let term = if i == 0 {
                    Rc::new(Expr::Number(coeff.clone()))
                } else {
                    let var_part = if i == 1 {
                        Expr::var(&self.var)
                    } else {
                        Expr::pow(Expr::var(&self.var), Expr::num(i as i64))
                    };

                    if coeff.is_one() {
                        var_part
                    } else if *coeff == -BigRational::one() {
                        Expr::neg(var_part)
                    } else {
                        Expr::mul(Rc::new(Expr::Number(coeff.clone())), var_part)
                    }
                };
                terms.push(term);
            }
        }

        // Combine terms with Add
        if terms.is_empty() {
            return Expr::num(0);
        }
        
        let mut res = terms[0].clone();
        for term in terms.into_iter().skip(1) {
            res = Expr::add(res, term);
        }
        res
    }

    pub fn add(&self, other: &Self) -> Self {
        let len = max(self.coeffs.len(), other.coeffs.len());
        let mut new_coeffs = vec![BigRational::zero(); len];

        for i in 0..len {
            let c1 = self.coeffs.get(i).cloned().unwrap_or_else(BigRational::zero);
            let c2 = other.coeffs.get(i).cloned().unwrap_or_else(BigRational::zero);
            new_coeffs[i] = c1 + c2;
        }

        Polynomial::new(new_coeffs, self.var.clone())
    }

    pub fn sub(&self, other: &Self) -> Self {
        let len = max(self.coeffs.len(), other.coeffs.len());
        let mut new_coeffs = vec![BigRational::zero(); len];

        for i in 0..len {
            let c1 = self.coeffs.get(i).cloned().unwrap_or_else(BigRational::zero);
            let c2 = other.coeffs.get(i).cloned().unwrap_or_else(BigRational::zero);
            new_coeffs[i] = c1 - c2;
        }

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

    // Returns (quotient, remainder)
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        if divisor.is_zero() {
            panic!("Division by zero polynomial");
        }

        if self.degree() < divisor.degree() {
            return (Polynomial::zero(self.var.clone()), self.clone());
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

        (quotient, remainder)
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.clone();
        let mut b = other.clone();

        while !b.is_zero() {
            let (_, r) = a.div_rem(&b);
            a = b;
            b = r;
        }

        // Normalize GCD to have leading coefficient 1 (monic)
        if !a.is_zero() {
            let lc = a.leading_coeff();
            let inv_lc = BigRational::one() / lc;
            let scalar = Polynomial::new(vec![inv_lc], a.var.clone());
            a = a.mul(&scalar);
        }

        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_poly_ops() {
        let x = Polynomial::new(vec![BigRational::zero(), BigRational::one()], "x".to_string()); // x
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
        // (x^2 + 2x + 1) / (x + 1) = x + 1, rem 0
        let p_num = Polynomial::from_expr(&parse("x^2 + 2*x + 1").unwrap(), "x").unwrap();
        let p_den = Polynomial::from_expr(&parse("x + 1").unwrap(), "x").unwrap();

        let (q, r) = p_num.div_rem(&p_den);
        assert!(r.is_zero());
        assert_eq!(q.to_expr().to_string(), "x + 1"); // Display order is x + 1
    }

    #[test]
    fn test_gcd() {
        // gcd(x^2 - 1, x^2 + 2x + 1) = x + 1
        let p1 = Polynomial::from_expr(&parse("x^2 - 1").unwrap(), "x").unwrap();
        let p2 = Polynomial::from_expr(&parse("x^2 + 2*x + 1").unwrap(), "x").unwrap();

        let g = p1.gcd(&p2);
        // Should be x + 1 (normalized)
        // coeffs: 1, 1
        assert_eq!(g.coeffs.len(), 2);
        assert_eq!(g.coeffs[0], BigRational::one());
        assert_eq!(g.coeffs[1], BigRational::one());
    }
}
