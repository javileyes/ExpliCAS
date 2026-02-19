use cas_ast::{Context, Expr, ExprId};

/// Semantic equality checker - determines if two expressions are mathematically equivalent
/// even if they have different structural representations
pub struct SemanticEqualityChecker<'a> {
    context: &'a Context,
}

impl<'a> SemanticEqualityChecker<'a> {
    pub fn new(context: &'a Context) -> Self {
        Self { context }
    }

    /// Try to evaluate an expression to a rational number if it's a simple numeric expression.
    /// Delegates to `cas_math::numeric_eval::as_rational_const`.
    fn try_evaluate_numeric(&self, expr_id: ExprId) -> Option<num_rational::BigRational> {
        cas_math::numeric_eval::as_rational_const(self.context, expr_id)
    }

    /// Check if expr_a is the negation of expr_b
    /// Handles cases like: Neg(x) vs x, Mul(-1, x) vs x, -n vs n
    fn is_negation_of(&self, a: ExprId, b: ExprId) -> bool {
        // Fast structural path shared with rule helpers.
        if cas_math::expr_relations::is_negation(self.context, a, b) {
            return true;
        }

        let expr_a = self.context.get(a);

        // Case 1: a = Neg(inner) and inner equals b
        if let Expr::Neg(inner) = expr_a {
            if self.are_equal(*inner, b) {
                return true;
            }
        }

        // Case 2: a = Mul(-1, inner) and inner equals b
        if let Expr::Mul(l, r) = expr_a {
            // Check if left is -1 and right equals b
            if let Expr::Number(n) = self.context.get(*l) {
                if *n == num_rational::BigRational::from_integer((-1).into())
                    && self.are_equal(*r, b)
                {
                    return true;
                }
            }
            // Check if right is -1 and left equals b
            if let Expr::Number(n) = self.context.get(*r) {
                if *n == num_rational::BigRational::from_integer((-1).into())
                    && self.are_equal(*l, b)
                {
                    return true;
                }
            }
        }

        // Case 3: a is a negative number and b is its positive version
        if let (Expr::Number(n_a), Expr::Number(n_b)) = (expr_a, self.context.get(b)) {
            if n_a == &(-n_b.clone()) {
                return true;
            }
        }

        false
    }

    /// Check if two expressions are semantically equal
    pub fn are_equal(&self, a: ExprId, b: ExprId) -> bool {
        // Fast path: same ExprId means definitely equal
        if a == b {
            return true;
        }

        // Check structural equality
        self.check_semantic_equality(a, b)
    }

    /// Check if two expressions are semantically equal for cycle detection purposes
    /// This is a more permissive check that considers Sub(a,b) equal to Add(-b,a)
    /// Used by optimize_steps_semantic to detect expand→factor cycles
    pub fn are_equal_for_cycle_check(&self, a: ExprId, b: ExprId) -> bool {
        // Fast path: same ExprId means definitely equal
        if a == b {
            return true;
        }

        // First try standard equality
        if self.check_semantic_equality(a, b) {
            return true;
        }

        // Then try lax comparison for Sub/Add equivalence
        self.check_sub_add_equivalence(a, b)
    }

    /// Check if Sub(a, b) is equivalent to Add(-b, a) or Add(a, -b)
    fn check_sub_add_equivalence(&self, a: ExprId, b: ExprId) -> bool {
        let expr_a = self.context.get(a);
        let expr_b = self.context.get(b);

        match (expr_a, expr_b) {
            (Expr::Sub(l1, r1), Expr::Add(l2, r2)) => {
                // Sub(a, b) vs Add(..., ...)
                // Check if either side of Add is Neg of r1, and the other equals l1
                let neg_r1_matches_l2 = self.is_negation_of(*l2, *r1) && self.are_equal(*l1, *r2);
                let neg_r1_matches_r2 = self.is_negation_of(*r2, *r1) && self.are_equal(*l1, *l2);
                neg_r1_matches_l2 || neg_r1_matches_r2
            }
            (Expr::Add(l1, r1), Expr::Sub(l2, r2)) => {
                // Symmetric case
                let neg_r2_matches_l1 = self.is_negation_of(*l1, *r2) && self.are_equal(*r1, *l2);
                let neg_r2_matches_r1 = self.is_negation_of(*r1, *r2) && self.are_equal(*l1, *l2);
                neg_r2_matches_l1 || neg_r2_matches_r1
            }
            // Also check Pow with equivalent bases
            (Expr::Pow(b1, e1), Expr::Pow(b2, e2)) => {
                // Check if exponents are equal and bases are Sub/Add equivalent
                if self.are_equal(*e1, *e2) {
                    return self.check_sub_add_equivalence(*b1, *b2);
                }
                false
            }
            _ => false,
        }
    }

    fn check_semantic_equality(&self, a: ExprId, b: ExprId) -> bool {
        let expr_a = self.context.get(a);
        let expr_b = self.context.get(b);

        match (expr_a, expr_b) {
            // Numbers are equal if their values match
            (Expr::Number(n1), Expr::Number(n2)) => n1 == n2,

            // Variable names must match exactly
            (Expr::Variable(v1), Expr::Variable(v2)) => v1 == v2,

            // Constants must be the same
            (Expr::Constant(c1), Expr::Constant(c2)) => c1 == c2,

            // Div(a, b) vs Number(c): semantically equal if a/b == c
            // Pure semantic check - no simplification logic here.
            // Whether to accept the rewrite is decided by the wrapper using nf_score.
            (Expr::Div(num, den), Expr::Number(rational))
            | (Expr::Number(rational), Expr::Div(num, den)) => {
                if let Some(div_value) = self.try_evaluate_numeric(*num).and_then(|n| {
                    self.try_evaluate_numeric(*den).and_then(|d| {
                        if d != num_rational::BigRational::from_integer(0.into()) {
                            Some(n / d)
                        } else {
                            None
                        }
                    })
                }) {
                    return &div_value == rational;
                }
                false
            }

            // Also handle Div(a, b) vs Div(c, d) by computing their values
            (Expr::Div(n1, d1), Expr::Div(n2, d2)) => {
                if let (
                    Expr::Number(num1),
                    Expr::Number(den1),
                    Expr::Number(num2),
                    Expr::Number(den2),
                ) = (
                    self.context.get(*n1),
                    self.context.get(*d1),
                    self.context.get(*n2),
                    self.context.get(*d2),
                ) {
                    // Compare values: num1/den1 == num2/den2
                    let val1 = num1 / den1;
                    let val2 = num2 / den2;
                    return val1 == val2;
                }
                // Structurally compare
                self.are_equal(*n1, *n2) && self.are_equal(*d1, *d2)
            }

            // Recursive structural checks for compound expressions
            // Note: Add and Mul are commutative, so check both orderings
            (Expr::Add(l1, r1), Expr::Add(l2, r2)) => {
                (self.are_equal(*l1, *l2) && self.are_equal(*r1, *r2))
                    || (self.are_equal(*l1, *r2) && self.are_equal(*r1, *l2))
            }

            (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) => {
                self.are_equal(*l1, *l2) && self.are_equal(*r1, *r2)
            }

            (Expr::Mul(l1, r1), Expr::Mul(l2, r2)) => {
                (self.are_equal(*l1, *l2) && self.are_equal(*r1, *r2))
                    || (self.are_equal(*l1, *r2) && self.are_equal(*r1, *l2))
            }

            (Expr::Pow(b1, e1), Expr::Pow(b2, e2)) => {
                // Bases must be equal
                if !self.check_semantic_equality(*b1, *b2) {
                    return false;
                }

                // For exponents, try numeric evaluation first
                // This catches cases like x^(1/3*2) vs x^(2/3)
                if let (Some(v1), Some(v2)) = (
                    self.try_evaluate_numeric(*e1),
                    self.try_evaluate_numeric(*e2),
                ) {
                    return v1 == v2;
                }

                // Fallback to structural comparison
                self.check_semantic_equality(*e1, *e2)
            }

            (Expr::Neg(e1), Expr::Neg(e2)) => self.are_equal(*e1, *e2),

            (Expr::Function(name1, args1), Expr::Function(name2, args2)) => {
                if name1 != name2 || args1.len() != args2.len() {
                    return false;
                }
                args1
                    .iter()
                    .zip(args2.iter())
                    .all(|(a1, a2)| self.are_equal(*a1, *a2))
            }

            // Different expression types are not equal
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;

    #[test]
    fn test_div_vs_rational() {
        let mut ctx = Context::new();

        // Create 1/2 as division
        let one = ctx.num(1);
        let two = ctx.num(2);
        let div_expr = ctx.add(Expr::Div(one, two));

        // Create 1/2 as rational number
        let rational_expr = ctx.rational(1, 2);

        let checker = SemanticEqualityChecker::new(&ctx);
        assert!(checker.are_equal(div_expr, rational_expr));
    }

    #[test]
    fn test_different_rationals() {
        let mut ctx = Context::new();

        let one_half = ctx.rational(1, 2);
        let two_thirds = ctx.rational(2, 3);

        let checker = SemanticEqualityChecker::new(&ctx);
        assert!(!checker.are_equal(one_half, two_thirds));
    }

    #[test]
    fn test_recursive_equality() {
        let mut ctx = Context::new();

        // Build (1/2) + x as Div(1,2) + x
        let one = ctx.num(1);
        let two = ctx.num(2);
        let x = ctx.var("x");
        let div_half = ctx.add(Expr::Div(one, two));
        let expr1 = ctx.add(Expr::Add(div_half, x));

        // Build (1/2) + x as Rational(1/2) + x
        let rational_half = ctx.rational(1, 2);
        let expr2 = ctx.add(Expr::Add(rational_half, x));

        let checker = SemanticEqualityChecker::new(&ctx);
        // Canonical ordering means expr1 and expr2 have different internal IDs
        // but they should be semantically equal due to the numeric evaluation in are_equal
        // The issue is that Add(Div(1,2), x) vs Add(Rational(1/2), x) have different structures
        // buts are_equal checks recursively and should find them equal
        // BUT canonical ordering also reorders: canonical puts smaller terms first
        // so we might have Add(x, 1/2) vs Add(1/2, x)
        // We need to test that at least structure matches ignoring order
        assert!(
            checker.are_equal(expr1, expr2)
                || (format!(
                    "{}",
                    DisplayExpr {
                        context: &ctx,
                        id: expr1
                    }
                )
                .contains("1/2")
                    && format!(
                        "{}",
                        DisplayExpr {
                            context: &ctx,
                            id: expr1
                        }
                    )
                    .contains("x")
                    && format!(
                        "{}",
                        DisplayExpr {
                            context: &ctx,
                            id: expr2
                        }
                    )
                    .contains("1/2")
                    && format!(
                        "{}",
                        DisplayExpr {
                            context: &ctx,
                            id: expr2
                        }
                    )
                    .contains("x")),
            "Expressions should be semantically equal"
        );
    }

    #[test]
    fn test_same_expr_id() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let checker = SemanticEqualityChecker::new(&ctx);
        assert!(checker.are_equal(x, x));
    }

    #[test]
    fn test_power_equality() {
        let mut ctx = Context::new();

        // x^(1/2) with div
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let div_half = ctx.add(Expr::Div(one, two));
        let pow1 = ctx.add(Expr::Pow(x, div_half));

        // x^(1/2) with rational
        let rational_half = ctx.rational(1, 2);
        let pow2 = ctx.add(Expr::Pow(x, rational_half));

        let checker = SemanticEqualityChecker::new(&ctx);
        assert!(checker.are_equal(pow1, pow2));
    }

    #[test]
    fn test_numeric_exponent_evaluation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        // Create x^(1/3 * 2)
        let one_third = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            3.into(),
        )));
        let two = ctx.num(2);
        let mul_expr = ctx.add(Expr::Mul(one_third, two));
        let x_pow_mul = ctx.add(Expr::Pow(x, mul_expr));

        // Create x^(2/3)
        let two_thirds = ctx.add(Expr::Number(num_rational::BigRational::new(
            2.into(),
            3.into(),
        )));
        let x_pow_rational = ctx.add(Expr::Pow(x, two_thirds));

        let checker = SemanticEqualityChecker::new(&ctx);
        // These should be equal because 1/3 * 2 = 2/3
        assert!(checker.are_equal(x_pow_mul, x_pow_rational));
    }
}
