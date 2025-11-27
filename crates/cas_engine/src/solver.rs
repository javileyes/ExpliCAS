use cas_ast::{Expr, Equation, RelOp};
use std::rc::Rc;

pub fn solve(eq: &Equation, var: &str) -> Result<Equation, String> {
    // We want to isolate 'var' on LHS.
    // Current strategy:
    // 1. Check if LHS contains var. If not, swap LHS and RHS (if RHS has var).
    // 2. If both have var, we need to collect terms (not implemented yet in solver, assume simplified).
    // 3. Peel operations from LHS until var is isolated.

    let lhs_has_var = contains_var(&eq.lhs, var);
    let rhs_has_var = contains_var(&eq.rhs, var);

    if !lhs_has_var && !rhs_has_var {
        return Err(format!("Variable '{}' not found in equation", var));
    }

    if !lhs_has_var && rhs_has_var {
        // Swap to make LHS have the variable
        // a = x -> x = a
        // a < x -> x > a
        let new_op = match eq.op {
            RelOp::Eq => RelOp::Eq,
            RelOp::Neq => RelOp::Neq,
            RelOp::Lt => RelOp::Gt,
            RelOp::Gt => RelOp::Lt,
            RelOp::Leq => RelOp::Geq,
            RelOp::Geq => RelOp::Leq,
        };
        return solve(&Equation { lhs: eq.rhs.clone(), rhs: eq.lhs.clone(), op: new_op }, var);
    }

    if lhs_has_var && rhs_has_var {
        // Move all terms to LHS? 
        // For MVP, let's assume linear/simple equations where we can just subtract RHS from LHS?
        // Or just fail for now if variable is on both sides (requires 'collect').
        // Let's try to proceed with peeling LHS, but if we encounter var on RHS during peeling, it's tricky.
        // Better approach for "both sides": Move everything to LHS: LHS - RHS = 0. Then solve.
        // But that requires 'collect'.
        // Let's stick to "peeling" for now. If RHS has var, we might get x = x + 1 which is unsolvable by peeling.
        return Err("Variable appears on both sides. Please simplify/collect first.".to_string());
    }

    // Now LHS has var, RHS does not.
    isolate(&eq.lhs, &eq.rhs, eq.op.clone(), var)
}

fn isolate(lhs: &Rc<Expr>, rhs: &Rc<Expr>, op: RelOp, var: &str) -> Result<Equation, String> {
    match lhs.as_ref() {
        Expr::Variable(v) if v == var => {
            Ok(Equation { lhs: lhs.clone(), rhs: rhs.clone(), op })
        }
        Expr::Add(l, r) => {
            // (A + B) = RHS
            // If var in A: A = RHS - B
            // If var in B: B = RHS - A
            if contains_var(l, var) {
                isolate(l, &Expr::sub(rhs.clone(), r.clone()), op, var)
            } else {
                isolate(r, &Expr::sub(rhs.clone(), l.clone()), op, var)
            }
        }
        Expr::Sub(l, r) => {
            // (A - B) = RHS
            // If var in A: A = RHS + B
            // If var in B: -B = RHS - A -> B = A - RHS (careful with inequalities)
            if contains_var(l, var) {
                isolate(l, &Expr::add(rhs.clone(), r.clone()), op, var)
            } else {
                // -B = RHS - A  =>  B = A - RHS
                // Inequality flips if we multiply by -1
                let new_rhs = Expr::sub(l.clone(), rhs.clone());
                // Or: B = -(RHS - A) = A - RHS.
                // Multiplying by -1 flips inequality.
                let new_op = flip_inequality(op);
                isolate(r, &new_rhs, new_op, var)
            }
        }
        Expr::Mul(l, r) => {
            // A * B = RHS
            // If var in A: A = RHS / B
            // If var in B: B = RHS / A
            // Note: Division by negative number flips inequality. 
            // We don't know sign of A or B easily without evaluation.
            // For MVP, assume positive or ignore inequality flip for variables.
            if contains_var(l, var) {
                isolate(l, &Expr::div(rhs.clone(), r.clone()), op, var)
            } else {
                isolate(r, &Expr::div(rhs.clone(), l.clone()), op, var)
            }
        }
        Expr::Div(l, r) => {
            // A / B = RHS
            // If var in A: A = RHS * B
            // If var in B: B = A / RHS (assuming RHS != 0)
            if contains_var(l, var) {
                // Multiply by B. If B < 0, flip. Unknown sign -> assume positive for now.
                isolate(l, &Expr::mul(rhs.clone(), r.clone()), op, var)
            } else {
                // B = A / RHS
                // Effectively: 1/B = RHS/A -> B = A/RHS
                // Taking reciprocal flips inequality if signs are same (e.g. 2 < 3 -> 1/2 > 1/3).
                // This is getting complex for inequalities.
                // Let's just implement Eq for now or assume standard behavior.
                isolate(r, &Expr::div(l.clone(), rhs.clone()), op, var)
            }
        }
        Expr::Pow(b, e) => {
            // B^E = RHS
            // If var in B: B = RHS^(1/E) (root)
            // If var in E: E = log(B, RHS)
            if contains_var(b, var) {
                // B = RHS^(1/E)
                let inv_exp = Expr::div(Expr::num(1), e.clone());
                isolate(b, &Expr::pow(rhs.clone(), inv_exp), op, var)
            } else {
                // E = log(B, RHS)
                isolate(e, &Expr::log(b.clone(), rhs.clone()), op, var)
            }
        }
        Expr::Function(name, args) if args.len() == 1 => {
            // f(x) = RHS -> x = f_inv(RHS)
            let arg = &args[0];
            if contains_var(arg, var) {
                match name.as_str() {
                    "ln" => isolate(arg, &Expr::pow(Expr::e(), rhs.clone()), op, var),
                    "exp" => isolate(arg, &Expr::ln(rhs.clone()), op, var),
                    "sqrt" => isolate(arg, &Expr::pow(rhs.clone(), Expr::num(2)), op, var),
                    // Add trig inverses later
                    _ => Err(format!("Cannot invert function '{}'", name)),
                }
            } else {
                 Err(format!("Variable '{}' not found in function argument", var))
            }
        }
        _ => Err(format!("Cannot isolate '{}' from {:?}", var, lhs)),
    }
}

pub fn contains_var(expr: &Rc<Expr>, var: &str) -> bool {
    match expr.as_ref() {
        Expr::Variable(v) => v == var,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_var(l, var) || contains_var(r, var)
        }
        Expr::Neg(e) => contains_var(e, var),
        Expr::Function(_, args) => args.iter().any(|a| contains_var(a, var)),
        _ => false,
    }
}

fn flip_inequality(op: RelOp) -> RelOp {
    match op {
        RelOp::Eq => RelOp::Eq,
        RelOp::Neq => RelOp::Neq,
        RelOp::Lt => RelOp::Gt,
        RelOp::Gt => RelOp::Lt,
        RelOp::Leq => RelOp::Geq,
        RelOp::Geq => RelOp::Leq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse; // Note: parse returns Expr, we need to construct Equation manually for tests or update parser
    // Since we updated parser but it's in another crate, we might need to use parse_statement if available or construct manually.
    // For unit tests inside cas_engine, we depend on cas_parser? Yes.
    
    // Helper to make equation from strings
    fn make_eq(lhs: &str, rhs: &str) -> Equation {
        Equation {
            lhs: parse(lhs).unwrap(),
            rhs: parse(rhs).unwrap(),
            op: RelOp::Eq,
        }
    }

    #[test]
    fn test_solve_linear() {
        // x + 2 = 5 -> x = 3
        let eq = make_eq("x + 2", "5");
        let res = solve(&eq, "x").unwrap();
        // Result: x = 5 - 2
        // We don't simplify automatically in solve, so RHS is "5 - 2".
        assert_eq!(format!("{}", res.lhs), "x");
        assert_eq!(format!("{}", res.rhs), "5 - 2");
    }

    #[test]
    fn test_solve_mul() {
        // 2 * x = 6 -> x = 6 / 2
        let eq = make_eq("2 * x", "6");
        let res = solve(&eq, "x").unwrap();
        assert_eq!(format!("{}", res.rhs), "6 / 2");
    }
    
    #[test]
    fn test_solve_pow() {
        // x^2 = 4 -> x = 4^(1/2)
        let eq = make_eq("x^2", "4");
        let res = solve(&eq, "x").unwrap();
        assert_eq!(format!("{}", res.rhs), "4^(1 / 2)");
    }
}
