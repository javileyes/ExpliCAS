//! Session environment for variable bindings.
//!
//! Provides an `Environment` struct that stores variable → expression mappings,
//! and `substitute()` functions to replace variables in expressions.

use std::collections::{HashMap, HashSet};

use cas_ast::{Context, Expr, ExprId};

/// Maximum substitution depth to prevent stack overflow on long chains
const MAX_SUBSTITUTE_DEPTH: usize = 100;

/// Reserved names that cannot be assigned (keywords, built-in functions, constants)
const RESERVED_NAMES: &[&str] = &[
    // Keywords
    "let",
    "vars",
    "clear",
    "reset",
    "simplify",
    "expand",
    "rationalize",
    "factor",
    "collect",
    "solve",
    "diff",
    "integrate",
    "limit",
    "sum",
    "product",
    "timeline",
    "explain",
    "trace",
    "profile",
    "help",
    "quit",
    "exit",
    // Built-in functions
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "asin",
    "acos",
    "atan",
    "acot",
    "asec",
    "acsc",
    "sinh",
    "cosh",
    "tanh",
    "coth",
    "sech",
    "csch",
    "asinh",
    "acosh",
    "atanh",
    "acoth",
    "asech",
    "acsch",
    "ln",
    "log",
    "exp",
    "sqrt",
    "abs",
    "sign",
    "floor",
    "ceil",
    "round",
    "gcd",
    "lcm",
    "mod",
    "det",
    "transpose",
    "fact",
    "choose",
    "perm",
    // Constants
    "pi",
    "e",
    "i",
    "inf",
    "infinity",
    "undefined",
];

/// Check if a name is reserved and cannot be assigned
pub fn is_reserved(name: &str) -> bool {
    RESERVED_NAMES.contains(&name.to_lowercase().as_str())
}

/// Session environment holding variable → expression bindings
/// Storage for variable bindings
#[derive(Default, Debug, Clone)]
pub struct Environment {
    bindings: HashMap<String, ExprId>,
}

impl Environment {
    /// Create a new empty environment
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a binding (overwrites existing)
    pub fn set(&mut self, name: String, expr: ExprId) {
        self.bindings.insert(name, expr);
    }

    /// Get a binding by name
    pub fn get(&self, name: &str) -> Option<ExprId> {
        self.bindings.get(name).copied()
    }

    /// Remove a binding, returns true if it existed
    pub fn unset(&mut self, name: &str) -> bool {
        self.bindings.remove(name).is_some()
    }

    /// Clear all bindings
    pub fn clear_all(&mut self) {
        self.bindings.clear();
    }

    /// List all bindings, sorted by name for deterministic output
    pub fn list(&self) -> Vec<(&str, ExprId)> {
        let mut items: Vec<_> = self
            .bindings
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        items.sort_by_key(|(name, _)| *name);
        items
    }

    /// Check if a binding exists
    pub fn contains(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    /// Number of bindings
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Check if environment is empty
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

/// Substitute variables in an expression using environment bindings.
///
/// Features:
/// - Transitive closure: if `a → b+1` and `b → 3`, then `a` becomes `4`
/// - Cycle detection: `a → a+1` won't cause infinite loop
/// - Depth limit: prevents stack overflow on long chains
pub fn substitute(ctx: &mut Context, env: &Environment, expr: ExprId) -> ExprId {
    substitute_with_shadow(ctx, env, expr, &[])
}

/// Substitute variables but DON'T expand variables in `shadow` list.
///
/// Used for commands like `diff expr, x` where `x` is the differentiation
/// variable and should NOT be substituted even if bound in environment.
pub fn substitute_with_shadow(
    ctx: &mut Context,
    env: &Environment,
    expr: ExprId,
    shadow: &[&str],
) -> ExprId {
    let shadow_set: HashSet<&str> = shadow.iter().copied().collect();
    substitute_impl(ctx, env, expr, &shadow_set, &mut HashSet::new(), 0)
}

/// Internal recursive implementation with cycle detection
fn substitute_impl(
    ctx: &mut Context,
    env: &Environment,
    expr: ExprId,
    shadow: &HashSet<&str>,
    visiting: &mut HashSet<String>,
    depth: usize,
) -> ExprId {
    // Depth limit to prevent stack overflow
    if depth > MAX_SUBSTITUTE_DEPTH {
        return expr;
    }

    let expr_data = ctx.get(expr).clone();

    match expr_data {
        Expr::Variable(sym_id) => {
            // Resolve symbol to string for Environment lookup
            let name = ctx.sym_name(sym_id).to_string();

            // Don't substitute shadowed variables
            if shadow.contains(name.as_str()) {
                return expr;
            }

            // Check for binding
            if let Some(bound_expr) = env.get(&name) {
                // Cycle detection: if we're already visiting this variable, stop
                if visiting.contains(&name) {
                    return expr; // Leave as Var(name) to break cycle
                }

                // Mark as visiting and recurse (transitive closure)
                visiting.insert(name.clone());
                let result = substitute_impl(ctx, env, bound_expr, shadow, visiting, depth + 1);
                visiting.remove(&name);
                result
            } else {
                expr // No binding, leave unchanged
            }
        }

        // Recursively substitute in compound expressions
        Expr::Neg(a) => {
            let new_a = substitute_impl(ctx, env, a, shadow, visiting, depth + 1);
            if new_a == a {
                expr
            } else {
                ctx.add(Expr::Neg(new_a))
            }
        }

        Expr::Add(a, b) => {
            let new_a = substitute_impl(ctx, env, a, shadow, visiting, depth + 1);
            let new_b = substitute_impl(ctx, env, b, shadow, visiting, depth + 1);
            if new_a == a && new_b == b {
                expr
            } else {
                ctx.add(Expr::Add(new_a, new_b))
            }
        }

        Expr::Sub(a, b) => {
            let new_a = substitute_impl(ctx, env, a, shadow, visiting, depth + 1);
            let new_b = substitute_impl(ctx, env, b, shadow, visiting, depth + 1);
            if new_a == a && new_b == b {
                expr
            } else {
                ctx.add(Expr::Sub(new_a, new_b))
            }
        }

        Expr::Mul(a, b) => {
            let new_a = substitute_impl(ctx, env, a, shadow, visiting, depth + 1);
            let new_b = substitute_impl(ctx, env, b, shadow, visiting, depth + 1);
            if new_a == a && new_b == b {
                expr
            } else {
                ctx.add(Expr::Mul(new_a, new_b))
            }
        }

        Expr::Div(a, b) => {
            let new_a = substitute_impl(ctx, env, a, shadow, visiting, depth + 1);
            let new_b = substitute_impl(ctx, env, b, shadow, visiting, depth + 1);
            if new_a == a && new_b == b {
                expr
            } else {
                ctx.add(Expr::Div(new_a, new_b))
            }
        }

        Expr::Pow(a, b) => {
            let new_a = substitute_impl(ctx, env, a, shadow, visiting, depth + 1);
            let new_b = substitute_impl(ctx, env, b, shadow, visiting, depth + 1);
            if new_a == a && new_b == b {
                expr
            } else {
                ctx.add(Expr::Pow(new_a, new_b))
            }
        }

        Expr::Function(name, args) => {
            let mut changed = false;
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|&arg| {
                    let new_arg = substitute_impl(ctx, env, arg, shadow, visiting, depth + 1);
                    if new_arg != arg {
                        changed = true;
                    }
                    new_arg
                })
                .collect();
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }

        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let new_data: Vec<ExprId> = data
                .iter()
                .map(|&elem| {
                    let new_elem = substitute_impl(ctx, env, elem, shadow, visiting, depth + 1);
                    if new_elem != elem {
                        changed = true;
                    }
                    new_elem
                })
                .collect();
            if changed {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                expr
            }
        }

        // Literals: no substitution needed
        // SessionRef should be resolved before substitution, so leave unchanged
        Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => expr,

        // Hold: substitute inside but preserve the wrapper
        Expr::Hold(inner) => {
            let new_inner = substitute_impl(ctx, env, inner, shadow, visiting, depth + 1);
            if new_inner == inner {
                expr
            } else {
                ctx.add(Expr::Hold(new_inner))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_substitute_simple() {
        let mut ctx = Context::new();
        let mut env = Environment::new();

        // a = 2
        let two = ctx.num(2);
        env.set("a".to_string(), two);

        // a + 1
        let expr = parse("a + 1", &mut ctx).unwrap();

        // substitute → 2 + 1 (or 1 + 2 due to ordering)
        let result = substitute(&mut ctx, &env, expr);
        let result_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: result
            }
        );
        // Add is commutative, check both combinations
        assert!(
            result_str == "2 + 1" || result_str == "1 + 2",
            "Got: {}",
            result_str
        );
    }

    #[test]
    fn test_substitute_transitive() {
        let mut ctx = Context::new();
        let mut env = Environment::new();

        // b = 3
        let three = ctx.num(3);
        env.set("b".to_string(), three);

        // a = b + 1
        let a_expr = parse("b + 1", &mut ctx).unwrap();
        env.set("a".to_string(), a_expr);

        // a * 2 → (3 + 1) * 2
        let expr = parse("a * 2", &mut ctx).unwrap();
        let result = substitute(&mut ctx, &env, expr);
        let result_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: result
            }
        );
        assert!(result_str.contains("3") && result_str.contains("2"));
    }

    #[test]
    fn test_cycle_direct_no_hang() {
        let mut ctx = Context::new();
        let mut env = Environment::new();

        // a = a + 1 (direct cycle)
        let a_expr = parse("a + 1", &mut ctx).unwrap();
        env.set("a".to_string(), a_expr);

        // This should NOT hang - cycle detection should break it
        let expr = parse("a * 2", &mut ctx).unwrap();
        let _result = substitute(&mut ctx, &env, expr);
        // If we got here, no infinite loop - test passes
    }

    #[test]
    fn test_cycle_indirect_no_hang() {
        let mut ctx = Context::new();
        let mut env = Environment::new();

        // a = b + 1
        let a_expr = parse("b + 1", &mut ctx).unwrap();
        env.set("a".to_string(), a_expr);

        // b = a + 1 (indirect cycle)
        let b_expr = parse("a + 1", &mut ctx).unwrap();
        env.set("b".to_string(), b_expr);

        // This should NOT hang
        let expr = parse("a * 2", &mut ctx).unwrap();
        let _result = substitute(&mut ctx, &env, expr);
        // If we got here, no infinite loop - test passes
    }

    #[test]
    fn test_shadow_variable() {
        let mut ctx = Context::new();
        let mut env = Environment::new();

        // x = 3
        let three = ctx.num(3);
        env.set("x".to_string(), three);

        // x^2 with shadow ["x"] → should remain x^2, not 9
        let expr = parse("x^2", &mut ctx).unwrap();
        let result = substitute_with_shadow(&mut ctx, &env, expr, &["x"]);
        let result_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: result
            }
        );
        assert!(
            result_str.contains("x"),
            "x should be shadowed, got: {}",
            result_str
        );
    }

    #[test]
    fn test_is_reserved() {
        assert!(is_reserved("let"));
        assert!(is_reserved("sin"));
        assert!(is_reserved("pi"));
        assert!(is_reserved("Pi")); // case insensitive
        assert!(!is_reserved("myvar"));
        assert!(!is_reserved("a"));
    }

    #[test]
    fn test_env_operations() {
        let mut env = Environment::new();
        let mut ctx = Context::new();
        let expr_id = ctx.num(0);

        env.set("a".to_string(), expr_id);
        assert!(env.contains("a"));
        assert_eq!(env.get("a"), Some(expr_id));
        assert_eq!(env.len(), 1);

        env.unset("a");
        assert!(!env.contains("a"));
        assert!(env.is_empty());
    }
}
