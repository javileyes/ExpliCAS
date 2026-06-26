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
    "taylor",
    "series",
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
    "trace",
    "rank",
    "inverse",
    "inv",
    "fact",
    "choose",
    "perm",
    // Constants
    "pi",
    "e",
    "i",
    "inf",
    "oo",
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
    functions: HashMap<String, FunctionBinding>,
}

/// User-defined function binding stored in the session environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionBinding {
    pub params: Vec<String>,
    pub expr: ExprId,
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

    /// Set a function binding (overwrites existing).
    pub fn set_function(&mut self, name: String, params: Vec<String>, expr: ExprId) {
        self.functions
            .insert(name, FunctionBinding { params, expr });
    }

    /// Get a function binding by name.
    pub fn get_function(&self, name: &str) -> Option<&FunctionBinding> {
        self.functions.get(name)
    }

    /// Remove a function binding, returns true if it existed.
    pub fn unset_function(&mut self, name: &str) -> bool {
        self.functions.remove(name).is_some()
    }

    /// Clear all bindings
    pub fn clear_all(&mut self) {
        self.bindings.clear();
        self.functions.clear();
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

    /// List all function bindings, sorted by name for deterministic output.
    pub fn list_functions(&self) -> Vec<(&str, &FunctionBinding)> {
        let mut items: Vec<_> = self
            .functions
            .iter()
            .map(|(k, v)| (k.as_str(), v))
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
        self.bindings.is_empty() && self.functions.is_empty()
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

fn substitute_named_var(
    ctx: &mut Context,
    expr: ExprId,
    name: &str,
    replacement: ExprId,
) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == name => replacement,
        Expr::Neg(inner) => {
            let new_inner = substitute_named_var(ctx, inner, name, replacement);
            if new_inner == inner {
                expr
            } else {
                ctx.add(Expr::Neg(new_inner))
            }
        }
        Expr::Add(a, b) => rewrite_binary_named_var(ctx, expr, a, b, name, replacement, Expr::Add),
        Expr::Sub(a, b) => rewrite_binary_named_var(ctx, expr, a, b, name, replacement, Expr::Sub),
        Expr::Mul(a, b) => rewrite_binary_named_var(ctx, expr, a, b, name, replacement, Expr::Mul),
        Expr::Div(a, b) => rewrite_binary_named_var(ctx, expr, a, b, name, replacement, Expr::Div),
        Expr::Pow(a, b) => rewrite_binary_named_var(ctx, expr, a, b, name, replacement, Expr::Pow),
        Expr::Function(fn_id, args) => {
            let mut changed = false;
            let new_args: Vec<_> = args
                .iter()
                .map(|&arg| {
                    let new_arg = substitute_named_var(ctx, arg, name, replacement);
                    if new_arg != arg {
                        changed = true;
                    }
                    new_arg
                })
                .collect();
            if changed {
                ctx.add(Expr::Function(fn_id, new_args))
            } else {
                expr
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let new_data: Vec<_> = data
                .iter()
                .map(|&elem| {
                    let new_elem = substitute_named_var(ctx, elem, name, replacement);
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
        Expr::Hold(inner) => {
            let new_inner = substitute_named_var(ctx, inner, name, replacement);
            if new_inner == inner {
                expr
            } else {
                ctx.add(Expr::Hold(new_inner))
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
    }
}

fn rewrite_binary_named_var(
    ctx: &mut Context,
    original: ExprId,
    a: ExprId,
    b: ExprId,
    name: &str,
    replacement: ExprId,
    build: fn(ExprId, ExprId) -> Expr,
) -> ExprId {
    let new_a = substitute_named_var(ctx, a, name, replacement);
    let new_b = substitute_named_var(ctx, b, name, replacement);
    if new_a == a && new_b == b {
        original
    } else {
        ctx.add(build(new_a, new_b))
    }
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
            let fn_name = ctx.sym_name(name).to_string();
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

            if shadow.contains(fn_name.as_str()) {
                return if changed {
                    ctx.add(Expr::Function(name, new_args))
                } else {
                    expr
                };
            }

            if let Some(function) = env.get_function(&fn_name) {
                if function.params.len() == new_args.len() {
                    let visit_key = format!("fn:{}/{}", fn_name, new_args.len());
                    if visiting.contains(&visit_key) {
                        return if changed {
                            ctx.add(Expr::Function(name, new_args))
                        } else {
                            expr
                        };
                    }

                    visiting.insert(visit_key.clone());

                    let mut body = function.expr;
                    for (param, arg) in function.params.iter().zip(new_args.iter().copied()) {
                        body = substitute_named_var(ctx, body, param, arg);
                    }

                    let resolved = substitute_impl(ctx, env, body, shadow, visiting, depth + 1);
                    visiting.remove(&visit_key);
                    return resolved;
                }
            }

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
mod tests;
