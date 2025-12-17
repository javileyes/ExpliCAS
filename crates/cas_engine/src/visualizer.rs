use cas_ast::{Constant, Context, Expr, ExprId};
use std::collections::HashSet;

/// AST Visualizer - exports expression trees to Graphviz DOT format
pub struct AstVisualizer<'a> {
    context: &'a Context,
    visited: HashSet<ExprId>,
    node_counter: usize,
}

impl<'a> AstVisualizer<'a> {
    pub fn new(context: &'a Context) -> Self {
        Self {
            context,
            visited: HashSet::new(),
            node_counter: 0,
        }
    }

    /// Export expression tree to Graphviz DOT format
    pub fn to_dot(&mut self, expr: ExprId) -> String {
        self.visited.clear();
        self.node_counter = 0;

        let mut output = String::from("digraph AST {\n");
        output.push_str("  node [shape=box, style=\"rounded,filled\", fontname=\"Arial\"];\n");
        output.push_str("  edge [arrowsize=0.7];\n");
        output.push_str("  rankdir=TB;\n");
        output.push_str("  bgcolor=\"#f8f8f8\";\n\n");

        self.write_node(&mut output, expr);

        output.push_str("}\n");
        output
    }

    fn write_node(&mut self, output: &mut String, id: ExprId) -> usize {
        // Check if already visited (handle shared subexpressions)
        if self.visited.contains(&id) {
            // Find existing node id
            // For simplicity, we'll just create a new one
            // In a more sophisticated version, we'd track node_id mappings
        }
        self.visited.insert(id);

        let node_id = self.node_counter;
        self.node_counter += 1;

        let expr = self.context.get(id).clone();

        match expr {
            Expr::Number(n) => {
                output.push_str(&format!(
                    "  n{} [label=\"{}\", shape=circle, fillcolor=\"#e3f2fd\"];\n",
                    node_id, n
                ));
            }
            Expr::Variable(v) => {
                output.push_str(&format!(
                    "  n{} [label=\"{}\", shape=ellipse, fillcolor=\"#c8e6c9\"];\n",
                    node_id, v
                ));
            }
            Expr::Constant(c) => {
                let label = match c {
                    Constant::Pi => "π",
                    Constant::E => "e",
                    Constant::Infinity => "∞",
                    Constant::Undefined => "?",
                    Constant::I => "i",
                };
                output.push_str(&format!(
                    "  n{} [label=\"{}\", shape=diamond, fillcolor=\"#fff9c4\"];\n",
                    node_id, label
                ));
            }
            Expr::Add(l, r) => {
                output.push_str(&format!(
                    "  n{} [label=\"+\", fillcolor=\"#fff3e0\"];\n",
                    node_id
                ));
                let l_id = self.write_node(output, l);
                let r_id = self.write_node(output, r);
                output.push_str(&format!("  n{} -> n{} [label=\"L\"];\n", node_id, l_id));
                output.push_str(&format!("  n{} -> n{} [label=\"R\"];\n", node_id, r_id));
            }
            Expr::Sub(l, r) => {
                output.push_str(&format!(
                    "  n{} [label=\"−\", fillcolor=\"#fff3e0\"];\n",
                    node_id
                ));
                let l_id = self.write_node(output, l);
                let r_id = self.write_node(output, r);
                output.push_str(&format!("  n{} -> n{} [label=\"L\"];\n", node_id, l_id));
                output.push_str(&format!("  n{} -> n{} [label=\"R\"];\n", node_id, r_id));
            }
            Expr::Mul(l, r) => {
                output.push_str(&format!(
                    "  n{} [label=\"×\", fillcolor=\"#ffe0b2\"];\n",
                    node_id
                ));
                let l_id = self.write_node(output, l);
                let r_id = self.write_node(output, r);
                output.push_str(&format!("  n{} -> n{} [label=\"L\"];\n", node_id, l_id));
                output.push_str(&format!("  n{} -> n{} [label=\"R\"];\n", node_id, r_id));
            }
            Expr::Div(l, r) => {
                output.push_str(&format!(
                    "  n{} [label=\"÷\", fillcolor=\"#ffe0b2\"];\n",
                    node_id
                ));
                let l_id = self.write_node(output, l);
                let r_id = self.write_node(output, r);
                output.push_str(&format!("  n{} -> n{} [label=\"L\"];\n", node_id, l_id));
                output.push_str(&format!("  n{} -> n{} [label=\"R\"];\n", node_id, r_id));
            }
            Expr::Pow(b, e) => {
                output.push_str(&format!(
                    "  n{} [label=\"^\", fillcolor=\"#f8bbd0\"];\n",
                    node_id
                ));
                let b_id = self.write_node(output, b);
                let e_id = self.write_node(output, e);
                output.push_str(&format!("  n{} -> n{} [label=\"base\"];\n", node_id, b_id));
                output.push_str(&format!("  n{} -> n{} [label=\"exp\"];\n", node_id, e_id));
            }
            Expr::Neg(e) => {
                output.push_str(&format!(
                    "  n{} [label=\"−\", shape=triangle, fillcolor=\"#ffccbc\"];\n",
                    node_id
                ));
                let e_id = self.write_node(output, e);
                output.push_str(&format!("  n{} -> n{};\n", node_id, e_id));
            }
            Expr::Function(name, args) => {
                let escaped_name = name.replace("\"", "\\\"");
                output.push_str(&format!(
                    "  n{} [label=\"{}()\", shape=hexagon, fillcolor=\"#d1c4e9\"];\n",
                    node_id, escaped_name
                ));
                for (i, arg) in args.iter().enumerate() {
                    let arg_id = self.write_node(output, *arg);
                    output.push_str(&format!(
                        "  n{} -> n{} [label=\"arg{}\"];\n",
                        node_id, arg_id, i
                    ));
                }
            }
            Expr::Matrix { rows, cols, data } => {
                output.push_str(&format!(
                    "  n{} [label=\"Matrix {}x{}\", shape=box3d, fillcolor=\"#b2dfdb\"];\n",
                    node_id, rows, cols
                ));
                for (i, elem) in data.iter().enumerate() {
                    let elem_id = self.write_node(output, *elem);
                    output.push_str(&format!(
                        "  n{} -> n{} [label=\"[{}]\"];\n",
                        node_id, elem_id, i
                    ));
                }
            }
            Expr::SessionRef(id) => {
                output.push_str(&format!(
                    "  n{} [label=\"#{}\", shape=note, fillcolor=\"#e1bee7\"];\n",
                    node_id, id
                ));
            }
        }

        node_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_add() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Add(x, one));

        let mut viz = AstVisualizer::new(&ctx);
        let dot = viz.to_dot(expr);

        assert!(dot.contains("digraph AST"));
        assert!(dot.contains("label=\"+\""));
        assert!(dot.contains("label=\"x\""));
        assert!(dot.contains("label=\"1\""));
    }

    #[test]
    fn test_complex_expr() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);

        // x^2
        let x_squared = ctx.add(Expr::Pow(x, two));

        let mut viz = AstVisualizer::new(&ctx);
        let dot = viz.to_dot(x_squared);

        assert!(dot.contains("label=\"^\""));
        assert!(dot.contains("label=\"x\""));
        assert!(dot.contains("label=\"2\""));
    }
}
