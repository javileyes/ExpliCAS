#[macro_export]
macro_rules! define_rule {
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        $targets:expr, // Option<Vec<&str>>
        | $arg:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl crate::rule::Rule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply(&self, $arg: &std::rc::Rc<cas_ast::Expr>) -> Option<crate::rule::Rewrite> {
                $body
            }
            
            fn target_types(&self) -> Option<Vec<&str>> {
                $targets
            }
        }
    };
    // Overload for no targets (default None)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        | $arg:ident | $body:block
    ) => {
        $crate::define_rule!(
            $(#[$meta])*
            $struct_name,
            $name_str,
            None,
            | $arg | $body
        );
    };
}
