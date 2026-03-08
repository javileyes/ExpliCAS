const WEIERSTRASS_USAGE_MESSAGE: &str = "Usage: weierstrass <expression>\n\
                 Description: Apply Weierstrass substitution (t = tan(x/2))\n\
                 Transforms:\n\
                   sin(x) -> 2t/(1+t^2)\n\
                   cos(x) -> (1-t^2)/(1+t^2)\n\
                   tan(x) -> 2t/(1-t^2)\n\
                 Example: weierstrass sin(x) + cos(x)";

/// Usage string for `weierstrass`.
pub fn weierstrass_usage_message() -> &'static str {
    WEIERSTRASS_USAGE_MESSAGE
}
