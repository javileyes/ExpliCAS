mod equation;
mod substeps;
mod theme;

pub(super) fn solve_timeline_extra_css() -> String {
    [
        theme::THEME_CSS,
        equation::EQUATION_CSS,
        substeps::SUBSTEPS_CSS,
    ]
    .concat()
}
