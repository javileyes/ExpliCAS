mod final_result;
mod math;
mod step;
mod substeps;

pub(super) fn components_css() -> String {
    [
        step::STEP_CSS,
        math::MATH_CSS,
        substeps::SUBSTEPS_CSS,
        final_result::FINAL_RESULT_CSS,
    ]
    .concat()
}
