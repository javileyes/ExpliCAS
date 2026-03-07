mod final_result;
mod math;
mod step;
mod substeps;

pub(super) fn components_css() -> String {
    [
        &step::step_css(),
        &math::math_css(),
        &substeps::substeps_css(),
        final_result::FINAL_RESULT_CSS,
    ]
    .concat()
}
