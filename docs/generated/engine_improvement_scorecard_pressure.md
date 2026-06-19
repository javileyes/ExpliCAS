# Engine Improvement Scorecard

- Generated: 2026-06-19T16:59:08.518911+00:00
- Git branch: main
- Git commit: `65adb65d084c8f77eaafec2fc2d9d1ca0ade063b`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=10
- By area: calculus / integration:4, calculus / runtime:3, calculus / differentiation:2, calculus / robustness:1
- Recent 1: `calculus / integration` - 2026-06-08 - Discovery observe-only: polynomial cosecant/cotangent source-return still emits depth pressure
- Recent 2: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square atanh scaled-root runtime is not caused by the global empty-domain check
- Recent 3: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square inverse-root diff runtime is not fixed by raw target preservation

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=792.18ms avg_case_ms=7.92 simplify=226.86ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=693.42ms avg_case_ms=3.47 simplify=234.96ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=487.20ms avg_case_ms=4.87 simplify=140.54ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=332.16ms avg_case_ms=6.64 simplify=106.93ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=234.96ms avg_simplify_ms=1.17 wall=693.42ms, shifted_quotient simplify=226.86ms avg_simplify_ms=2.27 wall=792.18ms, product simplify=140.54ms avg_simplify_ms=1.41 wall=487.20ms, difference simplify=106.93ms avg_simplify_ms=2.14 wall=332.16ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=792.18ms avg_case_ms=7.92 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=502.74ms avg_case_ms=5.03 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=487.20ms avg_case_ms=4.87 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=332.16ms avg_case_ms=6.64 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=190.68ms avg_case_ms=1.91 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.21ms median_wire=13.29ms median_wall=49.78ms, difference@0+50 #174 difference runs=3 median_simplify=11.77ms median_wire=11.83ms median_wall=44.61ms, sum@0+100 #173 sum runs=3 median_simplify=11.72ms median_wire=11.77ms median_wall=44.68ms, product@0+100 #175 product runs=3 median_simplify=11.73ms median_wire=11.78ms median_wall=44.48ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.74ms median_wire=10.81ms median_wall=40.77ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
