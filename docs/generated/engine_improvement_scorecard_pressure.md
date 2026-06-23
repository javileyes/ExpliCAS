# Engine Improvement Scorecard

- Generated: 2026-06-23T21:08:04.907656+00:00
- Git branch: main
- Git commit: `98ea7cf3bd6bb3e60fc66a4f16e74a6acb336da3`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=807.42ms avg_case_ms=8.07 simplify=232.09ms avg_simplify_ms=2.32, sum total=200 failed=0 elapsed=702.97ms avg_case_ms=3.51 simplify=238.44ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=485.65ms avg_case_ms=4.86 simplify=140.58ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=330.13ms avg_case_ms=6.60 simplify=105.64ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=238.44ms avg_simplify_ms=1.19 wall=702.97ms, shifted_quotient simplify=232.09ms avg_simplify_ms=2.32 wall=807.42ms, product simplify=140.58ms avg_simplify_ms=1.41 wall=485.65ms, difference simplify=105.64ms avg_simplify_ms=2.11 wall=330.13ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=807.42ms avg_case_ms=8.07 avg_simplify_ms=2.32, sum@0+100 failed=0 elapsed=510.62ms avg_case_ms=5.11 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=485.65ms avg_case_ms=4.86 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=330.13ms avg_case_ms=6.60 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=192.34ms avg_case_ms=1.92 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.81ms median_wire=13.89ms median_wall=52.63ms, sum@0+100 #173 sum runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.45ms, product@0+100 #175 product runs=3 median_simplify=11.73ms median_wire=11.79ms median_wall=44.73ms, difference@0+50 #174 difference runs=3 median_simplify=11.77ms median_wire=11.82ms median_wall=45.84ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.18ms median_wire=11.26ms median_wall=42.55ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
