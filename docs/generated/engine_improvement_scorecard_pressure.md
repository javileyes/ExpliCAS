# Engine Improvement Scorecard

- Generated: 2026-06-22T20:10:01.977648+00:00
- Git branch: main
- Git commit: `00f459c2783adaf8e36f31f4f3bea5aa36d1ef81`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=796.57ms avg_case_ms=7.97 simplify=228.01ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=711.87ms avg_case_ms=3.56 simplify=242.86ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=489.64ms avg_case_ms=4.90 simplify=141.86ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=332.17ms avg_case_ms=6.64 simplify=106.87ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=242.86ms avg_simplify_ms=1.21 wall=711.87ms, shifted_quotient simplify=228.01ms avg_simplify_ms=2.28 wall=796.57ms, product simplify=141.86ms avg_simplify_ms=1.42 wall=489.64ms, difference simplify=106.87ms avg_simplify_ms=2.14 wall=332.17ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=796.57ms avg_case_ms=7.97 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=514.17ms avg_case_ms=5.14 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=489.64ms avg_case_ms=4.90 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=332.17ms avg_case_ms=6.64 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=197.70ms avg_case_ms=1.98 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.07ms median_wire=13.14ms median_wall=49.68ms, sum@0+100 #173 sum runs=3 median_simplify=11.88ms median_wire=11.93ms median_wall=45.19ms, difference@0+50 #174 difference runs=3 median_simplify=11.92ms median_wire=11.97ms median_wall=45.48ms, product@0+100 #175 product runs=3 median_simplify=11.98ms median_wire=12.03ms median_wall=45.41ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.56ms median_wire=10.63ms median_wall=40.28ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.49s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
