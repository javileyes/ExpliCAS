# Engine Improvement Scorecard

- Generated: 2026-07-10T08:56:50.045054+00:00
- Git branch: main
- Git commit: `4d3e12901b05b028d11a42db72c195233879cd5d`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=933.33ms avg_case_ms=9.33 simplify=257.01ms avg_simplify_ms=2.57, sum total=200 failed=0 elapsed=834.02ms avg_case_ms=4.17 simplify=268.49ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=584.86ms avg_case_ms=5.85 simplify=165.28ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=386.77ms avg_case_ms=7.74 simplify=116.32ms avg_simplify_ms=2.33
- Engine hotspots: sum simplify=268.49ms avg_simplify_ms=1.34 wall=834.02ms, shifted_quotient simplify=257.01ms avg_simplify_ms=2.57 wall=933.33ms, product simplify=165.28ms avg_simplify_ms=1.65 wall=584.86ms, difference simplify=116.32ms avg_simplify_ms=2.33 wall=386.77ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=933.33ms avg_case_ms=9.33 avg_simplify_ms=2.57, sum@0+100 failed=0 elapsed=607.17ms avg_case_ms=6.07 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=584.86ms avg_case_ms=5.85 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=386.77ms avg_case_ms=7.74 avg_simplify_ms=2.33, sum@700+100 failed=0 elapsed=226.85ms avg_case_ms=2.27 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.23ms median_wire=16.31ms median_wall=62.23ms, product@0+100 #175 product runs=3 median_simplify=14.59ms median_wire=14.63ms median_wall=56.42ms, sum@0+100 #173 sum runs=3 median_simplify=14.73ms median_wire=14.78ms median_wall=55.88ms, difference@0+50 #174 difference runs=3 median_simplify=14.69ms median_wire=14.73ms median_wall=56.49ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.67ms median_wire=12.74ms median_wall=48.10ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.74s | passed=450 failed=0 total=450 avg_case=6.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
