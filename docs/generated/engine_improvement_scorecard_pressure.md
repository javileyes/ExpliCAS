# Engine Improvement Scorecard

- Generated: 2026-07-08T11:18:20.881909+00:00
- Git branch: main
- Git commit: `5f255c3e2c0eb68ec6e30c78f37b27c1d6c731b7`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=948.18ms avg_case_ms=9.48 simplify=262.69ms avg_simplify_ms=2.63, sum total=200 failed=0 elapsed=812.70ms avg_case_ms=4.06 simplify=262.38ms avg_simplify_ms=1.31, product total=100 failed=0 elapsed=570.22ms avg_case_ms=5.70 simplify=161.99ms avg_simplify_ms=1.62, difference total=50 failed=0 elapsed=383.75ms avg_case_ms=7.67 simplify=115.74ms avg_simplify_ms=2.31
- Engine hotspots: shifted_quotient simplify=262.69ms avg_simplify_ms=2.63 wall=948.18ms, sum simplify=262.38ms avg_simplify_ms=1.31 wall=812.70ms, product simplify=161.99ms avg_simplify_ms=1.62 wall=570.22ms, difference simplify=115.74ms avg_simplify_ms=2.31 wall=383.75ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=948.18ms avg_case_ms=9.48 avg_simplify_ms=2.63, sum@0+100 failed=0 elapsed=593.77ms avg_case_ms=5.94 avg_simplify_ms=1.84, product@0+100 failed=0 elapsed=570.22ms avg_case_ms=5.70 avg_simplify_ms=1.62, difference@0+50 failed=0 elapsed=383.75ms avg_case_ms=7.67 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=218.93ms avg_case_ms=2.19 avg_simplify_ms=0.78
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.30ms median_wire=17.37ms median_wall=65.73ms, sum@0+100 #173 sum runs=3 median_simplify=14.93ms median_wire=14.97ms median_wall=57.43ms, product@0+100 #175 product runs=3 median_simplify=14.69ms median_wire=14.73ms median_wall=57.14ms, difference@0+50 #174 difference runs=3 median_simplify=14.93ms median_wire=14.98ms median_wall=56.86ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.59ms median_wire=12.66ms median_wall=48.93ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.72s | passed=450 failed=0 total=450 avg_case=6.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.96s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
