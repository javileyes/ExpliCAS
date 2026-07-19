# Engine Improvement Scorecard

- Generated: 2026-07-19T18:41:59.180780+00:00
- Git branch: main
- Git commit: `2faa5a6de6d7b10a60a86798b7e5935e88f82d8c`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=984.69ms avg_case_ms=9.85 simplify=270.53ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=880.59ms avg_case_ms=4.40 simplify=282.49ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=597.18ms avg_case_ms=5.97 simplify=171.74ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=397.68ms avg_case_ms=7.95 simplify=121.44ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=282.49ms avg_simplify_ms=1.41 wall=880.59ms, shifted_quotient simplify=270.53ms avg_simplify_ms=2.71 wall=984.69ms, product simplify=171.74ms avg_simplify_ms=1.72 wall=597.18ms, difference simplify=121.44ms avg_simplify_ms=2.43 wall=397.68ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=984.69ms avg_case_ms=9.85 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=653.80ms avg_case_ms=6.54 avg_simplify_ms=2.01, product@0+100 failed=0 elapsed=597.18ms avg_case_ms=5.97 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=397.68ms avg_case_ms=7.95 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=226.79ms avg_case_ms=2.27 avg_simplify_ms=0.81
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.14ms median_wire=15.18ms median_wall=58.16ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.38ms median_wire=16.46ms median_wall=64.55ms, product@0+100 #175 product runs=3 median_simplify=15.09ms median_wire=15.14ms median_wall=57.96ms, difference@0+50 #174 difference runs=3 median_simplify=15.65ms median_wire=15.71ms median_wall=58.73ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.44ms median_wire=12.52ms median_wall=48.58ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.56s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
