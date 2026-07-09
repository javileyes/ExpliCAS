# Engine Improvement Scorecard

- Generated: 2026-07-09T09:44:08.975219+00:00
- Git branch: main
- Git commit: `db7ab74994a6b75e68a92993b4b067b946560c97`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=943.52ms avg_case_ms=9.44 simplify=263.63ms avg_simplify_ms=2.64, sum total=200 failed=0 elapsed=817.24ms avg_case_ms=4.09 simplify=264.12ms avg_simplify_ms=1.32, product total=100 failed=0 elapsed=578.90ms avg_case_ms=5.79 simplify=165.19ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=382.91ms avg_case_ms=7.66 simplify=115.70ms avg_simplify_ms=2.31
- Engine hotspots: sum simplify=264.12ms avg_simplify_ms=1.32 wall=817.24ms, shifted_quotient simplify=263.63ms avg_simplify_ms=2.64 wall=943.52ms, product simplify=165.19ms avg_simplify_ms=1.65 wall=578.90ms, difference simplify=115.70ms avg_simplify_ms=2.31 wall=382.91ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=943.52ms avg_case_ms=9.44 avg_simplify_ms=2.64, sum@0+100 failed=0 elapsed=598.48ms avg_case_ms=5.98 avg_simplify_ms=1.86, product@0+100 failed=0 elapsed=578.90ms avg_case_ms=5.79 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=382.91ms avg_case_ms=7.66 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=218.76ms avg_case_ms=2.19 avg_simplify_ms=0.78
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.27ms median_wire=16.34ms median_wall=62.43ms, product@0+100 #175 product runs=3 median_simplify=14.48ms median_wire=14.53ms median_wall=55.92ms, sum@0+100 #173 sum runs=3 median_simplify=14.57ms median_wire=14.62ms median_wall=56.56ms, difference@0+50 #174 difference runs=3 median_simplify=14.73ms median_wire=14.77ms median_wall=55.64ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.52ms median_wire=12.59ms median_wall=47.84ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.72s | passed=450 failed=0 total=450 avg_case=6.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
