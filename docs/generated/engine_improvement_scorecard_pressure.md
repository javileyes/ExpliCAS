# Engine Improvement Scorecard

- Generated: 2026-07-19T17:42:01.821769+00:00
- Git branch: main
- Git commit: `c8d224d48908ca06661ce933076acf29b60a03b4`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=971.21ms avg_case_ms=9.71 simplify=271.60ms avg_simplify_ms=2.72, sum total=200 failed=0 elapsed=880.95ms avg_case_ms=4.40 simplify=282.43ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=609.46ms avg_case_ms=6.09 simplify=176.85ms avg_simplify_ms=1.77, difference total=50 failed=0 elapsed=399.36ms avg_case_ms=7.99 simplify=122.28ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=282.43ms avg_simplify_ms=1.41 wall=880.95ms, shifted_quotient simplify=271.60ms avg_simplify_ms=2.72 wall=971.21ms, product simplify=176.85ms avg_simplify_ms=1.77 wall=609.46ms, difference simplify=122.28ms avg_simplify_ms=2.45 wall=399.36ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=971.21ms avg_case_ms=9.71 avg_simplify_ms=2.72, sum@0+100 failed=0 elapsed=652.01ms avg_case_ms=6.52 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=609.46ms avg_case_ms=6.09 avg_simplify_ms=1.77, difference@0+50 failed=0 elapsed=399.36ms avg_case_ms=7.99 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=228.94ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.27ms median_wire=15.32ms median_wall=60.71ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.61ms median_wire=16.68ms median_wall=63.55ms, difference@0+50 #174 difference runs=3 median_simplify=15.19ms median_wire=15.23ms median_wall=58.29ms, product@0+100 #175 product runs=3 median_simplify=15.12ms median_wire=15.17ms median_wall=57.44ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.97ms median_wire=13.03ms median_wall=48.91ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
