# Engine Improvement Scorecard

- Generated: 2026-07-24T00:12:19.875350+00:00
- Git branch: main
- Git commit: `1e95afade6b0d413f95292814517ff62429083ed`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=982.23ms avg_case_ms=9.82 simplify=276.72ms avg_simplify_ms=2.77, sum total=200 failed=0 elapsed=879.49ms avg_case_ms=4.40 simplify=282.88ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=604.75ms avg_case_ms=6.05 simplify=174.43ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=401.66ms avg_case_ms=8.03 simplify=122.64ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=282.88ms avg_simplify_ms=1.41 wall=879.49ms, shifted_quotient simplify=276.72ms avg_simplify_ms=2.77 wall=982.23ms, product simplify=174.43ms avg_simplify_ms=1.74 wall=604.75ms, difference simplify=122.64ms avg_simplify_ms=2.45 wall=401.66ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=982.23ms avg_case_ms=9.82 avg_simplify_ms=2.77, sum@0+100 failed=0 elapsed=647.40ms avg_case_ms=6.47 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=604.75ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=401.66ms avg_case_ms=8.03 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=232.09ms avg_case_ms=2.32 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.61ms median_wire=17.68ms median_wall=68.05ms, product@0+100 #175 product runs=3 median_simplify=15.57ms median_wire=15.62ms median_wall=59.50ms, difference@0+50 #174 difference runs=3 median_simplify=15.31ms median_wire=15.37ms median_wall=59.42ms, sum@0+100 #173 sum runs=3 median_simplify=15.16ms median_wire=15.21ms median_wall=58.24ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.13ms median_wire=13.22ms median_wall=49.25ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.61s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
