# Engine Improvement Scorecard

- Generated: 2026-07-23T02:22:31.083105+00:00
- Git branch: main
- Git commit: `222320bb2ed1afe9c64aaa42194a4191dbb5b101`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=988.68ms avg_case_ms=9.89 simplify=276.27ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=879.06ms avg_case_ms=4.40 simplify=280.18ms avg_simplify_ms=1.40, product total=100 failed=0 elapsed=605.04ms avg_case_ms=6.05 simplify=174.13ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=399.95ms avg_case_ms=8.00 simplify=122.11ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=280.18ms avg_simplify_ms=1.40 wall=879.06ms, shifted_quotient simplify=276.27ms avg_simplify_ms=2.76 wall=988.68ms, product simplify=174.13ms avg_simplify_ms=1.74 wall=605.04ms, difference simplify=122.11ms avg_simplify_ms=2.44 wall=399.95ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=988.68ms avg_case_ms=9.89 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=646.82ms avg_case_ms=6.47 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=605.04ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=399.95ms avg_case_ms=8.00 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=232.24ms avg_case_ms=2.32 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.78ms median_wire=16.85ms median_wall=63.80ms, product@0+100 #175 product runs=3 median_simplify=15.47ms median_wire=15.52ms median_wall=59.27ms, sum@0+100 #173 sum runs=3 median_simplify=15.16ms median_wire=15.20ms median_wall=60.30ms, difference@0+50 #174 difference runs=3 median_simplify=16.15ms median_wire=16.21ms median_wall=58.96ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.28ms median_wire=13.36ms median_wall=50.23ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.54s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
