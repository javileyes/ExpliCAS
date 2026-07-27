# Engine Improvement Scorecard

- Generated: 2026-07-27T15:40:04.222206+00:00
- Git branch: main
- Git commit: `fe4d27e1f241e44195618a1ffb067a3791c47ebc`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=974.19ms avg_case_ms=9.74 simplify=273.18ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=871.18ms avg_case_ms=4.36 simplify=283.71ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=600.93ms avg_case_ms=6.01 simplify=172.84ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=404.13ms avg_case_ms=8.08 simplify=122.70ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=283.71ms avg_simplify_ms=1.42 wall=871.18ms, shifted_quotient simplify=273.18ms avg_simplify_ms=2.73 wall=974.19ms, product simplify=172.84ms avg_simplify_ms=1.73 wall=600.93ms, difference simplify=122.70ms avg_simplify_ms=2.45 wall=404.13ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=974.19ms avg_case_ms=9.74 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=636.38ms avg_case_ms=6.36 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=600.93ms avg_case_ms=6.01 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=404.13ms avg_case_ms=8.08 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=234.80ms avg_case_ms=2.35 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.11ms median_wire=17.18ms median_wall=65.00ms, sum@0+100 #173 sum runs=3 median_simplify=15.36ms median_wire=15.41ms median_wall=58.90ms, difference@0+50 #174 difference runs=3 median_simplify=15.33ms median_wire=15.38ms median_wall=58.59ms, product@0+100 #175 product runs=3 median_simplify=15.17ms median_wire=15.22ms median_wall=58.79ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.76ms median_wire=12.84ms median_wall=49.59ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.85s | passed=450 failed=0 total=450 avg_case=6.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.31s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
