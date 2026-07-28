# Engine Improvement Scorecard

- Generated: 2026-07-28T18:57:40.863437+00:00
- Git branch: main
- Git commit: `d692f640e34eef4a358e28caf92ebe6fda5cdacd`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=970.15ms avg_case_ms=9.70 simplify=273.45ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=849.87ms avg_case_ms=4.25 simplify=275.64ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=599.61ms avg_case_ms=6.00 simplify=172.06ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=401.32ms avg_case_ms=8.03 simplify=121.92ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=275.64ms avg_simplify_ms=1.38 wall=849.87ms, shifted_quotient simplify=273.45ms avg_simplify_ms=2.73 wall=970.15ms, product simplify=172.06ms avg_simplify_ms=1.72 wall=599.61ms, difference simplify=121.92ms avg_simplify_ms=2.44 wall=401.32ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=970.15ms avg_case_ms=9.70 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=621.06ms avg_case_ms=6.21 avg_simplify_ms=1.94, product@0+100 failed=0 elapsed=599.61ms avg_case_ms=6.00 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=401.32ms avg_case_ms=8.03 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=228.81ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.69ms median_wire=16.75ms median_wall=63.48ms, product@0+100 #175 product runs=3 median_simplify=14.98ms median_wire=15.03ms median_wall=57.34ms, difference@0+50 #174 difference runs=3 median_simplify=14.96ms median_wire=15.01ms median_wall=57.12ms, sum@0+100 #173 sum runs=3 median_simplify=14.91ms median_wire=14.96ms median_wall=56.49ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.69ms median_wire=12.76ms median_wall=47.99ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.82s | passed=450 failed=0 total=450 avg_case=6.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.11s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.84s | passed=1 failed=0 |
