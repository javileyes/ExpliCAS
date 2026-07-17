# Engine Improvement Scorecard

- Generated: 2026-07-17T09:43:53.249231+00:00
- Git branch: main
- Git commit: `2bea86449c2131098d37545aa5fa1eb365edb44f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=976.19ms avg_case_ms=9.76 simplify=273.49ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=874.45ms avg_case_ms=4.37 simplify=285.76ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=605.33ms avg_case_ms=6.05 simplify=174.42ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=399.01ms avg_case_ms=7.98 simplify=122.34ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=285.76ms avg_simplify_ms=1.43 wall=874.45ms, shifted_quotient simplify=273.49ms avg_simplify_ms=2.73 wall=976.19ms, product simplify=174.42ms avg_simplify_ms=1.74 wall=605.33ms, difference simplify=122.34ms avg_simplify_ms=2.45 wall=399.01ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=976.19ms avg_case_ms=9.76 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=645.52ms avg_case_ms=6.46 avg_simplify_ms=2.04, product@0+100 failed=0 elapsed=605.33ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=399.01ms avg_case_ms=7.98 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=228.92ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.33ms median_wire=17.40ms median_wall=65.85ms, sum@0+100 #157 sum runs=3 median_simplify=9.83ms median_wire=9.89ms median_wall=37.15ms, product@0+100 #175 product runs=3 median_simplify=15.29ms median_wire=15.34ms median_wall=61.80ms, sum@0+100 #173 sum runs=3 median_simplify=15.73ms median_wire=15.78ms median_wall=58.94ms, difference@0+50 #174 difference runs=3 median_simplify=15.36ms median_wire=15.41ms median_wall=59.20ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #157 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^3) + ln(y^2) - ln(x^3 * y^2)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.57s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
