# Engine Improvement Scorecard

- Generated: 2026-07-24T10:36:35.278573+00:00
- Git branch: main
- Git commit: `2d68b2c2c2c020dafa54a347cebdd7d135fdf017`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.01s avg_case_ms=10.12 simplify=283.84ms avg_simplify_ms=2.84, sum total=200 failed=0 elapsed=888.69ms avg_case_ms=4.44 simplify=285.74ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=608.79ms avg_case_ms=6.09 simplify=175.59ms avg_simplify_ms=1.76, difference total=50 failed=0 elapsed=398.66ms avg_case_ms=7.97 simplify=122.07ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=285.74ms avg_simplify_ms=1.43 wall=888.69ms, shifted_quotient simplify=283.84ms avg_simplify_ms=2.84 wall=1.01s, product simplify=175.59ms avg_simplify_ms=1.76 wall=608.79ms, difference simplify=122.07ms avg_simplify_ms=2.44 wall=398.66ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.01s avg_case_ms=10.12 avg_simplify_ms=2.84, sum@0+100 failed=0 elapsed=656.08ms avg_case_ms=6.56 avg_simplify_ms=2.01, product@0+100 failed=0 elapsed=608.79ms avg_case_ms=6.09 avg_simplify_ms=1.76, difference@0+50 failed=0 elapsed=398.66ms avg_case_ms=7.97 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=232.61ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.64ms median_wire=16.71ms median_wall=64.15ms, product@0+100 #175 product runs=3 median_simplify=15.48ms median_wire=15.53ms median_wall=58.56ms, sum@0+100 #173 sum runs=3 median_simplify=15.21ms median_wire=15.26ms median_wall=58.50ms, difference@0+50 #174 difference runs=3 median_simplify=15.47ms median_wire=15.52ms median_wall=59.46ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.25ms median_wire=13.33ms median_wall=50.32ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.91s | passed=450 failed=0 total=450 avg_case=6.467ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.62s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
