# Engine Improvement Scorecard

- Generated: 2026-06-12T11:48:44.633262+00:00
- Git branch: main
- Git commit: `ec5bb381633cf3ee175294bb98ee289c5c15e76c`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=782.67ms avg_case_ms=7.83 simplify=223.36ms avg_simplify_ms=2.23, sum total=200 failed=0 elapsed=691.21ms avg_case_ms=3.46 simplify=231.38ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=474.96ms avg_case_ms=4.75 simplify=136.47ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=323.92ms avg_case_ms=6.48 simplify=102.83ms avg_simplify_ms=2.06
- Engine hotspots: sum simplify=231.38ms avg_simplify_ms=1.16 wall=691.21ms, shifted_quotient simplify=223.36ms avg_simplify_ms=2.23 wall=782.67ms, product simplify=136.47ms avg_simplify_ms=1.36 wall=474.96ms, difference simplify=102.83ms avg_simplify_ms=2.06 wall=323.92ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=782.67ms avg_case_ms=7.83 avg_simplify_ms=2.23, sum@0+100 failed=0 elapsed=501.66ms avg_case_ms=5.02 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=474.96ms avg_case_ms=4.75 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=323.92ms avg_case_ms=6.48 avg_simplify_ms=2.06, sum@700+100 failed=0 elapsed=189.55ms avg_case_ms=1.90 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.87ms median_wire=12.93ms median_wall=48.79ms, sum@0+100 #173 sum runs=3 median_simplify=11.43ms median_wire=11.48ms median_wall=43.45ms, product@0+100 #175 product runs=3 median_simplify=11.43ms median_wire=11.48ms median_wall=43.85ms, difference@0+50 #174 difference runs=3 median_simplify=11.39ms median_wire=11.44ms median_wall=43.47ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.47ms median_wire=10.54ms median_wall=39.77ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
