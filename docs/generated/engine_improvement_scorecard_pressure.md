# Engine Improvement Scorecard

- Generated: 2026-06-27T01:46:10.757804+00:00
- Git branch: main
- Git commit: `054832bb7cb61c1b336375f4171659c3ca39d6f1`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=803.36ms avg_case_ms=8.03 simplify=229.12ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=707.03ms avg_case_ms=3.54 simplify=243.80ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=487.67ms avg_case_ms=4.88 simplify=143.32ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=349.34ms avg_case_ms=6.99 simplify=112.93ms avg_simplify_ms=2.26
- Engine hotspots: sum simplify=243.80ms avg_simplify_ms=1.22 wall=707.03ms, shifted_quotient simplify=229.12ms avg_simplify_ms=2.29 wall=803.36ms, product simplify=143.32ms avg_simplify_ms=1.43 wall=487.67ms, difference simplify=112.93ms avg_simplify_ms=2.26 wall=349.34ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=803.36ms avg_case_ms=8.03 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=509.35ms avg_case_ms=5.09 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=487.67ms avg_case_ms=4.88 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=349.34ms avg_case_ms=6.99 avg_simplify_ms=2.26, sum@700+100 failed=0 elapsed=197.68ms avg_case_ms=1.98 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.37ms median_wire=13.44ms median_wall=50.74ms, difference@0+50 #174 difference runs=3 median_simplify=11.87ms median_wire=11.92ms median_wall=44.53ms, product@0+100 #175 product runs=3 median_simplify=11.40ms median_wire=11.45ms median_wall=44.63ms, sum@0+100 #173 sum runs=3 median_simplify=11.63ms median_wire=11.69ms median_wall=44.46ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.59ms median_wire=10.66ms median_wall=40.40ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.35s | passed=450 failed=0 total=450 avg_case=5.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
