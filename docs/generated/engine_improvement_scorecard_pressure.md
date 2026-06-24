# Engine Improvement Scorecard

- Generated: 2026-06-24T22:13:14.016952+00:00
- Git branch: main
- Git commit: `9e5661245171284092d7741c5588a8f27a8c2872`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=795.50ms avg_case_ms=7.96 simplify=228.16ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=708.22ms avg_case_ms=3.54 simplify=240.24ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=489.12ms avg_case_ms=4.89 simplify=141.44ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=339.36ms avg_case_ms=6.79 simplify=108.02ms avg_simplify_ms=2.16
- Engine hotspots: sum simplify=240.24ms avg_simplify_ms=1.20 wall=708.22ms, shifted_quotient simplify=228.16ms avg_simplify_ms=2.28 wall=795.50ms, product simplify=141.44ms avg_simplify_ms=1.41 wall=489.12ms, difference simplify=108.02ms avg_simplify_ms=2.16 wall=339.36ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=795.50ms avg_case_ms=7.96 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=510.65ms avg_case_ms=5.11 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=489.12ms avg_case_ms=4.89 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=339.36ms avg_case_ms=6.79 avg_simplify_ms=2.16, sum@700+100 failed=0 elapsed=197.57ms avg_case_ms=1.98 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.31ms median_wire=13.38ms median_wall=50.24ms, difference@0+50 #174 difference runs=3 median_simplify=11.48ms median_wire=11.54ms median_wall=44.01ms, product@0+100 #175 product runs=3 median_simplify=11.67ms median_wire=11.73ms median_wall=44.47ms, sum@0+100 #173 sum runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.20ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.79ms median_wire=10.87ms median_wall=40.80ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.54s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
