# Engine Improvement Scorecard

- Generated: 2026-06-21T16:03:34.306241+00:00
- Git branch: main
- Git commit: `0ac3354cde1aacf6b2afa397890d66ff8858a6ef`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=790.60ms avg_case_ms=7.91 simplify=225.72ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=707.31ms avg_case_ms=3.54 simplify=239.27ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=483.08ms avg_case_ms=4.83 simplify=139.45ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=333.10ms avg_case_ms=6.66 simplify=106.20ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=239.27ms avg_simplify_ms=1.20 wall=707.31ms, shifted_quotient simplify=225.72ms avg_simplify_ms=2.26 wall=790.60ms, product simplify=139.45ms avg_simplify_ms=1.39 wall=483.08ms, difference simplify=106.20ms avg_simplify_ms=2.12 wall=333.10ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=790.60ms avg_case_ms=7.91 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=510.28ms avg_case_ms=5.10 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=483.08ms avg_case_ms=4.83 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=333.10ms avg_case_ms=6.66 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=197.03ms avg_case_ms=1.97 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.10ms median_wire=13.17ms median_wall=49.89ms, sum@0+100 #173 sum runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.90ms, product@0+100 #175 product runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.41ms, difference@0+50 #174 difference runs=3 median_simplify=11.88ms median_wire=11.93ms median_wall=44.94ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.74ms median_wire=10.82ms median_wall=40.90ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
