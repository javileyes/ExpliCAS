# Engine Improvement Scorecard

- Generated: 2026-06-20T22:43:06.669977+00:00
- Git branch: main
- Git commit: `49340ff46760fcfd6eeaa5f204735b66f0d6cdad`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=801.09ms avg_case_ms=8.01 simplify=230.33ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=702.10ms avg_case_ms=3.51 simplify=237.90ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=483.59ms avg_case_ms=4.84 simplify=139.72ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=329.31ms avg_case_ms=6.59 simplify=105.11ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=237.90ms avg_simplify_ms=1.19 wall=702.10ms, shifted_quotient simplify=230.33ms avg_simplify_ms=2.30 wall=801.09ms, product simplify=139.72ms avg_simplify_ms=1.40 wall=483.59ms, difference simplify=105.11ms avg_simplify_ms=2.10 wall=329.31ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=801.09ms avg_case_ms=8.01 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=505.41ms avg_case_ms=5.05 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=483.59ms avg_case_ms=4.84 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=329.31ms avg_case_ms=6.59 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=196.69ms avg_case_ms=1.97 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.27ms median_wire=13.34ms median_wall=50.38ms, product@0+100 #175 product runs=3 median_simplify=11.77ms median_wire=11.83ms median_wall=44.61ms, difference@0+50 #174 difference runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.44ms, sum@0+100 #173 sum runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=44.40ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.04ms median_wire=11.11ms median_wall=41.21ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
