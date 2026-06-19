# Engine Improvement Scorecard

- Generated: 2026-06-19T21:18:01.460251+00:00
- Git branch: main
- Git commit: `1545ce3985baa78a22eddc8beb0f8d585e19ff16`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=796.42ms avg_case_ms=7.96 simplify=228.31ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=702.25ms avg_case_ms=3.51 simplify=238.79ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=481.68ms avg_case_ms=4.82 simplify=138.90ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=333.60ms avg_case_ms=6.67 simplify=106.60ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=238.79ms avg_simplify_ms=1.19 wall=702.25ms, shifted_quotient simplify=228.31ms avg_simplify_ms=2.28 wall=796.42ms, product simplify=138.90ms avg_simplify_ms=1.39 wall=481.68ms, difference simplify=106.60ms avg_simplify_ms=2.13 wall=333.60ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=796.42ms avg_case_ms=7.96 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=506.89ms avg_case_ms=5.07 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=481.68ms avg_case_ms=4.82 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=333.60ms avg_case_ms=6.67 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=195.37ms avg_case_ms=1.95 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.06ms median_wire=13.13ms median_wall=49.38ms, sum@0+100 #173 sum runs=3 median_simplify=11.66ms median_wire=11.71ms median_wall=44.36ms, difference@0+50 #174 difference runs=3 median_simplify=11.64ms median_wire=11.68ms median_wall=44.17ms, product@0+100 #175 product runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.54ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.85ms median_wire=10.92ms median_wall=41.20ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
