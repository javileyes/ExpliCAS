# Engine Improvement Scorecard

- Generated: 2026-06-28T14:14:31.991371+00:00
- Git branch: main
- Git commit: `670a2c4dee1be4129df84857519157a618aae8ff`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=831.01ms avg_case_ms=8.31 simplify=243.52ms avg_simplify_ms=2.44, sum total=200 failed=0 elapsed=746.44ms avg_case_ms=3.73 simplify=259.54ms avg_simplify_ms=1.30, product total=100 failed=0 elapsed=506.47ms avg_case_ms=5.06 simplify=149.93ms avg_simplify_ms=1.50, difference total=50 failed=0 elapsed=349.69ms avg_case_ms=6.99 simplify=113.23ms avg_simplify_ms=2.26
- Engine hotspots: sum simplify=259.54ms avg_simplify_ms=1.30 wall=746.44ms, shifted_quotient simplify=243.52ms avg_simplify_ms=2.44 wall=831.01ms, product simplify=149.93ms avg_simplify_ms=1.50 wall=506.47ms, difference simplify=113.23ms avg_simplify_ms=2.26 wall=349.69ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=831.01ms avg_case_ms=8.31 avg_simplify_ms=2.44, sum@0+100 failed=0 elapsed=535.92ms avg_case_ms=5.36 avg_simplify_ms=1.77, product@0+100 failed=0 elapsed=506.47ms avg_case_ms=5.06 avg_simplify_ms=1.50, difference@0+50 failed=0 elapsed=349.69ms avg_case_ms=6.99 avg_simplify_ms=2.26, sum@700+100 failed=0 elapsed=210.52ms avg_case_ms=2.11 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #160 shifted_quotient runs=3 median_simplify=9.65ms median_wire=9.73ms median_wall=36.32ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.37ms median_wire=13.45ms median_wall=51.28ms, product@0+100 #175 product runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=45.14ms, difference@0+50 #174 difference runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=45.36ms, sum@0+100 #173 sum runs=3 median_simplify=12.03ms median_wire=12.08ms median_wall=45.93ms
- Steady-state dominant expressions: shifted_quotient@0+100 #160 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.43s | passed=450 failed=0 total=450 avg_case=5.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.53s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.09s | passed=1 failed=0 |
