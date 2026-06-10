# Engine Improvement Scorecard

- Generated: 2026-06-10T22:27:21.422775+00:00
- Git branch: main
- Git commit: `52f0722017bfc87236f7ad37dd8216f476c284ea`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=782.49ms avg_case_ms=7.82 simplify=222.36ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=719.46ms avg_case_ms=3.60 simplify=239.96ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=477.66ms avg_case_ms=4.78 simplify=137.77ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=331.22ms avg_case_ms=6.62 simplify=105.60ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=239.96ms avg_simplify_ms=1.20 wall=719.46ms, shifted_quotient simplify=222.36ms avg_simplify_ms=2.22 wall=782.49ms, product simplify=137.77ms avg_simplify_ms=1.38 wall=477.66ms, difference simplify=105.60ms avg_simplify_ms=2.11 wall=331.22ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=782.49ms avg_case_ms=7.82 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=525.81ms avg_case_ms=5.26 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=477.66ms avg_case_ms=4.78 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=331.22ms avg_case_ms=6.62 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=193.65ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.23ms median_wire=13.30ms median_wall=50.65ms, difference@0+50 #174 difference runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.48ms, product@0+100 #175 product runs=3 median_simplify=11.59ms median_wire=11.64ms median_wall=43.74ms, sum@0+100 #173 sum runs=3 median_simplify=11.41ms median_wire=11.46ms median_wall=43.92ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.95ms median_wire=11.03ms median_wall=41.18ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.95s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.22s | passed=1 failed=0 |
