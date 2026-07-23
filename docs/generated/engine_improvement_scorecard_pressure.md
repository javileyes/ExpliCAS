# Engine Improvement Scorecard

- Generated: 2026-07-23T07:58:36.858858+00:00
- Git branch: main
- Git commit: `7d5a25a4c7d50d8c8c5df92beebb3713719887a6`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.01s avg_case_ms=10.10 simplify=284.67ms avg_simplify_ms=2.85, sum total=200 failed=0 elapsed=897.55ms avg_case_ms=4.49 simplify=291.62ms avg_simplify_ms=1.46, product total=100 failed=0 elapsed=626.79ms avg_case_ms=6.27 simplify=181.06ms avg_simplify_ms=1.81, difference total=50 failed=0 elapsed=414.08ms avg_case_ms=8.28 simplify=127.49ms avg_simplify_ms=2.55
- Engine hotspots: sum simplify=291.62ms avg_simplify_ms=1.46 wall=897.55ms, shifted_quotient simplify=284.67ms avg_simplify_ms=2.85 wall=1.01s, product simplify=181.06ms avg_simplify_ms=1.81 wall=626.79ms, difference simplify=127.49ms avg_simplify_ms=2.55 wall=414.08ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.01s avg_case_ms=10.10 avg_simplify_ms=2.85, sum@0+100 failed=0 elapsed=661.73ms avg_case_ms=6.62 avg_simplify_ms=2.06, product@0+100 failed=0 elapsed=626.79ms avg_case_ms=6.27 avg_simplify_ms=1.81, difference@0+50 failed=0 elapsed=414.08ms avg_case_ms=8.28 avg_simplify_ms=2.55, sum@700+100 failed=0 elapsed=235.81ms avg_case_ms=2.36 avg_simplify_ms=0.86
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.14ms median_wire=17.22ms median_wall=65.24ms, product@0+100 #175 product runs=3 median_simplify=15.67ms median_wire=15.72ms median_wall=60.00ms, difference@0+50 #174 difference runs=3 median_simplify=15.60ms median_wire=15.66ms median_wall=59.86ms, sum@0+100 #173 sum runs=3 median_simplify=15.44ms median_wire=15.50ms median_wall=59.26ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.13ms median_wire=13.22ms median_wall=49.68ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.95s | passed=450 failed=0 total=450 avg_case=6.556ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.95s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.96s | passed=1 failed=0 |
