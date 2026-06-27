# Engine Improvement Scorecard

- Generated: 2026-06-27T21:05:38.548025+00:00
- Git branch: main
- Git commit: `68a13a240d51735da93625f286768a7c94e031d9`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=786.91ms avg_case_ms=7.87 simplify=225.67ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=709.55ms avg_case_ms=3.55 simplify=244.65ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=480.56ms avg_case_ms=4.81 simplify=140.73ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=330.05ms avg_case_ms=6.60 simplify=106.47ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=244.65ms avg_simplify_ms=1.22 wall=709.55ms, shifted_quotient simplify=225.67ms avg_simplify_ms=2.26 wall=786.91ms, product simplify=140.73ms avg_simplify_ms=1.41 wall=480.56ms, difference simplify=106.47ms avg_simplify_ms=2.13 wall=330.05ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=786.91ms avg_case_ms=7.87 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=511.46ms avg_case_ms=5.11 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=480.56ms avg_case_ms=4.81 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=330.05ms avg_case_ms=6.60 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=198.09ms avg_case_ms=1.98 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.97ms median_wire=13.04ms median_wall=50.01ms, product@0+100 #175 product runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.72ms, difference@0+50 #174 difference runs=3 median_simplify=11.62ms median_wire=11.67ms median_wall=44.13ms, sum@0+100 #173 sum runs=3 median_simplify=11.54ms median_wire=11.60ms median_wall=44.07ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.78ms median_wire=10.85ms median_wall=40.63ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
