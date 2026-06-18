# Engine Improvement Scorecard

- Generated: 2026-06-18T07:37:47.333727+00:00
- Git branch: main
- Git commit: `b5ea58768dd2f330fb5df82f788e4f7b8214e312`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=800.48ms avg_case_ms=8.00 simplify=227.55ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=695.02ms avg_case_ms=3.48 simplify=235.52ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=491.36ms avg_case_ms=4.91 simplify=142.88ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=330.64ms avg_case_ms=6.61 simplify=105.80ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=235.52ms avg_simplify_ms=1.18 wall=695.02ms, shifted_quotient simplify=227.55ms avg_simplify_ms=2.28 wall=800.48ms, product simplify=142.88ms avg_simplify_ms=1.43 wall=491.36ms, difference simplify=105.80ms avg_simplify_ms=2.12 wall=330.64ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=800.48ms avg_case_ms=8.00 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=500.78ms avg_case_ms=5.01 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=491.36ms avg_case_ms=4.91 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=330.64ms avg_case_ms=6.61 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=194.24ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.10ms median_wire=13.17ms median_wall=49.42ms, difference@0+50 #174 difference runs=3 median_simplify=11.63ms median_wire=11.67ms median_wall=44.10ms, product@0+100 #175 product runs=3 median_simplify=11.72ms median_wire=11.77ms median_wall=44.82ms, sum@0+100 #173 sum runs=3 median_simplify=11.88ms median_wire=11.93ms median_wall=45.03ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.61ms median_wire=10.68ms median_wall=40.30ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
