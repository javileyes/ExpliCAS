# Engine Improvement Scorecard

- Generated: 2026-06-24T11:17:37.222581+00:00
- Git branch: main
- Git commit: `d422aada1e30824044c128fc3833cd2dfd141dad`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=796.26ms avg_case_ms=7.96 simplify=227.01ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=706.72ms avg_case_ms=3.53 simplify=240.05ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=482.36ms avg_case_ms=4.82 simplify=139.17ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=330.14ms avg_case_ms=6.60 simplify=105.69ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=240.05ms avg_simplify_ms=1.20 wall=706.72ms, shifted_quotient simplify=227.01ms avg_simplify_ms=2.27 wall=796.26ms, product simplify=139.17ms avg_simplify_ms=1.39 wall=482.36ms, difference simplify=105.69ms avg_simplify_ms=2.11 wall=330.14ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=796.26ms avg_case_ms=7.96 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=513.21ms avg_case_ms=5.13 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=482.36ms avg_case_ms=4.82 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=330.14ms avg_case_ms=6.60 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=193.50ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.13ms median_wire=13.20ms median_wall=50.12ms, sum@0+100 #173 sum runs=3 median_simplify=11.68ms median_wire=11.73ms median_wall=44.77ms, difference@0+50 #174 difference runs=3 median_simplify=11.66ms median_wire=11.72ms median_wall=44.34ms, product@0+100 #175 product runs=3 median_simplify=11.72ms median_wire=11.78ms median_wall=44.81ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.86ms median_wire=10.93ms median_wall=40.64ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
