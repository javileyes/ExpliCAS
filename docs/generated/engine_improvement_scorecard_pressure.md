# Engine Improvement Scorecard

- Generated: 2026-07-29T09:56:05.073857+00:00
- Git branch: main
- Git commit: `ea16d35646a5a2f01f45e1db0444ce94618d139e`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=991.31ms avg_case_ms=9.91 simplify=279.61ms avg_simplify_ms=2.80, sum total=200 failed=0 elapsed=864.87ms avg_case_ms=4.32 simplify=280.74ms avg_simplify_ms=1.40, product total=100 failed=0 elapsed=608.67ms avg_case_ms=6.09 simplify=175.00ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=403.73ms avg_case_ms=8.07 simplify=122.88ms avg_simplify_ms=2.46
- Engine hotspots: sum simplify=280.74ms avg_simplify_ms=1.40 wall=864.87ms, shifted_quotient simplify=279.61ms avg_simplify_ms=2.80 wall=991.31ms, product simplify=175.00ms avg_simplify_ms=1.75 wall=608.67ms, difference simplify=122.88ms avg_simplify_ms=2.46 wall=403.73ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=991.31ms avg_case_ms=9.91 avg_simplify_ms=2.80, sum@0+100 failed=0 elapsed=631.18ms avg_case_ms=6.31 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=608.67ms avg_case_ms=6.09 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=403.73ms avg_case_ms=8.07 avg_simplify_ms=2.46, sum@700+100 failed=0 elapsed=233.69ms avg_case_ms=2.34 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.75ms median_wire=16.82ms median_wall=63.72ms, product@0+100 #175 product runs=3 median_simplify=15.21ms median_wire=15.27ms median_wall=58.31ms, difference@0+50 #174 difference runs=3 median_simplify=15.27ms median_wire=15.32ms median_wall=58.26ms, sum@0+100 #173 sum runs=3 median_simplify=15.32ms median_wire=15.37ms median_wall=57.95ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.86ms median_wire=12.93ms median_wall=48.78ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.29s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
