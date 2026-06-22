# Engine Improvement Scorecard

- Generated: 2026-06-22T20:43:03.652594+00:00
- Git branch: main
- Git commit: `4c76272a812258f9d60a3697be402d1329256978`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.07ms avg_case_ms=7.94 simplify=227.86ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=707.68ms avg_case_ms=3.54 simplify=241.28ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=486.58ms avg_case_ms=4.87 simplify=141.26ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=332.95ms avg_case_ms=6.66 simplify=106.49ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=241.28ms avg_simplify_ms=1.21 wall=707.68ms, shifted_quotient simplify=227.86ms avg_simplify_ms=2.28 wall=794.07ms, product simplify=141.26ms avg_simplify_ms=1.41 wall=486.58ms, difference simplify=106.49ms avg_simplify_ms=2.13 wall=332.95ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.07ms avg_case_ms=7.94 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=510.38ms avg_case_ms=5.10 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=486.58ms avg_case_ms=4.87 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=332.95ms avg_case_ms=6.66 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=197.30ms avg_case_ms=1.97 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.16ms median_wire=13.23ms median_wall=49.97ms, sum@0+100 #173 sum runs=3 median_simplify=12.02ms median_wire=12.07ms median_wall=45.27ms, difference@0+50 #174 difference runs=3 median_simplify=11.82ms median_wire=11.88ms median_wall=44.95ms, product@0+100 #175 product runs=3 median_simplify=11.85ms median_wire=11.90ms median_wall=44.71ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.69ms median_wire=10.77ms median_wall=40.55ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
