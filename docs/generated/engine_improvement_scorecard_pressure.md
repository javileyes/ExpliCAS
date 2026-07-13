# Engine Improvement Scorecard

- Generated: 2026-07-13T16:35:04.585517+00:00
- Git branch: main
- Git commit: `221d7c8f60ab51b217e34a870496349931e870af`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=976.81ms avg_case_ms=9.77 simplify=270.52ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=861.72ms avg_case_ms=4.31 simplify=276.38ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=593.65ms avg_case_ms=5.94 simplify=169.43ms avg_simplify_ms=1.69, difference total=50 failed=0 elapsed=402.46ms avg_case_ms=8.05 simplify=123.09ms avg_simplify_ms=2.46
- Engine hotspots: sum simplify=276.38ms avg_simplify_ms=1.38 wall=861.72ms, shifted_quotient simplify=270.52ms avg_simplify_ms=2.71 wall=976.81ms, product simplify=169.43ms avg_simplify_ms=1.69 wall=593.65ms, difference simplify=123.09ms avg_simplify_ms=2.46 wall=402.46ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=976.81ms avg_case_ms=9.77 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=632.17ms avg_case_ms=6.32 avg_simplify_ms=1.94, product@0+100 failed=0 elapsed=593.65ms avg_case_ms=5.94 avg_simplify_ms=1.69, difference@0+50 failed=0 elapsed=402.46ms avg_case_ms=8.05 avg_simplify_ms=2.46, sum@700+100 failed=0 elapsed=229.55ms avg_case_ms=2.30 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.63ms median_wire=16.70ms median_wall=63.37ms, sum@0+100 #173 sum runs=3 median_simplify=15.15ms median_wire=15.20ms median_wall=57.40ms, difference@0+50 #174 difference runs=3 median_simplify=14.96ms median_wire=15.01ms median_wall=57.93ms, product@0+100 #175 product runs=3 median_simplify=17.42ms median_wire=17.47ms median_wall=59.11ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.57ms median_wire=12.64ms median_wall=47.83ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.84s | passed=450 failed=0 total=450 avg_case=6.311ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
