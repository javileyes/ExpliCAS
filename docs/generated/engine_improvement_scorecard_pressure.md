# Engine Improvement Scorecard

- Generated: 2026-07-29T01:07:01.160529+00:00
- Git branch: main
- Git commit: `a8d89a40e8be1518c3339249d90c9394f7513258`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=981.04ms avg_case_ms=9.81 simplify=275.66ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=858.60ms avg_case_ms=4.29 simplify=278.64ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=604.88ms avg_case_ms=6.05 simplify=173.47ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=397.84ms avg_case_ms=7.96 simplify=120.58ms avg_simplify_ms=2.41
- Engine hotspots: sum simplify=278.64ms avg_simplify_ms=1.39 wall=858.60ms, shifted_quotient simplify=275.66ms avg_simplify_ms=2.76 wall=981.04ms, product simplify=173.47ms avg_simplify_ms=1.73 wall=604.88ms, difference simplify=120.58ms avg_simplify_ms=2.41 wall=397.84ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=981.04ms avg_case_ms=9.81 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=628.99ms avg_case_ms=6.29 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=604.88ms avg_case_ms=6.05 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=397.84ms avg_case_ms=7.96 avg_simplify_ms=2.41, sum@700+100 failed=0 elapsed=229.62ms avg_case_ms=2.30 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.62ms median_wire=16.68ms median_wall=63.54ms, difference@0+50 #174 difference runs=3 median_simplify=15.30ms median_wire=15.35ms median_wall=58.31ms, sum@0+100 #173 sum runs=3 median_simplify=15.29ms median_wire=15.33ms median_wall=58.50ms, product@0+100 #175 product runs=3 median_simplify=15.72ms median_wire=15.78ms median_wall=58.93ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.68ms median_wire=12.74ms median_wall=47.98ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.84s | passed=450 failed=0 total=450 avg_case=6.311ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.22s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.84s | passed=1 failed=0 |
