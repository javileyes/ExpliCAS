# Engine Improvement Scorecard

- Generated: 2026-07-30T15:10:35.454197+00:00
- Git branch: main
- Git commit: `f029834d8e7a1ca03ad8585bc440532829fe8fca`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=969.14ms avg_case_ms=9.69 simplify=273.41ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=854.93ms avg_case_ms=4.27 simplify=273.54ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=594.12ms avg_case_ms=5.94 simplify=170.44ms avg_simplify_ms=1.70, difference total=50 failed=0 elapsed=388.37ms avg_case_ms=7.77 simplify=113.84ms avg_simplify_ms=2.28
- Engine hotspots: sum simplify=273.54ms avg_simplify_ms=1.37 wall=854.93ms, shifted_quotient simplify=273.41ms avg_simplify_ms=2.73 wall=969.14ms, product simplify=170.44ms avg_simplify_ms=1.70 wall=594.12ms, difference simplify=113.84ms avg_simplify_ms=2.28 wall=388.37ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=969.14ms avg_case_ms=9.69 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=627.63ms avg_case_ms=6.28 avg_simplify_ms=1.92, product@0+100 failed=0 elapsed=594.12ms avg_case_ms=5.94 avg_simplify_ms=1.70, difference@0+50 failed=0 elapsed=388.37ms avg_case_ms=7.77 avg_simplify_ms=2.28, sum@700+100 failed=0 elapsed=227.30ms avg_case_ms=2.27 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.62ms median_wire=16.68ms median_wall=63.84ms, difference@0+50 #174 difference runs=3 median_simplify=14.93ms median_wire=14.98ms median_wall=57.35ms, sum@0+100 #173 sum runs=3 median_simplify=15.35ms median_wire=15.40ms median_wall=57.11ms, product@0+100 #175 product runs=3 median_simplify=15.05ms median_wire=15.09ms median_wall=57.20ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.90ms median_wire=12.97ms median_wall=48.81ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.81s | passed=450 failed=0 total=450 avg_case=6.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.18s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
