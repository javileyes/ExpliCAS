# Engine Improvement Scorecard

- Generated: 2026-08-02T09:23:37.030399+00:00
- Git branch: main
- Git commit: `6030043f8749a34ec58bc53fd822a0fc16c763d8`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=391

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=992.54ms avg_case_ms=9.93 simplify=278.63ms avg_simplify_ms=2.79, sum total=200 failed=0 elapsed=865.48ms avg_case_ms=4.33 simplify=278.86ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=612.57ms avg_case_ms=6.13 simplify=175.09ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=393.20ms avg_case_ms=7.86 simplify=115.61ms avg_simplify_ms=2.31
- Engine hotspots: sum simplify=278.86ms avg_simplify_ms=1.39 wall=865.48ms, shifted_quotient simplify=278.63ms avg_simplify_ms=2.79 wall=992.54ms, product simplify=175.09ms avg_simplify_ms=1.75 wall=612.57ms, difference simplify=115.61ms avg_simplify_ms=2.31 wall=393.20ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=992.54ms avg_case_ms=9.93 avg_simplify_ms=2.79, sum@0+100 failed=0 elapsed=623.50ms avg_case_ms=6.24 avg_simplify_ms=1.92, product@0+100 failed=0 elapsed=612.57ms avg_case_ms=6.13 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=393.20ms avg_case_ms=7.86 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=241.98ms avg_case_ms=2.42 avg_simplify_ms=0.86
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.84ms median_wire=16.91ms median_wall=63.62ms, product@0+100 #175 product runs=3 median_simplify=14.91ms median_wire=14.96ms median_wall=57.33ms, difference@0+50 #174 difference runs=3 median_simplify=15.14ms median_wire=15.19ms median_wall=57.43ms, sum@0+100 #173 sum runs=3 median_simplify=15.18ms median_wire=15.23ms median_wall=57.85ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.85ms median_wire=12.93ms median_wall=48.81ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.37s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.75s | passed=1 failed=0 |
