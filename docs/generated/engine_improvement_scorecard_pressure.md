# Engine Improvement Scorecard

- Generated: 2026-07-17T18:01:25.483579+00:00
- Git branch: main
- Git commit: `ea08e6ea22477dcfd5797834bb12ca9e55666f2d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=978.45ms avg_case_ms=9.78 simplify=275.23ms avg_simplify_ms=2.75, sum total=200 failed=0 elapsed=894.30ms avg_case_ms=4.47 simplify=295.77ms avg_simplify_ms=1.48, product total=100 failed=0 elapsed=605.79ms avg_case_ms=6.06 simplify=174.07ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=400.95ms avg_case_ms=8.02 simplify=122.48ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=295.77ms avg_simplify_ms=1.48 wall=894.30ms, shifted_quotient simplify=275.23ms avg_simplify_ms=2.75 wall=978.45ms, product simplify=174.07ms avg_simplify_ms=1.74 wall=605.79ms, difference simplify=122.48ms avg_simplify_ms=2.45 wall=400.95ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=978.45ms avg_case_ms=9.78 avg_simplify_ms=2.75, sum@0+100 failed=0 elapsed=661.91ms avg_case_ms=6.62 avg_simplify_ms=2.13, product@0+100 failed=0 elapsed=605.79ms avg_case_ms=6.06 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=400.95ms avg_case_ms=8.02 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=232.38ms avg_case_ms=2.32 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.62ms median_wire=16.68ms median_wall=64.40ms, sum@0+100 #173 sum runs=3 median_simplify=15.33ms median_wire=15.38ms median_wall=58.16ms, difference@0+50 #174 difference runs=3 median_simplify=15.26ms median_wire=15.32ms median_wall=60.97ms, product@0+100 #175 product runs=3 median_simplify=15.15ms median_wire=15.20ms median_wall=59.04ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.09ms median_wire=13.16ms median_wall=49.91ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.53s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
