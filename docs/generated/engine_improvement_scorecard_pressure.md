# Engine Improvement Scorecard

- Generated: 2026-06-26T14:04:08.897243+00:00
- Git branch: main
- Git commit: `3794bdab50f66367a896c6725b3354b1ca9ddc11`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=789.16ms avg_case_ms=7.89 simplify=225.48ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=720.82ms avg_case_ms=3.60 simplify=249.24ms avg_simplify_ms=1.25, product total=100 failed=0 elapsed=490.12ms avg_case_ms=4.90 simplify=143.34ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=338.46ms avg_case_ms=6.77 simplify=108.66ms avg_simplify_ms=2.17
- Engine hotspots: sum simplify=249.24ms avg_simplify_ms=1.25 wall=720.82ms, shifted_quotient simplify=225.48ms avg_simplify_ms=2.25 wall=789.16ms, product simplify=143.34ms avg_simplify_ms=1.43 wall=490.12ms, difference simplify=108.66ms avg_simplify_ms=2.17 wall=338.46ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=789.16ms avg_case_ms=7.89 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=510.57ms avg_case_ms=5.11 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=490.12ms avg_case_ms=4.90 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=338.46ms avg_case_ms=6.77 avg_simplify_ms=2.17, sum@700+100 failed=0 elapsed=210.26ms avg_case_ms=2.10 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.10ms median_wire=13.17ms median_wall=50.02ms, sum@0+100 #173 sum runs=3 median_simplify=11.88ms median_wire=11.93ms median_wall=44.63ms, difference@0+50 #174 difference runs=3 median_simplify=11.89ms median_wire=11.94ms median_wall=45.33ms, product@0+100 #175 product runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=45.25ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.80ms median_wire=10.88ms median_wall=40.92ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
