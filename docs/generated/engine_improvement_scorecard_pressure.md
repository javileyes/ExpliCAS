# Engine Improvement Scorecard

- Generated: 2026-06-25T11:17:19.975846+00:00
- Git branch: main
- Git commit: `9d5d2a8955b83c67d96b238d13091ead30798a47`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=789.99ms avg_case_ms=7.90 simplify=226.11ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=704.29ms avg_case_ms=3.52 simplify=242.99ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=479.15ms avg_case_ms=4.79 simplify=140.55ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=324.42ms avg_case_ms=6.49 simplify=104.50ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=242.99ms avg_simplify_ms=1.21 wall=704.29ms, shifted_quotient simplify=226.11ms avg_simplify_ms=2.26 wall=789.99ms, product simplify=140.55ms avg_simplify_ms=1.41 wall=479.15ms, difference simplify=104.50ms avg_simplify_ms=2.09 wall=324.42ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=789.99ms avg_case_ms=7.90 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=506.72ms avg_case_ms=5.07 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=479.15ms avg_case_ms=4.79 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=324.42ms avg_case_ms=6.49 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=197.57ms avg_case_ms=1.98 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.98ms median_wire=13.05ms median_wall=49.66ms, sum@0+100 #173 sum runs=3 median_simplify=11.60ms median_wire=11.65ms median_wall=44.38ms, difference@0+50 #174 difference runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.27ms, product@0+100 #175 product runs=3 median_simplify=11.84ms median_wire=11.89ms median_wall=44.87ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.54ms median_wire=10.62ms median_wall=39.64ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
