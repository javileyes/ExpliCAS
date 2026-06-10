# Engine Improvement Scorecard

- Generated: 2026-06-10T13:54:25.587559+00:00
- Git branch: main
- Git commit: `b9049daa5c1d1fcfe9ad6dfae09299b0cb606bdc`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=779.26ms avg_case_ms=7.79 simplify=221.08ms avg_simplify_ms=2.21, sum total=200 failed=0 elapsed=706.61ms avg_case_ms=3.53 simplify=235.09ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=479.80ms avg_case_ms=4.80 simplify=138.23ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=330.16ms avg_case_ms=6.60 simplify=104.79ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=235.09ms avg_simplify_ms=1.18 wall=706.61ms, shifted_quotient simplify=221.08ms avg_simplify_ms=2.21 wall=779.26ms, product simplify=138.23ms avg_simplify_ms=1.38 wall=479.80ms, difference simplify=104.79ms avg_simplify_ms=2.10 wall=330.16ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=779.26ms avg_case_ms=7.79 avg_simplify_ms=2.21, sum@0+100 failed=0 elapsed=514.75ms avg_case_ms=5.15 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=479.80ms avg_case_ms=4.80 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=330.16ms avg_case_ms=6.60 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=191.86ms avg_case_ms=1.92 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.62ms median_wire=13.70ms median_wall=51.11ms, product@0+100 #175 product runs=3 median_simplify=11.42ms median_wire=11.47ms median_wall=43.69ms, sum@0+100 #173 sum runs=3 median_simplify=11.57ms median_wire=11.62ms median_wall=43.98ms, difference@0+50 #174 difference runs=3 median_simplify=11.77ms median_wire=11.82ms median_wall=44.85ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.99ms median_wire=11.06ms median_wall=41.52ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.41s | passed=1 failed=0 |
