# Engine Improvement Scorecard

- Generated: 2026-07-29T08:46:56.440499+00:00
- Git branch: main
- Git commit: `8a2790bad4e4035d820881423402af10b5283829`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=965.62ms avg_case_ms=9.66 simplify=271.05ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=874.12ms avg_case_ms=4.37 simplify=284.32ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=605.08ms avg_case_ms=6.05 simplify=173.96ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=399.32ms avg_case_ms=7.99 simplify=121.51ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=284.32ms avg_simplify_ms=1.42 wall=874.12ms, shifted_quotient simplify=271.05ms avg_simplify_ms=2.71 wall=965.62ms, product simplify=173.96ms avg_simplify_ms=1.74 wall=605.08ms, difference simplify=121.51ms avg_simplify_ms=2.43 wall=399.32ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=965.62ms avg_case_ms=9.66 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=625.27ms avg_case_ms=6.25 avg_simplify_ms=1.96, product@0+100 failed=0 elapsed=605.08ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=399.32ms avg_case_ms=7.99 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=248.85ms avg_case_ms=2.49 avg_simplify_ms=0.89
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.69ms median_wire=16.77ms median_wall=63.91ms, difference@0+50 #174 difference runs=3 median_simplify=15.12ms median_wire=15.17ms median_wall=57.57ms, sum@0+100 #173 sum runs=3 median_simplify=15.13ms median_wire=15.18ms median_wall=58.18ms, product@0+100 #175 product runs=3 median_simplify=15.15ms median_wire=15.20ms median_wall=57.90ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.80ms median_wire=12.88ms median_wall=48.75ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.84s | passed=450 failed=0 total=450 avg_case=6.311ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.26s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.84s | passed=1 failed=0 |
