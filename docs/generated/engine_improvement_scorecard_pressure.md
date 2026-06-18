# Engine Improvement Scorecard

- Generated: 2026-06-18T15:32:59.339118+00:00
- Git branch: main
- Git commit: `82165f396a42c7bae9dffe0f0502efec56df6e6c`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=803.96ms avg_case_ms=8.04 simplify=230.69ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=749.18ms avg_case_ms=3.75 simplify=256.21ms avg_simplify_ms=1.28, product total=100 failed=0 elapsed=504.98ms avg_case_ms=5.05 simplify=147.16ms avg_simplify_ms=1.47, difference total=50 failed=0 elapsed=336.77ms avg_case_ms=6.74 simplify=107.99ms avg_simplify_ms=2.16
- Engine hotspots: sum simplify=256.21ms avg_simplify_ms=1.28 wall=749.18ms, shifted_quotient simplify=230.69ms avg_simplify_ms=2.31 wall=803.96ms, product simplify=147.16ms avg_simplify_ms=1.47 wall=504.98ms, difference simplify=107.99ms avg_simplify_ms=2.16 wall=336.77ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=803.96ms avg_case_ms=8.04 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=549.07ms avg_case_ms=5.49 avg_simplify_ms=1.80, product@0+100 failed=0 elapsed=504.98ms avg_case_ms=5.05 avg_simplify_ms=1.47, difference@0+50 failed=0 elapsed=336.77ms avg_case_ms=6.74 avg_simplify_ms=2.16, sum@700+100 failed=0 elapsed=200.11ms avg_case_ms=2.00 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.27ms median_wall=50.80ms, difference@0+50 #174 difference runs=3 median_simplify=12.16ms median_wire=12.22ms median_wall=45.64ms, sum@0+100 #173 sum runs=3 median_simplify=11.77ms median_wire=11.83ms median_wall=44.71ms, product@0+100 #175 product runs=3 median_simplify=11.72ms median_wire=11.77ms median_wall=44.64ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.74ms median_wire=10.82ms median_wall=41.06ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.40s | passed=450 failed=0 total=450 avg_case=5.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.54s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
