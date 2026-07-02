# Engine Improvement Scorecard

- Generated: 2026-07-02T02:42:25.966733+00:00
- Git branch: main
- Git commit: `1250a156eb42a62ad1b2d7cfcc50202fa549e9d8`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=355

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=810.17ms avg_case_ms=8.10 simplify=234.51ms avg_simplify_ms=2.35, sum total=200 failed=0 elapsed=722.29ms avg_case_ms=3.61 simplify=249.91ms avg_simplify_ms=1.25, product total=100 failed=0 elapsed=496.32ms avg_case_ms=4.96 simplify=147.61ms avg_simplify_ms=1.48, difference total=50 failed=0 elapsed=330.33ms avg_case_ms=6.61 simplify=107.06ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=249.91ms avg_simplify_ms=1.25 wall=722.29ms, shifted_quotient simplify=234.51ms avg_simplify_ms=2.35 wall=810.17ms, product simplify=147.61ms avg_simplify_ms=1.48 wall=496.32ms, difference simplify=107.06ms avg_simplify_ms=2.14 wall=330.33ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=810.17ms avg_case_ms=8.10 avg_simplify_ms=2.35, sum@0+100 failed=0 elapsed=527.12ms avg_case_ms=5.27 avg_simplify_ms=1.75, product@0+100 failed=0 elapsed=496.32ms avg_case_ms=4.96 avg_simplify_ms=1.48, difference@0+50 failed=0 elapsed=330.33ms avg_case_ms=6.61 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=195.17ms avg_case_ms=1.95 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.12ms median_wire=13.19ms median_wall=49.56ms, difference@0+50 #174 difference runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=44.36ms, product@0+100 #175 product runs=3 median_simplify=11.48ms median_wire=11.53ms median_wall=43.45ms, sum@0+100 #173 sum runs=3 median_simplify=11.68ms median_wire=11.72ms median_wall=44.34ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.01ms median_wire=11.08ms median_wall=41.36ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
