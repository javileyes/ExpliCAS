# Engine Improvement Scorecard

- Generated: 2026-06-14T21:20:01.825446+00:00
- Git branch: main
- Git commit: `14e69161bdaa022b3968656fe88ef344d9d25529`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=351

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=816.22ms avg_case_ms=8.16 simplify=233.81ms avg_simplify_ms=2.34, sum total=200 failed=0 elapsed=722.42ms avg_case_ms=3.61 simplify=243.87ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=491.72ms avg_case_ms=4.92 simplify=142.10ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=336.91ms avg_case_ms=6.74 simplify=107.06ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=243.87ms avg_simplify_ms=1.22 wall=722.42ms, shifted_quotient simplify=233.81ms avg_simplify_ms=2.34 wall=816.22ms, product simplify=142.10ms avg_simplify_ms=1.42 wall=491.72ms, difference simplify=107.06ms avg_simplify_ms=2.14 wall=336.91ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=816.22ms avg_case_ms=8.16 avg_simplify_ms=2.34, sum@0+100 failed=0 elapsed=523.06ms avg_case_ms=5.23 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=491.72ms avg_case_ms=4.92 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=336.91ms avg_case_ms=6.74 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=199.37ms avg_case_ms=1.99 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.44ms median_wire=13.51ms median_wall=50.68ms, sum@0+100 #173 sum runs=3 median_simplify=12.01ms median_wire=12.07ms median_wall=45.54ms, difference@0+50 #174 difference runs=3 median_simplify=11.88ms median_wire=11.94ms median_wall=45.62ms, product@0+100 #175 product runs=3 median_simplify=12.03ms median_wire=12.09ms median_wall=46.21ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.25ms median_wire=11.33ms median_wall=41.92ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.37s | passed=450 failed=0 total=450 avg_case=5.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.90s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.97s | passed=1 failed=0 |
