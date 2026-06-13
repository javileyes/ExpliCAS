# Engine Improvement Scorecard

- Generated: 2026-06-13T10:05:09.875096+00:00
- Git branch: main
- Git commit: `81643c2e6aa75c0901e34bde0499dff32d165824`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=781.76ms avg_case_ms=7.82 simplify=221.88ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=686.00ms avg_case_ms=3.43 simplify=230.82ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=474.04ms avg_case_ms=4.74 simplify=136.04ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=322.38ms avg_case_ms=6.45 simplify=101.86ms avg_simplify_ms=2.04
- Engine hotspots: sum simplify=230.82ms avg_simplify_ms=1.15 wall=686.00ms, shifted_quotient simplify=221.88ms avg_simplify_ms=2.22 wall=781.76ms, product simplify=136.04ms avg_simplify_ms=1.36 wall=474.04ms, difference simplify=101.86ms avg_simplify_ms=2.04 wall=322.38ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=781.76ms avg_case_ms=7.82 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=497.06ms avg_case_ms=4.97 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=474.04ms avg_case_ms=4.74 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=322.38ms avg_case_ms=6.45 avg_simplify_ms=2.04, sum@700+100 failed=0 elapsed=188.95ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.25ms median_wall=49.57ms, product@0+100 #175 product runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.53ms, difference@0+50 #174 difference runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.66ms, sum@0+100 #173 sum runs=3 median_simplify=11.49ms median_wire=11.54ms median_wall=44.06ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.48ms median_wire=10.55ms median_wall=39.62ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
