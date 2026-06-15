# Engine Improvement Scorecard

- Generated: 2026-06-15T18:39:12.312913+00:00
- Git branch: main
- Git commit: `df1390a562ab3662d9d6aed6664a5ce3f6931439`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=789.51ms avg_case_ms=7.90 simplify=223.60ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=698.92ms avg_case_ms=3.49 simplify=233.54ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=485.42ms avg_case_ms=4.85 simplify=139.48ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=330.15ms avg_case_ms=6.60 simplify=104.34ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=233.54ms avg_simplify_ms=1.17 wall=698.92ms, shifted_quotient simplify=223.60ms avg_simplify_ms=2.24 wall=789.51ms, product simplify=139.48ms avg_simplify_ms=1.39 wall=485.42ms, difference simplify=104.34ms avg_simplify_ms=2.09 wall=330.15ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=789.51ms avg_case_ms=7.90 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=502.32ms avg_case_ms=5.02 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=485.42ms avg_case_ms=4.85 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=330.15ms avg_case_ms=6.60 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=196.60ms avg_case_ms=1.97 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.15ms median_wall=50.07ms, product@0+100 #175 product runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.85ms, difference@0+50 #174 difference runs=3 median_simplify=11.57ms median_wire=11.62ms median_wall=44.17ms, sum@0+100 #173 sum runs=3 median_simplify=11.99ms median_wire=12.04ms median_wall=45.50ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.62ms median_wire=10.70ms median_wall=40.25ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.88s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.96s | passed=1 failed=0 |
