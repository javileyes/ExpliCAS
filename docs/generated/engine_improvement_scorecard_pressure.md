# Engine Improvement Scorecard

- Generated: 2026-06-16T10:51:28.019275+00:00
- Git branch: main
- Git commit: `8d131dd8acbaeb25b42e1c1360a1a7878798b167`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=793.66ms avg_case_ms=7.94 simplify=227.33ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=703.87ms avg_case_ms=3.52 simplify=238.39ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=485.05ms avg_case_ms=4.85 simplify=139.68ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=330.80ms avg_case_ms=6.62 simplify=106.00ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=238.39ms avg_simplify_ms=1.19 wall=703.87ms, shifted_quotient simplify=227.33ms avg_simplify_ms=2.27 wall=793.66ms, product simplify=139.68ms avg_simplify_ms=1.40 wall=485.05ms, difference simplify=106.00ms avg_simplify_ms=2.12 wall=330.80ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=793.66ms avg_case_ms=7.94 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=510.10ms avg_case_ms=5.10 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=485.05ms avg_case_ms=4.85 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=330.80ms avg_case_ms=6.62 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=193.77ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.22ms median_wire=13.29ms median_wall=49.93ms, product@0+100 #175 product runs=3 median_simplify=11.78ms median_wire=11.84ms median_wall=47.07ms, difference@0+50 #174 difference runs=3 median_simplify=12.68ms median_wire=12.73ms median_wall=47.84ms, sum@0+100 #173 sum runs=3 median_simplify=11.96ms median_wire=12.01ms median_wall=45.21ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.72ms median_wire=10.79ms median_wall=40.37ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.91s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.98s | passed=1 failed=0 |
