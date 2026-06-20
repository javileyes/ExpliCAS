# Engine Improvement Scorecard

- Generated: 2026-06-20T12:45:25.633493+00:00
- Git branch: main
- Git commit: `2c34ed9eaf0bfe26f706225b42e9d540032aa528`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=799.62ms avg_case_ms=8.00 simplify=228.78ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=699.77ms avg_case_ms=3.50 simplify=236.88ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=509.58ms avg_case_ms=5.10 simplify=147.79ms avg_simplify_ms=1.48, difference total=50 failed=0 elapsed=337.37ms avg_case_ms=6.75 simplify=108.21ms avg_simplify_ms=2.16
- Engine hotspots: sum simplify=236.88ms avg_simplify_ms=1.18 wall=699.77ms, shifted_quotient simplify=228.78ms avg_simplify_ms=2.29 wall=799.62ms, product simplify=147.79ms avg_simplify_ms=1.48 wall=509.58ms, difference simplify=108.21ms avg_simplify_ms=2.16 wall=337.37ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=799.62ms avg_case_ms=8.00 avg_simplify_ms=2.29, product@0+100 failed=0 elapsed=509.58ms avg_case_ms=5.10 avg_simplify_ms=1.48, sum@0+100 failed=0 elapsed=506.04ms avg_case_ms=5.06 avg_simplify_ms=1.64, difference@0+50 failed=0 elapsed=337.37ms avg_case_ms=6.75 avg_simplify_ms=2.16, sum@700+100 failed=0 elapsed=193.73ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: difference@0+50 #174 difference runs=3 median_simplify=11.82ms median_wire=11.87ms median_wall=45.34ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.30ms median_wire=13.38ms median_wall=53.14ms, product@0+100 #175 product runs=3 median_simplify=13.44ms median_wire=13.51ms median_wall=51.65ms, sum@0+100 #173 sum runs=3 median_simplify=13.63ms median_wire=13.70ms median_wall=51.87ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.14ms median_wire=12.22ms median_wall=46.05ms
- Steady-state dominant expressions: difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.35s | passed=450 failed=0 total=450 avg_case=5.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.58s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
