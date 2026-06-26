# Engine Improvement Scorecard

- Generated: 2026-06-26T23:47:17.545072+00:00
- Git branch: main
- Git commit: `11175155182bcf0d12fd35fb0e6e28d71f753cdc`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=792.97ms avg_case_ms=7.93 simplify=227.37ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=746.09ms avg_case_ms=3.73 simplify=260.35ms avg_simplify_ms=1.30, product total=100 failed=0 elapsed=500.56ms avg_case_ms=5.01 simplify=148.63ms avg_simplify_ms=1.49, difference total=50 failed=0 elapsed=336.51ms avg_case_ms=6.73 simplify=109.06ms avg_simplify_ms=2.18
- Engine hotspots: sum simplify=260.35ms avg_simplify_ms=1.30 wall=746.09ms, shifted_quotient simplify=227.37ms avg_simplify_ms=2.27 wall=792.97ms, product simplify=148.63ms avg_simplify_ms=1.49 wall=500.56ms, difference simplify=109.06ms avg_simplify_ms=2.18 wall=336.51ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=792.97ms avg_case_ms=7.93 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=537.43ms avg_case_ms=5.37 avg_simplify_ms=1.79, product@0+100 failed=0 elapsed=500.56ms avg_case_ms=5.01 avg_simplify_ms=1.49, difference@0+50 failed=0 elapsed=336.51ms avg_case_ms=6.73 avg_simplify_ms=2.18, sum@700+100 failed=0 elapsed=208.65ms avg_case_ms=2.09 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.10ms median_wire=13.17ms median_wall=50.56ms, difference@0+50 #174 difference runs=3 median_simplify=11.82ms median_wire=11.88ms median_wall=45.10ms, product@0+100 #175 product runs=3 median_simplify=11.74ms median_wire=11.80ms median_wall=44.81ms, sum@0+100 #173 sum runs=3 median_simplify=12.07ms median_wire=12.12ms median_wall=45.62ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.85ms median_wire=10.92ms median_wall=41.24ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.38s | passed=450 failed=0 total=450 avg_case=5.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.08s | passed=1 failed=0 |
