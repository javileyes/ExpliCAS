# Engine Improvement Scorecard

- Generated: 2026-06-16T07:12:48.963918+00:00
- Git branch: main
- Git commit: `dec7e0cc30f2993700ce7d20a27f80d60755b314`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=792.19ms avg_case_ms=7.92 simplify=225.93ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=699.19ms avg_case_ms=3.50 simplify=234.36ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=479.16ms avg_case_ms=4.79 simplify=137.37ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=332.57ms avg_case_ms=6.65 simplify=105.19ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=234.36ms avg_simplify_ms=1.17 wall=699.19ms, shifted_quotient simplify=225.93ms avg_simplify_ms=2.26 wall=792.19ms, product simplify=137.37ms avg_simplify_ms=1.37 wall=479.16ms, difference simplify=105.19ms avg_simplify_ms=2.10 wall=332.57ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=792.19ms avg_case_ms=7.92 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=508.78ms avg_case_ms=5.09 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=479.16ms avg_case_ms=4.79 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=332.57ms avg_case_ms=6.65 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=190.41ms avg_case_ms=1.90 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.20ms median_wire=13.27ms median_wall=50.11ms, product@0+100 #175 product runs=3 median_simplify=11.60ms median_wire=11.65ms median_wall=44.20ms, difference@0+50 #174 difference runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.40ms, sum@0+100 #173 sum runs=3 median_simplify=11.96ms median_wire=12.02ms median_wall=45.45ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.72ms median_wire=10.79ms median_wall=40.44ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.85s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
