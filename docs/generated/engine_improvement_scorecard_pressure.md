# Engine Improvement Scorecard

- Generated: 2026-07-04T00:22:18.149622+00:00
- Git branch: main
- Git commit: `9c3743ba197b98c35500cad723aa8a4c3ea2b73b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=928.48ms avg_case_ms=9.28 simplify=258.24ms avg_simplify_ms=2.58, sum total=200 failed=0 elapsed=819.51ms avg_case_ms=4.10 simplify=264.32ms avg_simplify_ms=1.32, product total=100 failed=0 elapsed=569.33ms avg_case_ms=5.69 simplify=161.80ms avg_simplify_ms=1.62, difference total=50 failed=0 elapsed=380.15ms avg_case_ms=7.60 simplify=115.48ms avg_simplify_ms=2.31
- Engine hotspots: sum simplify=264.32ms avg_simplify_ms=1.32 wall=819.51ms, shifted_quotient simplify=258.24ms avg_simplify_ms=2.58 wall=928.48ms, product simplify=161.80ms avg_simplify_ms=1.62 wall=569.33ms, difference simplify=115.48ms avg_simplify_ms=2.31 wall=380.15ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=928.48ms avg_case_ms=9.28 avg_simplify_ms=2.58, sum@0+100 failed=0 elapsed=598.56ms avg_case_ms=5.99 avg_simplify_ms=1.86, product@0+100 failed=0 elapsed=569.33ms avg_case_ms=5.69 avg_simplify_ms=1.62, difference@0+50 failed=0 elapsed=380.15ms avg_case_ms=7.60 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=220.95ms avg_case_ms=2.21 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.21ms median_wire=16.27ms median_wall=62.40ms, difference@0+50 #174 difference runs=3 median_simplify=14.57ms median_wire=14.61ms median_wall=55.74ms, sum@0+100 #173 sum runs=3 median_simplify=14.58ms median_wire=14.62ms median_wall=55.55ms, product@0+100 #175 product runs=3 median_simplify=14.33ms median_wire=14.38ms median_wall=55.14ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.38ms median_wire=12.45ms median_wall=47.46ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.70s | passed=450 failed=0 total=450 avg_case=6.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.94s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
