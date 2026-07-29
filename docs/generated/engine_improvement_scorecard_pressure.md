# Engine Improvement Scorecard

- Generated: 2026-07-29T16:13:40.429383+00:00
- Git branch: main
- Git commit: `4a0eddadd6b3f2f9ed24285517c818e722c418b8`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=976.84ms avg_case_ms=9.77 simplify=273.64ms avg_simplify_ms=2.74, sum total=200 failed=0 elapsed=856.75ms avg_case_ms=4.28 simplify=279.32ms avg_simplify_ms=1.40, product total=100 failed=0 elapsed=605.30ms avg_case_ms=6.05 simplify=173.41ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=403.49ms avg_case_ms=8.07 simplify=122.77ms avg_simplify_ms=2.46
- Engine hotspots: sum simplify=279.32ms avg_simplify_ms=1.40 wall=856.75ms, shifted_quotient simplify=273.64ms avg_simplify_ms=2.74 wall=976.84ms, product simplify=173.41ms avg_simplify_ms=1.73 wall=605.30ms, difference simplify=122.77ms avg_simplify_ms=2.46 wall=403.49ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=976.84ms avg_case_ms=9.77 avg_simplify_ms=2.74, sum@0+100 failed=0 elapsed=628.39ms avg_case_ms=6.28 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=605.30ms avg_case_ms=6.05 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=403.49ms avg_case_ms=8.07 avg_simplify_ms=2.46, sum@700+100 failed=0 elapsed=228.35ms avg_case_ms=2.28 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.01ms median_wire=17.09ms median_wall=64.35ms, difference@0+50 #174 difference runs=3 median_simplify=15.53ms median_wire=15.59ms median_wall=58.95ms, product@0+100 #175 product runs=3 median_simplify=15.35ms median_wire=15.41ms median_wall=58.58ms, sum@0+100 #173 sum runs=3 median_simplify=15.27ms median_wire=15.33ms median_wall=58.08ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.13ms median_wire=13.21ms median_wall=50.47ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.84s | passed=450 failed=0 total=450 avg_case=6.311ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.37s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.84s | passed=1 failed=0 |
