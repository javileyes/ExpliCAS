# Engine Improvement Scorecard

- Generated: 2026-07-10T10:35:14.261996+00:00
- Git branch: main
- Git commit: `3456a05b42c5df0aef5db1e0ce41385a07f6fe3d`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=954.93ms avg_case_ms=9.55 simplify=265.98ms avg_simplify_ms=2.66, sum total=200 failed=0 elapsed=819.45ms avg_case_ms=4.10 simplify=264.32ms avg_simplify_ms=1.32, product total=100 failed=0 elapsed=607.78ms avg_case_ms=6.08 simplify=172.74ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=383.44ms avg_case_ms=7.67 simplify=116.27ms avg_simplify_ms=2.33
- Engine hotspots: shifted_quotient simplify=265.98ms avg_simplify_ms=2.66 wall=954.93ms, sum simplify=264.32ms avg_simplify_ms=1.32 wall=819.45ms, product simplify=172.74ms avg_simplify_ms=1.73 wall=607.78ms, difference simplify=116.27ms avg_simplify_ms=2.33 wall=383.44ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=954.93ms avg_case_ms=9.55 avg_simplify_ms=2.66, product@0+100 failed=0 elapsed=607.78ms avg_case_ms=6.08 avg_simplify_ms=1.73, sum@0+100 failed=0 elapsed=597.34ms avg_case_ms=5.97 avg_simplify_ms=1.85, difference@0+50 failed=0 elapsed=383.44ms avg_case_ms=7.67 avg_simplify_ms=2.33, sum@700+100 failed=0 elapsed=222.11ms avg_case_ms=2.22 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.38ms median_wire=16.44ms median_wall=62.04ms, product@0+100 #175 product runs=3 median_simplify=14.68ms median_wire=14.73ms median_wall=56.52ms, sum@0+100 #173 sum runs=3 median_simplify=16.01ms median_wire=16.05ms median_wall=62.02ms, difference@0+50 #174 difference runs=3 median_simplify=15.76ms median_wire=15.81ms median_wall=60.24ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.40ms median_wire=12.47ms median_wall=47.38ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.77s | passed=450 failed=0 total=450 avg_case=6.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
