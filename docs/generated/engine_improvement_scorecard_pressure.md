# Engine Improvement Scorecard

- Generated: 2026-06-14T13:12:19.065155+00:00
- Git branch: main
- Git commit: `d0afe87b7c23a342591faf1d6a69cf97cfb1ae13`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=348

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=779.74ms avg_case_ms=7.80 simplify=221.82ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=733.03ms avg_case_ms=3.67 simplify=245.58ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=474.88ms avg_case_ms=4.75 simplify=136.06ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=340.13ms avg_case_ms=6.80 simplify=109.27ms avg_simplify_ms=2.19
- Engine hotspots: sum simplify=245.58ms avg_simplify_ms=1.23 wall=733.03ms, shifted_quotient simplify=221.82ms avg_simplify_ms=2.22 wall=779.74ms, product simplify=136.06ms avg_simplify_ms=1.36 wall=474.88ms, difference simplify=109.27ms avg_simplify_ms=2.19 wall=340.13ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=779.74ms avg_case_ms=7.80 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=541.18ms avg_case_ms=5.41 avg_simplify_ms=1.74, product@0+100 failed=0 elapsed=474.88ms avg_case_ms=4.75 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=340.13ms avg_case_ms=6.80 avg_simplify_ms=2.19, sum@700+100 failed=0 elapsed=191.86ms avg_case_ms=1.92 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.75ms median_wire=12.82ms median_wall=48.91ms, difference@0+50 #174 difference runs=3 median_simplify=11.39ms median_wire=11.44ms median_wall=43.50ms, sum@0+100 #173 sum runs=3 median_simplify=11.52ms median_wire=11.57ms median_wall=43.98ms, product@0+100 #175 product runs=3 median_simplify=11.40ms median_wire=11.45ms median_wall=43.78ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.76ms median_wire=10.83ms median_wall=40.75ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.85s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
