# Engine Improvement Scorecard

- Generated: 2026-06-14T09:18:37.715779+00:00
- Git branch: main
- Git commit: `fab1c1702e3e767260148234d9b4594c51aa3aba`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=788.44ms avg_case_ms=7.88 simplify=224.58ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=736.95ms avg_case_ms=3.68 simplify=248.07ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=480.98ms avg_case_ms=4.81 simplify=137.67ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=359.99ms avg_case_ms=7.20 simplify=115.47ms avg_simplify_ms=2.31
- Engine hotspots: sum simplify=248.07ms avg_simplify_ms=1.24 wall=736.95ms, shifted_quotient simplify=224.58ms avg_simplify_ms=2.25 wall=788.44ms, product simplify=137.67ms avg_simplify_ms=1.38 wall=480.98ms, difference simplify=115.47ms avg_simplify_ms=2.31 wall=359.99ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=788.44ms avg_case_ms=7.88 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=510.02ms avg_case_ms=5.10 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=480.98ms avg_case_ms=4.81 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=359.99ms avg_case_ms=7.20 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=226.94ms avg_case_ms=2.27 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.70ms median_wire=12.76ms median_wall=48.48ms, difference@0+50 #174 difference runs=3 median_simplify=11.51ms median_wire=11.57ms median_wall=43.92ms, sum@0+100 #173 sum runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=43.76ms, product@0+100 #175 product runs=3 median_simplify=11.54ms median_wire=11.59ms median_wall=44.09ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.43ms median_wire=10.50ms median_wall=39.48ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.37s | passed=450 failed=0 total=450 avg_case=5.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
