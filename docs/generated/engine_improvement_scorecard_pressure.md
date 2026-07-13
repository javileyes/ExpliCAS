# Engine Improvement Scorecard

- Generated: 2026-07-13T09:19:17.747889+00:00
- Git branch: main
- Git commit: `af26670c7f3cecca7be5c29eb51b4abd095f90ff`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=991.11ms avg_case_ms=9.91 simplify=278.81ms avg_simplify_ms=2.79, sum total=200 failed=0 elapsed=868.62ms avg_case_ms=4.34 simplify=283.39ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=607.86ms avg_case_ms=6.08 simplify=173.35ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=404.39ms avg_case_ms=8.09 simplify=123.28ms avg_simplify_ms=2.47
- Engine hotspots: sum simplify=283.39ms avg_simplify_ms=1.42 wall=868.62ms, shifted_quotient simplify=278.81ms avg_simplify_ms=2.79 wall=991.11ms, product simplify=173.35ms avg_simplify_ms=1.73 wall=607.86ms, difference simplify=123.28ms avg_simplify_ms=2.47 wall=404.39ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=991.11ms avg_case_ms=9.91 avg_simplify_ms=2.79, sum@0+100 failed=0 elapsed=637.67ms avg_case_ms=6.38 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=607.86ms avg_case_ms=6.08 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=404.39ms avg_case_ms=8.09 avg_simplify_ms=2.47, sum@700+100 failed=0 elapsed=230.95ms avg_case_ms=2.31 avg_simplify_ms=0.83
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.18ms median_wire=15.23ms median_wall=58.02ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.78ms median_wire=16.85ms median_wall=64.82ms, difference@0+50 #174 difference runs=3 median_simplify=15.42ms median_wire=15.47ms median_wall=58.90ms, product@0+100 #175 product runs=3 median_simplify=15.34ms median_wire=15.39ms median_wall=58.91ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.82ms median_wire=12.90ms median_wall=48.77ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.07s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
