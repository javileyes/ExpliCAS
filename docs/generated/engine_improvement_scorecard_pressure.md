# Engine Improvement Scorecard

- Generated: 2026-07-19T22:28:16.720810+00:00
- Git branch: main
- Git commit: `6a61ad702cd7388f7bdd0d50855664d1f26efd81`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=975.75ms avg_case_ms=9.76 simplify=273.37ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=883.47ms avg_case_ms=4.42 simplify=291.67ms avg_simplify_ms=1.46, product total=100 failed=0 elapsed=609.38ms avg_case_ms=6.09 simplify=174.65ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=400.37ms avg_case_ms=8.01 simplify=122.26ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=291.67ms avg_simplify_ms=1.46 wall=883.47ms, shifted_quotient simplify=273.37ms avg_simplify_ms=2.73 wall=975.75ms, product simplify=174.65ms avg_simplify_ms=1.75 wall=609.38ms, difference simplify=122.26ms avg_simplify_ms=2.45 wall=400.37ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=975.75ms avg_case_ms=9.76 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=652.51ms avg_case_ms=6.53 avg_simplify_ms=2.09, product@0+100 failed=0 elapsed=609.38ms avg_case_ms=6.09 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=400.37ms avg_case_ms=8.01 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=230.96ms avg_case_ms=2.31 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.90ms median_wire=16.96ms median_wall=64.26ms, product@0+100 #175 product runs=3 median_simplify=15.07ms median_wire=15.12ms median_wall=62.31ms, difference@0+50 #174 difference runs=3 median_simplify=15.04ms median_wire=15.09ms median_wall=57.94ms, sum@0+100 #173 sum runs=3 median_simplify=15.20ms median_wire=15.25ms median_wall=58.24ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.49ms median_wire=12.56ms median_wall=48.09ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.62s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
