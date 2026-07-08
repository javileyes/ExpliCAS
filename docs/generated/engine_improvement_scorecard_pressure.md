# Engine Improvement Scorecard

- Generated: 2026-07-08T20:03:30.480835+00:00
- Git branch: main
- Git commit: `2a0a35264656dde5834e439cec7d5f10bf1c1b4b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=931.01ms avg_case_ms=9.31 simplify=257.10ms avg_simplify_ms=2.57, sum total=200 failed=0 elapsed=820.08ms avg_case_ms=4.10 simplify=265.22ms avg_simplify_ms=1.33, product total=100 failed=0 elapsed=583.40ms avg_case_ms=5.83 simplify=166.07ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=390.78ms avg_case_ms=7.82 simplify=117.13ms avg_simplify_ms=2.34
- Engine hotspots: sum simplify=265.22ms avg_simplify_ms=1.33 wall=820.08ms, shifted_quotient simplify=257.10ms avg_simplify_ms=2.57 wall=931.01ms, product simplify=166.07ms avg_simplify_ms=1.66 wall=583.40ms, difference simplify=117.13ms avg_simplify_ms=2.34 wall=390.78ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=931.01ms avg_case_ms=9.31 avg_simplify_ms=2.57, sum@0+100 failed=0 elapsed=600.39ms avg_case_ms=6.00 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=583.40ms avg_case_ms=5.83 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=390.78ms avg_case_ms=7.82 avg_simplify_ms=2.34, sum@700+100 failed=0 elapsed=219.70ms avg_case_ms=2.20 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.12ms median_wire=16.19ms median_wall=62.05ms, difference@0+50 #174 difference runs=3 median_simplify=14.70ms median_wire=14.75ms median_wall=56.54ms, sum@0+100 #173 sum runs=3 median_simplify=14.67ms median_wire=14.72ms median_wall=56.14ms, product@0+100 #175 product runs=3 median_simplify=14.82ms median_wire=14.86ms median_wall=56.74ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.31ms median_wire=12.38ms median_wall=47.42ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.73s | passed=450 failed=0 total=450 avg_case=6.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.98s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
