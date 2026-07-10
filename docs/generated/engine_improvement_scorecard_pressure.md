# Engine Improvement Scorecard

- Generated: 2026-07-10T20:25:11.135515+00:00
- Git branch: main
- Git commit: `571b40a90186c440d2e5a87520a5d6e24b5d4dcc`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=930.79ms avg_case_ms=9.31 simplify=257.83ms avg_simplify_ms=2.58, sum total=200 failed=0 elapsed=836.39ms avg_case_ms=4.18 simplify=270.15ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=571.95ms avg_case_ms=5.72 simplify=162.59ms avg_simplify_ms=1.63, difference total=50 failed=0 elapsed=393.02ms avg_case_ms=7.86 simplify=118.64ms avg_simplify_ms=2.37
- Engine hotspots: sum simplify=270.15ms avg_simplify_ms=1.35 wall=836.39ms, shifted_quotient simplify=257.83ms avg_simplify_ms=2.58 wall=930.79ms, product simplify=162.59ms avg_simplify_ms=1.63 wall=571.95ms, difference simplify=118.64ms avg_simplify_ms=2.37 wall=393.02ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=930.79ms avg_case_ms=9.31 avg_simplify_ms=2.58, sum@0+100 failed=0 elapsed=612.82ms avg_case_ms=6.13 avg_simplify_ms=1.91, product@0+100 failed=0 elapsed=571.95ms avg_case_ms=5.72 avg_simplify_ms=1.63, difference@0+50 failed=0 elapsed=393.02ms avg_case_ms=7.86 avg_simplify_ms=2.37, sum@700+100 failed=0 elapsed=223.57ms avg_case_ms=2.24 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.26ms median_wire=16.32ms median_wall=62.58ms, sum@0+100 #173 sum runs=3 median_simplify=14.82ms median_wire=14.87ms median_wall=56.25ms, difference@0+50 #174 difference runs=3 median_simplify=14.81ms median_wire=14.85ms median_wall=58.86ms, product@0+100 #175 product runs=3 median_simplify=14.48ms median_wire=14.52ms median_wall=55.66ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.18ms median_wire=12.25ms median_wall=47.02ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.73s | passed=450 failed=0 total=450 avg_case=6.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
