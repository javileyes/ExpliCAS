# Engine Improvement Scorecard

- Generated: 2026-07-06T21:20:48.594914+00:00
- Git branch: main
- Git commit: `a426053479f462d0e3cca47ab6a814642614b0ab`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=929.74ms avg_case_ms=9.30 simplify=257.01ms avg_simplify_ms=2.57, sum total=200 failed=0 elapsed=825.06ms avg_case_ms=4.13 simplify=267.26ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=582.11ms avg_case_ms=5.82 simplify=165.32ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=387.06ms avg_case_ms=7.74 simplify=117.04ms avg_simplify_ms=2.34
- Engine hotspots: sum simplify=267.26ms avg_simplify_ms=1.34 wall=825.06ms, shifted_quotient simplify=257.01ms avg_simplify_ms=2.57 wall=929.74ms, product simplify=165.32ms avg_simplify_ms=1.65 wall=582.11ms, difference simplify=117.04ms avg_simplify_ms=2.34 wall=387.06ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=929.74ms avg_case_ms=9.30 avg_simplify_ms=2.57, sum@0+100 failed=0 elapsed=601.05ms avg_case_ms=6.01 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=582.11ms avg_case_ms=5.82 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=387.06ms avg_case_ms=7.74 avg_simplify_ms=2.34, sum@700+100 failed=0 elapsed=224.02ms avg_case_ms=2.24 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.23ms median_wire=16.29ms median_wall=62.74ms, product@0+100 #175 product runs=3 median_simplify=14.74ms median_wire=14.79ms median_wall=57.11ms, difference@0+50 #174 difference runs=3 median_simplify=15.07ms median_wire=15.11ms median_wall=56.94ms, sum@0+100 #173 sum runs=3 median_simplify=14.89ms median_wire=14.94ms median_wall=56.84ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.52ms median_wire=12.59ms median_wall=48.14ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.73s | passed=450 failed=0 total=450 avg_case=6.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.98s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.88s | passed=1 failed=0 |
