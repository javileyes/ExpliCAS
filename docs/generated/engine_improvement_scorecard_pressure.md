# Engine Improvement Scorecard

- Generated: 2026-07-18T18:21:37.777463+00:00
- Git branch: main
- Git commit: `aa8baeb5e249275f25837efd065c7eba4ef7e254`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.02s avg_case_ms=10.16 simplify=285.76ms avg_simplify_ms=2.86, sum total=200 failed=0 elapsed=901.28ms avg_case_ms=4.51 simplify=294.37ms avg_simplify_ms=1.47, product total=100 failed=0 elapsed=624.93ms avg_case_ms=6.25 simplify=181.11ms avg_simplify_ms=1.81, difference total=50 failed=0 elapsed=411.05ms avg_case_ms=8.22 simplify=125.20ms avg_simplify_ms=2.50
- Engine hotspots: sum simplify=294.37ms avg_simplify_ms=1.47 wall=901.28ms, shifted_quotient simplify=285.76ms avg_simplify_ms=2.86 wall=1.02s, product simplify=181.11ms avg_simplify_ms=1.81 wall=624.93ms, difference simplify=125.20ms avg_simplify_ms=2.50 wall=411.05ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.02s avg_case_ms=10.16 avg_simplify_ms=2.86, sum@0+100 failed=0 elapsed=664.98ms avg_case_ms=6.65 avg_simplify_ms=2.10, product@0+100 failed=0 elapsed=624.93ms avg_case_ms=6.25 avg_simplify_ms=1.81, difference@0+50 failed=0 elapsed=411.05ms avg_case_ms=8.22 avg_simplify_ms=2.50, sum@700+100 failed=0 elapsed=236.30ms avg_case_ms=2.36 avg_simplify_ms=0.85
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.77ms median_wire=15.81ms median_wall=60.27ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.37ms median_wire=17.45ms median_wall=66.34ms, product@0+100 #175 product runs=3 median_simplify=15.66ms median_wire=15.72ms median_wall=59.64ms, difference@0+50 #174 difference runs=3 median_simplify=15.35ms median_wire=15.41ms median_wall=58.85ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.17ms median_wire=13.26ms median_wall=49.74ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.95s | passed=450 failed=0 total=450 avg_case=6.556ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.87s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.96s | passed=1 failed=0 |
