# Engine Improvement Scorecard

- Generated: 2026-06-10T08:32:54.515567+00:00
- Git branch: main
- Git commit: `c7b88f006cd242929e16459ce04be5a9ecdc7587`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=770.19ms avg_case_ms=7.70 simplify=217.94ms avg_simplify_ms=2.18, sum total=200 failed=0 elapsed=709.67ms avg_case_ms=3.55 simplify=236.47ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=498.65ms avg_case_ms=4.99 simplify=146.31ms avg_simplify_ms=1.46, difference total=50 failed=0 elapsed=327.31ms avg_case_ms=6.55 simplify=104.43ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=236.47ms avg_simplify_ms=1.18 wall=709.67ms, shifted_quotient simplify=217.94ms avg_simplify_ms=2.18 wall=770.19ms, product simplify=146.31ms avg_simplify_ms=1.46 wall=498.65ms, difference simplify=104.43ms avg_simplify_ms=2.09 wall=327.31ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=770.19ms avg_case_ms=7.70 avg_simplify_ms=2.18, sum@0+100 failed=0 elapsed=515.71ms avg_case_ms=5.16 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=498.65ms avg_case_ms=4.99 avg_simplify_ms=1.46, difference@0+50 failed=0 elapsed=327.31ms avg_case_ms=6.55 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=193.97ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.33ms median_wire=13.41ms median_wall=51.52ms, product@0+100 #175 product runs=3 median_simplify=11.64ms median_wire=11.69ms median_wall=44.43ms, difference@0+50 #174 difference runs=3 median_simplify=11.77ms median_wire=11.82ms median_wall=44.39ms, sum@0+100 #173 sum runs=3 median_simplify=11.58ms median_wire=11.63ms median_wall=44.73ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.75ms median_wire=10.83ms median_wall=41.04ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.30s | passed=1 failed=0 |
