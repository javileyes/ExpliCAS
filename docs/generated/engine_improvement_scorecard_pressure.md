# Engine Improvement Scorecard

- Generated: 2026-06-28T13:54:01.981650+00:00
- Git branch: main
- Git commit: `459cbf06c21b84023ea18873da6b7052719b5b2f`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=808.40ms avg_case_ms=8.08 simplify=232.15ms avg_simplify_ms=2.32, sum total=200 failed=0 elapsed=709.59ms avg_case_ms=3.55 simplify=245.39ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=485.60ms avg_case_ms=4.86 simplify=142.39ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=332.67ms avg_case_ms=6.65 simplify=107.53ms avg_simplify_ms=2.15
- Engine hotspots: sum simplify=245.39ms avg_simplify_ms=1.23 wall=709.59ms, shifted_quotient simplify=232.15ms avg_simplify_ms=2.32 wall=808.40ms, product simplify=142.39ms avg_simplify_ms=1.42 wall=485.60ms, difference simplify=107.53ms avg_simplify_ms=2.15 wall=332.67ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=808.40ms avg_case_ms=8.08 avg_simplify_ms=2.32, sum@0+100 failed=0 elapsed=512.24ms avg_case_ms=5.12 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=485.60ms avg_case_ms=4.86 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=332.67ms avg_case_ms=6.65 avg_simplify_ms=2.15, sum@700+100 failed=0 elapsed=197.35ms avg_case_ms=1.97 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.23ms median_wire=13.31ms median_wall=50.18ms, difference@0+50 #174 difference runs=3 median_simplify=12.05ms median_wire=12.11ms median_wall=45.44ms, sum@0+100 #173 sum runs=3 median_simplify=12.13ms median_wire=12.18ms median_wall=45.83ms, product@0+100 #175 product runs=3 median_simplify=11.96ms median_wire=12.02ms median_wall=45.74ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.71ms median_wire=10.79ms median_wall=40.94ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.51s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.10s | passed=1 failed=0 |
