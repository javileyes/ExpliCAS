# Engine Improvement Scorecard

- Generated: 2026-06-12T00:36:05.149612+00:00
- Git branch: main
- Git commit: `a994b724be15739e5bd744642058654f9475eb96`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=776.40ms avg_case_ms=7.76 simplify=219.38ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=685.79ms avg_case_ms=3.43 simplify=230.25ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=495.84ms avg_case_ms=4.96 simplify=141.83ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=326.77ms avg_case_ms=6.54 simplify=102.87ms avg_simplify_ms=2.06
- Engine hotspots: sum simplify=230.25ms avg_simplify_ms=1.15 wall=685.79ms, shifted_quotient simplify=219.38ms avg_simplify_ms=2.19 wall=776.40ms, product simplify=141.83ms avg_simplify_ms=1.42 wall=495.84ms, difference simplify=102.87ms avg_simplify_ms=2.06 wall=326.77ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=776.40ms avg_case_ms=7.76 avg_simplify_ms=2.19, product@0+100 failed=0 elapsed=495.84ms avg_case_ms=4.96 avg_simplify_ms=1.42, sum@0+100 failed=0 elapsed=494.53ms avg_case_ms=4.95 avg_simplify_ms=1.59, difference@0+50 failed=0 elapsed=326.77ms avg_case_ms=6.54 avg_simplify_ms=2.06, sum@700+100 failed=0 elapsed=191.26ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.81ms median_wire=12.87ms median_wall=49.15ms, product@0+100 #175 product runs=3 median_simplify=11.40ms median_wire=11.45ms median_wall=43.47ms, difference@0+50 #174 difference runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=43.71ms, sum@0+100 #173 sum runs=3 median_simplify=11.59ms median_wire=11.63ms median_wall=43.87ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.58ms median_wire=10.65ms median_wall=40.14ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.11s | passed=1 failed=0 |
