# Engine Improvement Scorecard

- Generated: 2026-07-24T17:52:51.776048+00:00
- Git branch: main
- Git commit: `c6bb991da635218042e7777b6500b29d886124c5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.45s avg_case_ms=14.51 simplify=399.16ms avg_simplify_ms=3.99, sum total=200 failed=0 elapsed=882.38ms avg_case_ms=4.41 simplify=292.75ms avg_simplify_ms=1.46, product total=100 failed=0 elapsed=612.38ms avg_case_ms=6.12 simplify=176.53ms avg_simplify_ms=1.77, difference total=50 failed=0 elapsed=406.09ms avg_case_ms=8.12 simplify=123.85ms avg_simplify_ms=2.48
- Engine hotspots: shifted_quotient simplify=399.16ms avg_simplify_ms=3.99 wall=1.45s, sum simplify=292.75ms avg_simplify_ms=1.46 wall=882.38ms, product simplify=176.53ms avg_simplify_ms=1.77 wall=612.38ms, difference simplify=123.85ms avg_simplify_ms=2.48 wall=406.09ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.45s avg_case_ms=14.51 avg_simplify_ms=3.99, sum@0+100 failed=0 elapsed=650.42ms avg_case_ms=6.50 avg_simplify_ms=2.10, product@0+100 failed=0 elapsed=612.38ms avg_case_ms=6.12 avg_simplify_ms=1.77, difference@0+50 failed=0 elapsed=406.09ms avg_case_ms=8.12 avg_simplify_ms=2.48, sum@700+100 failed=0 elapsed=231.96ms avg_case_ms=2.32 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #224 shifted_quotient runs=3 median_simplify=7.59ms median_wire=7.76ms median_wall=90.24ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=63.60ms median_wire=63.68ms median_wall=298.61ms, shifted_quotient@0+100 #324 shifted_quotient runs=3 median_simplify=1.93ms median_wire=1.98ms median_wall=6.09ms, product@0+100 #175 product runs=3 median_simplify=45.98ms median_wire=46.04ms median_wall=145.47ms, sum@0+100 #173 sum runs=3 median_simplify=15.81ms median_wire=15.87ms median_wall=230.90ms
- Steady-state dominant expressions: shifted_quotient@0+100 #224 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((sqrt(a^2 + 2*a*b + b^2) - abs(a+b)) + 1), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), shifted_quotient@0+100 #324 shifted_quotient expr=((tan(x) + cot(x) - sec(x)*csc(x)) + 1)/((1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)) + 1)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 3.35s | passed=450 failed=0 total=450 avg_case=7.444ms |
| `calculus_diff_exhaustive_contract` | `pass` | 21.01s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
