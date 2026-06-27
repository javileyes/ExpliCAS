# Engine Improvement Scorecard

- Generated: 2026-06-27T10:19:05.976989+00:00
- Git branch: main
- Git commit: `6b1c10267de8fc048856b14a15932377dc47b4b5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=797.97ms avg_case_ms=7.98 simplify=228.36ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=711.97ms avg_case_ms=3.56 simplify=245.57ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=492.90ms avg_case_ms=4.93 simplify=144.43ms avg_simplify_ms=1.44, difference total=50 failed=0 elapsed=333.59ms avg_case_ms=6.67 simplify=107.63ms avg_simplify_ms=2.15
- Engine hotspots: sum simplify=245.57ms avg_simplify_ms=1.23 wall=711.97ms, shifted_quotient simplify=228.36ms avg_simplify_ms=2.28 wall=797.97ms, product simplify=144.43ms avg_simplify_ms=1.44 wall=492.90ms, difference simplify=107.63ms avg_simplify_ms=2.15 wall=333.59ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=797.97ms avg_case_ms=7.98 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=512.67ms avg_case_ms=5.13 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=492.90ms avg_case_ms=4.93 avg_simplify_ms=1.44, difference@0+50 failed=0 elapsed=333.59ms avg_case_ms=6.67 avg_simplify_ms=2.15, sum@700+100 failed=0 elapsed=199.30ms avg_case_ms=1.99 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.25ms median_wall=49.99ms, difference@0+50 #174 difference runs=3 median_simplify=11.85ms median_wire=11.90ms median_wall=44.98ms, product@0+100 #175 product runs=3 median_simplify=11.58ms median_wire=11.63ms median_wall=44.11ms, sum@0+100 #173 sum runs=3 median_simplify=11.60ms median_wire=11.65ms median_wall=44.72ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.74ms median_wire=10.82ms median_wall=41.10ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.08s | passed=1 failed=0 |
