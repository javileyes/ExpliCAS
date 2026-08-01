# Engine Improvement Scorecard

- Generated: 2026-08-01T15:01:14.671593+00:00
- Git branch: main
- Git commit: `dc65a9b0eb526f9816db36fe6da3df4e87d4c7a7`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=391

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=973.44ms avg_case_ms=9.73 simplify=271.66ms avg_simplify_ms=2.72, sum total=200 failed=0 elapsed=848.75ms avg_case_ms=4.24 simplify=270.88ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=602.38ms avg_case_ms=6.02 simplify=172.77ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=394.75ms avg_case_ms=7.90 simplify=115.62ms avg_simplify_ms=2.31
- Engine hotspots: shifted_quotient simplify=271.66ms avg_simplify_ms=2.72 wall=973.44ms, sum simplify=270.88ms avg_simplify_ms=1.35 wall=848.75ms, product simplify=172.77ms avg_simplify_ms=1.73 wall=602.38ms, difference simplify=115.62ms avg_simplify_ms=2.31 wall=394.75ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=973.44ms avg_case_ms=9.73 avg_simplify_ms=2.72, sum@0+100 failed=0 elapsed=617.54ms avg_case_ms=6.18 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=602.38ms avg_case_ms=6.02 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=394.75ms avg_case_ms=7.90 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=231.21ms avg_case_ms=2.31 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.76ms median_wire=16.82ms median_wall=64.33ms, sum@0+100 #173 sum runs=3 median_simplify=15.33ms median_wire=15.38ms median_wall=57.84ms, product@0+100 #175 product runs=3 median_simplify=15.13ms median_wire=15.18ms median_wall=57.72ms, difference@0+50 #174 difference runs=3 median_simplify=15.15ms median_wire=15.20ms median_wall=57.61ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.92ms median_wire=12.99ms median_wall=49.10ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.82s | passed=450 failed=0 total=450 avg_case=6.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.17s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.74s | passed=1 failed=0 |
