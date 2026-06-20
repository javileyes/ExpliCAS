# Engine Improvement Scorecard

- Generated: 2026-06-20T19:57:28.877382+00:00
- Git branch: main
- Git commit: `3d9a7835b253beca1ca1dbe089dccacf25c39ab3`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=787.82ms avg_case_ms=7.88 simplify=225.40ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=694.07ms avg_case_ms=3.47 simplify=234.65ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=480.04ms avg_case_ms=4.80 simplify=138.08ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=324.40ms avg_case_ms=6.49 simplify=103.92ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=234.65ms avg_simplify_ms=1.17 wall=694.07ms, shifted_quotient simplify=225.40ms avg_simplify_ms=2.25 wall=787.82ms, product simplify=138.08ms avg_simplify_ms=1.38 wall=480.04ms, difference simplify=103.92ms avg_simplify_ms=2.08 wall=324.40ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=787.82ms avg_case_ms=7.88 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=503.58ms avg_case_ms=5.04 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=480.04ms avg_case_ms=4.80 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=324.40ms avg_case_ms=6.49 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=190.49ms avg_case_ms=1.90 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.22ms median_wire=13.28ms median_wall=49.94ms, sum@0+100 #173 sum runs=3 median_simplify=11.62ms median_wire=11.67ms median_wall=44.18ms, product@0+100 #175 product runs=3 median_simplify=11.59ms median_wire=11.64ms median_wall=43.96ms, difference@0+50 #174 difference runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=44.21ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.51ms median_wire=10.58ms median_wall=39.89ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
