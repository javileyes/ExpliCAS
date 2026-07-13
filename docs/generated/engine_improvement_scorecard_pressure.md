# Engine Improvement Scorecard

- Generated: 2026-07-13T08:33:58.574808+00:00
- Git branch: main
- Git commit: `3e93d80500bb882a4c86f794f2f18a29e3e0893f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=947.36ms avg_case_ms=9.47 simplify=263.67ms avg_simplify_ms=2.64, sum total=200 failed=0 elapsed=872.93ms avg_case_ms=4.36 simplify=287.78ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=596.42ms avg_case_ms=5.96 simplify=169.60ms avg_simplify_ms=1.70, difference total=50 failed=0 elapsed=401.43ms avg_case_ms=8.03 simplify=124.01ms avg_simplify_ms=2.48
- Engine hotspots: sum simplify=287.78ms avg_simplify_ms=1.44 wall=872.93ms, shifted_quotient simplify=263.67ms avg_simplify_ms=2.64 wall=947.36ms, product simplify=169.60ms avg_simplify_ms=1.70 wall=596.42ms, difference simplify=124.01ms avg_simplify_ms=2.48 wall=401.43ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=947.36ms avg_case_ms=9.47 avg_simplify_ms=2.64, sum@0+100 failed=0 elapsed=636.69ms avg_case_ms=6.37 avg_simplify_ms=2.01, product@0+100 failed=0 elapsed=596.42ms avg_case_ms=5.96 avg_simplify_ms=1.70, difference@0+50 failed=0 elapsed=401.43ms avg_case_ms=8.03 avg_simplify_ms=2.48, sum@700+100 failed=0 elapsed=236.24ms avg_case_ms=2.36 avg_simplify_ms=0.86
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=14.96ms median_wire=15.01ms median_wall=57.16ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.59ms median_wire=16.66ms median_wall=63.70ms, difference@0+50 #174 difference runs=3 median_simplify=15.35ms median_wire=15.41ms median_wall=66.74ms, product@0+100 #175 product runs=3 median_simplify=15.22ms median_wire=15.28ms median_wall=58.54ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.82ms median_wire=12.89ms median_wall=48.65ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.82s | passed=450 failed=0 total=450 avg_case=6.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.06s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
