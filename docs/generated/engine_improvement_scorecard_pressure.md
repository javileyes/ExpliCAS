# Engine Improvement Scorecard

- Generated: 2026-07-14T10:22:33.867460+00:00
- Git branch: main
- Git commit: `c5b7bf6f55ee1d7e15b213a0c3dbd5c810db21b8`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=955.47ms avg_case_ms=9.55 simplify=262.38ms avg_simplify_ms=2.62, sum total=200 failed=0 elapsed=855.21ms avg_case_ms=4.28 simplify=275.50ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=580.73ms avg_case_ms=5.81 simplify=165.64ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=390.64ms avg_case_ms=7.81 simplify=118.43ms avg_simplify_ms=2.37
- Engine hotspots: sum simplify=275.50ms avg_simplify_ms=1.38 wall=855.21ms, shifted_quotient simplify=262.38ms avg_simplify_ms=2.62 wall=955.47ms, product simplify=165.64ms avg_simplify_ms=1.66 wall=580.73ms, difference simplify=118.43ms avg_simplify_ms=2.37 wall=390.64ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=955.47ms avg_case_ms=9.55 avg_simplify_ms=2.62, sum@0+100 failed=0 elapsed=630.55ms avg_case_ms=6.31 avg_simplify_ms=1.95, product@0+100 failed=0 elapsed=580.73ms avg_case_ms=5.81 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=390.64ms avg_case_ms=7.81 avg_simplify_ms=2.37, sum@700+100 failed=0 elapsed=224.65ms avg_case_ms=2.25 avg_simplify_ms=0.80
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.43ms median_wire=15.49ms median_wall=59.08ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.30ms median_wire=16.37ms median_wall=63.53ms, difference@0+50 #174 difference runs=3 median_simplify=14.76ms median_wire=14.80ms median_wall=56.55ms, product@0+100 #175 product runs=3 median_simplify=16.21ms median_wire=16.27ms median_wall=60.52ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.45ms median_wire=12.53ms median_wall=49.62ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.78s | passed=450 failed=0 total=450 avg_case=6.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.39s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
