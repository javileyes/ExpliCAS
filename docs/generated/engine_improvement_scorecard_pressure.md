# Engine Improvement Scorecard

- Generated: 2026-07-13T17:37:16.307950+00:00
- Git branch: main
- Git commit: `7b968349811913529598556b46d71883520ce669`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=977.20ms avg_case_ms=9.77 simplify=270.09ms avg_simplify_ms=2.70, sum total=200 failed=0 elapsed=856.12ms avg_case_ms=4.28 simplify=275.40ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=580.39ms avg_case_ms=5.80 simplify=165.32ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=390.81ms avg_case_ms=7.82 simplify=118.23ms avg_simplify_ms=2.36
- Engine hotspots: sum simplify=275.40ms avg_simplify_ms=1.38 wall=856.12ms, shifted_quotient simplify=270.09ms avg_simplify_ms=2.70 wall=977.20ms, product simplify=165.32ms avg_simplify_ms=1.65 wall=580.39ms, difference simplify=118.23ms avg_simplify_ms=2.36 wall=390.81ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=977.20ms avg_case_ms=9.77 avg_simplify_ms=2.70, sum@0+100 failed=0 elapsed=631.85ms avg_case_ms=6.32 avg_simplify_ms=1.96, product@0+100 failed=0 elapsed=580.39ms avg_case_ms=5.80 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=390.81ms avg_case_ms=7.82 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=224.26ms avg_case_ms=2.24 avg_simplify_ms=0.80
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=14.85ms median_wire=14.90ms median_wall=57.21ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.56ms median_wire=16.64ms median_wall=64.29ms, difference@0+50 #174 difference runs=3 median_simplify=14.98ms median_wire=15.02ms median_wall=57.79ms, product@0+100 #175 product runs=3 median_simplify=16.32ms median_wire=16.37ms median_wall=59.62ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.44ms median_wire=12.51ms median_wall=47.16ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.80s | passed=450 failed=0 total=450 avg_case=6.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.30s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
