# Engine Improvement Scorecard

- Generated: 2026-07-07T09:59:12.397216+00:00
- Git branch: main
- Git commit: `82e5c0e0e27127276f9c2aad702d448bb9f1933b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=925.69ms avg_case_ms=9.26 simplify=256.40ms avg_simplify_ms=2.56, sum total=200 failed=0 elapsed=818.40ms avg_case_ms=4.09 simplify=263.95ms avg_simplify_ms=1.32, product total=100 failed=0 elapsed=574.19ms avg_case_ms=5.74 simplify=163.08ms avg_simplify_ms=1.63, difference total=50 failed=0 elapsed=382.34ms avg_case_ms=7.65 simplify=115.77ms avg_simplify_ms=2.32
- Engine hotspots: sum simplify=263.95ms avg_simplify_ms=1.32 wall=818.40ms, shifted_quotient simplify=256.40ms avg_simplify_ms=2.56 wall=925.69ms, product simplify=163.08ms avg_simplify_ms=1.63 wall=574.19ms, difference simplify=115.77ms avg_simplify_ms=2.32 wall=382.34ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=925.69ms avg_case_ms=9.26 avg_simplify_ms=2.56, sum@0+100 failed=0 elapsed=596.98ms avg_case_ms=5.97 avg_simplify_ms=1.86, product@0+100 failed=0 elapsed=574.19ms avg_case_ms=5.74 avg_simplify_ms=1.63, difference@0+50 failed=0 elapsed=382.34ms avg_case_ms=7.65 avg_simplify_ms=2.32, sum@700+100 failed=0 elapsed=221.42ms avg_case_ms=2.21 avg_simplify_ms=0.78
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.18ms median_wire=16.25ms median_wall=61.74ms, difference@0+50 #174 difference runs=3 median_simplify=14.62ms median_wire=14.66ms median_wall=55.98ms, sum@0+100 #173 sum runs=3 median_simplify=14.73ms median_wire=14.78ms median_wall=56.50ms, product@0+100 #175 product runs=3 median_simplify=14.56ms median_wire=14.61ms median_wall=56.07ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.51ms median_wire=12.57ms median_wall=47.24ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.70s | passed=450 failed=0 total=450 avg_case=6.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.94s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.84s | passed=1 failed=0 |
