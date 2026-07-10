# Engine Improvement Scorecard

- Generated: 2026-07-10T11:23:41.417853+00:00
- Git branch: main
- Git commit: `a3f9501b8a6eaa92570379ebbb91c17fc4061e6d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=936.89ms avg_case_ms=9.37 simplify=259.35ms avg_simplify_ms=2.59, sum total=200 failed=0 elapsed=824.85ms avg_case_ms=4.12 simplify=266.30ms avg_simplify_ms=1.33, product total=100 failed=0 elapsed=573.05ms avg_case_ms=5.73 simplify=162.57ms avg_simplify_ms=1.63, difference total=50 failed=0 elapsed=383.02ms avg_case_ms=7.66 simplify=115.14ms avg_simplify_ms=2.30
- Engine hotspots: sum simplify=266.30ms avg_simplify_ms=1.33 wall=824.85ms, shifted_quotient simplify=259.35ms avg_simplify_ms=2.59 wall=936.89ms, product simplify=162.57ms avg_simplify_ms=1.63 wall=573.05ms, difference simplify=115.14ms avg_simplify_ms=2.30 wall=383.02ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=936.89ms avg_case_ms=9.37 avg_simplify_ms=2.59, sum@0+100 failed=0 elapsed=604.59ms avg_case_ms=6.05 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=573.05ms avg_case_ms=5.73 avg_simplify_ms=1.63, difference@0+50 failed=0 elapsed=383.02ms avg_case_ms=7.66 avg_simplify_ms=2.30, sum@700+100 failed=0 elapsed=220.25ms avg_case_ms=2.20 avg_simplify_ms=0.78
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.91ms median_wire=16.98ms median_wall=64.44ms, sum@0+100 #173 sum runs=3 median_simplify=15.37ms median_wire=15.41ms median_wall=59.81ms, product@0+100 #175 product runs=3 median_simplify=15.40ms median_wire=15.45ms median_wall=58.98ms, difference@0+50 #174 difference runs=3 median_simplify=15.69ms median_wire=15.74ms median_wall=60.30ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.42ms median_wire=13.49ms median_wall=51.00ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.72s | passed=450 failed=0 total=450 avg_case=6.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
