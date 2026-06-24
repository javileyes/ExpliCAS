# Engine Improvement Scorecard

- Generated: 2026-06-24T14:29:21.084168+00:00
- Git branch: main
- Git commit: `f89fdeaf9890843616c31eb41c359993027d64ff`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=862.76ms avg_case_ms=8.63 simplify=245.76ms avg_simplify_ms=2.46, sum total=200 failed=0 elapsed=767.28ms avg_case_ms=3.84 simplify=260.54ms avg_simplify_ms=1.30, product total=100 failed=0 elapsed=523.55ms avg_case_ms=5.24 simplify=150.88ms avg_simplify_ms=1.51, difference total=50 failed=0 elapsed=360.10ms avg_case_ms=7.20 simplify=114.44ms avg_simplify_ms=2.29
- Engine hotspots: sum simplify=260.54ms avg_simplify_ms=1.30 wall=767.28ms, shifted_quotient simplify=245.76ms avg_simplify_ms=2.46 wall=862.76ms, product simplify=150.88ms avg_simplify_ms=1.51 wall=523.55ms, difference simplify=114.44ms avg_simplify_ms=2.29 wall=360.10ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=862.76ms avg_case_ms=8.63 avg_simplify_ms=2.46, sum@0+100 failed=0 elapsed=555.87ms avg_case_ms=5.56 avg_simplify_ms=1.82, product@0+100 failed=0 elapsed=523.55ms avg_case_ms=5.24 avg_simplify_ms=1.51, difference@0+50 failed=0 elapsed=360.10ms avg_case_ms=7.20 avg_simplify_ms=2.29, sum@700+100 failed=0 elapsed=211.42ms avg_case_ms=2.11 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=14.18ms median_wire=14.27ms median_wall=54.69ms, sum@0+100 #173 sum runs=3 median_simplify=12.84ms median_wire=12.90ms median_wall=48.52ms, difference@0+50 #174 difference runs=3 median_simplify=12.68ms median_wire=12.74ms median_wall=48.51ms, product@0+100 #175 product runs=3 median_simplify=12.82ms median_wire=12.88ms median_wall=48.32ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.89ms median_wire=11.97ms median_wall=45.21ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.51s | passed=450 failed=0 total=450 avg_case=5.578ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
