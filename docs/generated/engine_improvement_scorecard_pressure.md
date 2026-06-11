# Engine Improvement Scorecard

- Generated: 2026-06-11T18:56:09.426072+00:00
- Git branch: main
- Git commit: `994930e90aed4a0d4b5192d2b978f1eab754286c`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=774.11ms avg_case_ms=7.74 simplify=219.23ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=685.14ms avg_case_ms=3.43 simplify=229.76ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=476.70ms avg_case_ms=4.77 simplify=137.08ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=328.25ms avg_case_ms=6.57 simplify=103.76ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=229.76ms avg_simplify_ms=1.15 wall=685.14ms, shifted_quotient simplify=219.23ms avg_simplify_ms=2.19 wall=774.11ms, product simplify=137.08ms avg_simplify_ms=1.37 wall=476.70ms, difference simplify=103.76ms avg_simplify_ms=2.08 wall=328.25ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=774.11ms avg_case_ms=7.74 avg_simplify_ms=2.19, sum@0+100 failed=0 elapsed=496.16ms avg_case_ms=4.96 avg_simplify_ms=1.60, product@0+100 failed=0 elapsed=476.70ms avg_case_ms=4.77 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=328.25ms avg_case_ms=6.57 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=188.98ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.86ms median_wire=12.93ms median_wall=49.12ms, product@0+100 #175 product runs=3 median_simplify=11.46ms median_wire=11.50ms median_wall=43.51ms, difference@0+50 #174 difference runs=3 median_simplify=11.28ms median_wire=11.33ms median_wall=43.28ms, sum@0+100 #173 sum runs=3 median_simplify=11.54ms median_wire=11.58ms median_wall=44.10ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.63ms median_wire=11.71ms median_wall=43.05ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.84s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.35s | passed=1 failed=0 |
