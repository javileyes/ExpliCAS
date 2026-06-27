# Engine Improvement Scorecard

- Generated: 2026-06-27T19:34:06.007465+00:00
- Git branch: main
- Git commit: `29de42c6a699f7663591c590cc8f25a53c92a261`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=793.14ms avg_case_ms=7.93 simplify=226.20ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=703.96ms avg_case_ms=3.52 simplify=243.60ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=485.90ms avg_case_ms=4.86 simplify=142.57ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=331.98ms avg_case_ms=6.64 simplify=106.91ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=243.60ms avg_simplify_ms=1.22 wall=703.96ms, shifted_quotient simplify=226.20ms avg_simplify_ms=2.26 wall=793.14ms, product simplify=142.57ms avg_simplify_ms=1.43 wall=485.90ms, difference simplify=106.91ms avg_simplify_ms=2.14 wall=331.98ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=793.14ms avg_case_ms=7.93 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=507.47ms avg_case_ms=5.07 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=485.90ms avg_case_ms=4.86 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=331.98ms avg_case_ms=6.64 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=196.50ms avg_case_ms=1.96 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.93ms median_wire=13.00ms median_wall=49.53ms, sum@0+100 #173 sum runs=3 median_simplify=11.68ms median_wire=11.73ms median_wall=44.49ms, product@0+100 #175 product runs=3 median_simplify=11.72ms median_wire=11.76ms median_wall=44.92ms, difference@0+50 #174 difference runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=44.84ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.77ms median_wire=10.85ms median_wall=40.74ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
