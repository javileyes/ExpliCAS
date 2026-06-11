# Engine Improvement Scorecard

- Generated: 2026-06-11T10:17:18.977694+00:00
- Git branch: main
- Git commit: `91b523c1f3268bb8fa2555102a0b31f67dc503c5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=787.09ms avg_case_ms=7.87 simplify=223.39ms avg_simplify_ms=2.23, sum total=200 failed=0 elapsed=701.38ms avg_case_ms=3.51 simplify=233.86ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=472.76ms avg_case_ms=4.73 simplify=135.87ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=319.81ms avg_case_ms=6.40 simplify=101.49ms avg_simplify_ms=2.03
- Engine hotspots: sum simplify=233.86ms avg_simplify_ms=1.17 wall=701.38ms, shifted_quotient simplify=223.39ms avg_simplify_ms=2.23 wall=787.09ms, product simplify=135.87ms avg_simplify_ms=1.36 wall=472.76ms, difference simplify=101.49ms avg_simplify_ms=2.03 wall=319.81ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=787.09ms avg_case_ms=7.87 avg_simplify_ms=2.23, sum@0+100 failed=0 elapsed=511.56ms avg_case_ms=5.12 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=472.76ms avg_case_ms=4.73 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=319.81ms avg_case_ms=6.40 avg_simplify_ms=2.03, sum@700+100 failed=0 elapsed=189.83ms avg_case_ms=1.90 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.84ms median_wire=12.91ms median_wall=49.01ms, product@0+100 #175 product runs=3 median_simplify=11.45ms median_wire=11.50ms median_wall=44.05ms, sum@0+100 #173 sum runs=3 median_simplify=11.52ms median_wire=11.57ms median_wall=44.09ms, difference@0+50 #174 difference runs=3 median_simplify=11.72ms median_wire=11.76ms median_wall=44.42ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.96ms median_wire=13.04ms median_wall=44.59ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.16s | passed=1 failed=0 |
