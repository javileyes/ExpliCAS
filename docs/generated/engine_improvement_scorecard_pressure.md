# Engine Improvement Scorecard

- Generated: 2026-07-10T09:19:03.825006+00:00
- Git branch: main
- Git commit: `d43c20d484efcfeb1a61f66b945b138680831bfb`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=931.66ms avg_case_ms=9.32 simplify=257.22ms avg_simplify_ms=2.57, sum total=200 failed=0 elapsed=831.63ms avg_case_ms=4.16 simplify=267.92ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=584.14ms avg_case_ms=5.84 simplify=165.28ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=384.18ms avg_case_ms=7.68 simplify=116.27ms avg_simplify_ms=2.33
- Engine hotspots: sum simplify=267.92ms avg_simplify_ms=1.34 wall=831.63ms, shifted_quotient simplify=257.22ms avg_simplify_ms=2.57 wall=931.66ms, product simplify=165.28ms avg_simplify_ms=1.65 wall=584.14ms, difference simplify=116.27ms avg_simplify_ms=2.33 wall=384.18ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=931.66ms avg_case_ms=9.32 avg_simplify_ms=2.57, sum@0+100 failed=0 elapsed=606.16ms avg_case_ms=6.06 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=584.14ms avg_case_ms=5.84 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=384.18ms avg_case_ms=7.68 avg_simplify_ms=2.33, sum@700+100 failed=0 elapsed=225.46ms avg_case_ms=2.25 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.53ms median_wire=16.59ms median_wall=68.96ms, sum@0+100 #173 sum runs=3 median_simplify=16.29ms median_wire=16.34ms median_wall=61.92ms, product@0+100 #175 product runs=3 median_simplify=15.38ms median_wire=15.42ms median_wall=59.07ms, difference@0+50 #174 difference runs=3 median_simplify=15.81ms median_wire=15.86ms median_wall=61.82ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.42ms median_wire=12.49ms median_wall=47.69ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.73s | passed=450 failed=0 total=450 avg_case=6.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.95s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
