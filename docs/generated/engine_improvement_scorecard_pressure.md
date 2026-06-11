# Engine Improvement Scorecard

- Generated: 2026-06-11T10:53:02.725253+00:00
- Git branch: main
- Git commit: `e1adceccc18c2f9bac2dc6e13ff9bc427fc74dcf`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=780.79ms avg_case_ms=7.81 simplify=220.59ms avg_simplify_ms=2.21, sum total=200 failed=0 elapsed=716.30ms avg_case_ms=3.58 simplify=241.44ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=480.95ms avg_case_ms=4.81 simplify=138.96ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=330.86ms avg_case_ms=6.62 simplify=105.41ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=241.44ms avg_simplify_ms=1.21 wall=716.30ms, shifted_quotient simplify=220.59ms avg_simplify_ms=2.21 wall=780.79ms, product simplify=138.96ms avg_simplify_ms=1.39 wall=480.95ms, difference simplify=105.41ms avg_simplify_ms=2.11 wall=330.86ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=780.79ms avg_case_ms=7.81 avg_simplify_ms=2.21, sum@0+100 failed=0 elapsed=524.35ms avg_case_ms=5.24 avg_simplify_ms=1.70, product@0+100 failed=0 elapsed=480.95ms avg_case_ms=4.81 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=330.86ms avg_case_ms=6.62 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=191.95ms avg_case_ms=1.92 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.21ms median_wire=13.28ms median_wall=53.83ms, product@0+100 #175 product runs=3 median_simplify=11.52ms median_wire=11.57ms median_wall=43.79ms, difference@0+50 #174 difference runs=3 median_simplify=11.48ms median_wire=11.53ms median_wall=43.69ms, sum@0+100 #173 sum runs=3 median_simplify=11.77ms median_wire=11.82ms median_wall=44.79ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.13ms median_wire=12.21ms median_wall=42.19ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.25s | passed=1 failed=0 |
