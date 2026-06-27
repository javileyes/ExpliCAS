# Engine Improvement Scorecard

- Generated: 2026-06-27T07:44:53.332727+00:00
- Git branch: main
- Git commit: `ba86bd31de77481d18f4a4734068dfcc30465b89`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=797.54ms avg_case_ms=7.98 simplify=228.63ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=710.21ms avg_case_ms=3.55 simplify=245.13ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=485.03ms avg_case_ms=4.85 simplify=142.42ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=334.18ms avg_case_ms=6.68 simplify=108.23ms avg_simplify_ms=2.16
- Engine hotspots: sum simplify=245.13ms avg_simplify_ms=1.23 wall=710.21ms, shifted_quotient simplify=228.63ms avg_simplify_ms=2.29 wall=797.54ms, product simplify=142.42ms avg_simplify_ms=1.42 wall=485.03ms, difference simplify=108.23ms avg_simplify_ms=2.16 wall=334.18ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=797.54ms avg_case_ms=7.98 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=512.50ms avg_case_ms=5.13 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=485.03ms avg_case_ms=4.85 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=334.18ms avg_case_ms=6.68 avg_simplify_ms=2.16, sum@700+100 failed=0 elapsed=197.71ms avg_case_ms=1.98 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.12ms median_wire=13.19ms median_wall=49.80ms, sum@0+100 #173 sum runs=3 median_simplify=11.82ms median_wire=11.87ms median_wall=44.62ms, difference@0+50 #174 difference runs=3 median_simplify=12.73ms median_wire=12.79ms median_wall=47.88ms, product@0+100 #175 product runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=44.89ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.88ms median_wire=10.95ms median_wall=41.47ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
