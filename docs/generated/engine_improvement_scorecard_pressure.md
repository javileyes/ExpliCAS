# Engine Improvement Scorecard

- Generated: 2026-06-20T15:11:41.490003+00:00
- Git branch: main
- Git commit: `985a904a778f1ccb4ad6a69e5507a3ff86676256`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=814.92ms avg_case_ms=8.15 simplify=234.11ms avg_simplify_ms=2.34, sum total=200 failed=0 elapsed=709.03ms avg_case_ms=3.55 simplify=240.87ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=481.82ms avg_case_ms=4.82 simplify=138.97ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=337.24ms avg_case_ms=6.74 simplify=108.62ms avg_simplify_ms=2.17
- Engine hotspots: sum simplify=240.87ms avg_simplify_ms=1.20 wall=709.03ms, shifted_quotient simplify=234.11ms avg_simplify_ms=2.34 wall=814.92ms, product simplify=138.97ms avg_simplify_ms=1.39 wall=481.82ms, difference simplify=108.62ms avg_simplify_ms=2.17 wall=337.24ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=814.92ms avg_case_ms=8.15 avg_simplify_ms=2.34, sum@0+100 failed=0 elapsed=513.33ms avg_case_ms=5.13 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=481.82ms avg_case_ms=4.82 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=337.24ms avg_case_ms=6.74 avg_simplify_ms=2.17, sum@700+100 failed=0 elapsed=195.70ms avg_case_ms=1.96 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.67ms median_wire=13.75ms median_wall=51.70ms, sum@0+100 #173 sum runs=3 median_simplify=12.27ms median_wire=12.33ms median_wall=46.53ms, difference@0+50 #174 difference runs=3 median_simplify=12.19ms median_wire=12.25ms median_wall=46.11ms, product@0+100 #175 product runs=3 median_simplify=11.90ms median_wire=11.95ms median_wall=45.90ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.78ms median_wire=10.85ms median_wall=40.89ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
