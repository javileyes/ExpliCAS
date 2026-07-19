# Engine Improvement Scorecard

- Generated: 2026-07-19T16:13:59.448770+00:00
- Git branch: main
- Git commit: `1f6b95e6a53c0dc27cbc4ca8e9fe903fb5648a98`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=974.88ms avg_case_ms=9.75 simplify=271.49ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=889.97ms avg_case_ms=4.45 simplify=291.78ms avg_simplify_ms=1.46, product total=100 failed=0 elapsed=605.23ms avg_case_ms=6.05 simplify=173.72ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=403.84ms avg_case_ms=8.08 simplify=122.62ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=291.78ms avg_simplify_ms=1.46 wall=889.97ms, shifted_quotient simplify=271.49ms avg_simplify_ms=2.71 wall=974.88ms, product simplify=173.72ms avg_simplify_ms=1.74 wall=605.23ms, difference simplify=122.62ms avg_simplify_ms=2.45 wall=403.84ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=974.88ms avg_case_ms=9.75 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=655.32ms avg_case_ms=6.55 avg_simplify_ms=2.07, product@0+100 failed=0 elapsed=605.23ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=403.84ms avg_case_ms=8.08 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=234.65ms avg_case_ms=2.35 avg_simplify_ms=0.85
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.27ms median_wire=15.32ms median_wall=57.79ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.86ms median_wire=16.94ms median_wall=65.34ms, difference@0+50 #174 difference runs=3 median_simplify=21.62ms median_wire=21.68ms median_wall=73.43ms, product@0+100 #175 product runs=3 median_simplify=15.95ms median_wire=16.02ms median_wall=61.10ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.86ms median_wire=12.93ms median_wall=48.85ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.95s | passed=1 failed=0 |
