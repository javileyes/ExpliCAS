# Engine Improvement Scorecard

- Generated: 2026-07-24T22:07:04.438008+00:00
- Git branch: main
- Git commit: `228f558f36271fc709b49b053d84b2f9dc2d7586`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=991.80ms avg_case_ms=9.92 simplify=278.42ms avg_simplify_ms=2.78, sum total=200 failed=0 elapsed=880.08ms avg_case_ms=4.40 simplify=285.71ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=616.57ms avg_case_ms=6.17 simplify=176.81ms avg_simplify_ms=1.77, difference total=50 failed=0 elapsed=398.87ms avg_case_ms=7.98 simplify=122.07ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=285.71ms avg_simplify_ms=1.43 wall=880.08ms, shifted_quotient simplify=278.42ms avg_simplify_ms=2.78 wall=991.80ms, product simplify=176.81ms avg_simplify_ms=1.77 wall=616.57ms, difference simplify=122.07ms avg_simplify_ms=2.44 wall=398.87ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=991.80ms avg_case_ms=9.92 avg_simplify_ms=2.78, sum@0+100 failed=0 elapsed=644.81ms avg_case_ms=6.45 avg_simplify_ms=2.02, product@0+100 failed=0 elapsed=616.57ms avg_case_ms=6.17 avg_simplify_ms=1.77, difference@0+50 failed=0 elapsed=398.87ms avg_case_ms=7.98 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=235.27ms avg_case_ms=2.35 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.85ms median_wire=16.93ms median_wall=66.73ms, sum@0+100 #173 sum runs=3 median_simplify=15.78ms median_wire=15.83ms median_wall=59.99ms, difference@0+50 #174 difference runs=3 median_simplify=15.35ms median_wire=15.40ms median_wall=58.89ms, product@0+100 #175 product runs=3 median_simplify=15.21ms median_wire=15.26ms median_wall=58.77ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.16ms median_wire=13.23ms median_wall=50.16ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.62s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
