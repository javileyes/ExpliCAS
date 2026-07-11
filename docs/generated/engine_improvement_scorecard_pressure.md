# Engine Improvement Scorecard

- Generated: 2026-07-11T00:39:53.688316+00:00
- Git branch: main
- Git commit: `dc0fba737766dec969bde427a9c46f0a0405ed20`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=960.94ms avg_case_ms=9.61 simplify=266.93ms avg_simplify_ms=2.67, sum total=200 failed=0 elapsed=836.98ms avg_case_ms=4.18 simplify=269.72ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=591.15ms avg_case_ms=5.91 simplify=167.98ms avg_simplify_ms=1.68, difference total=50 failed=0 elapsed=389.79ms avg_case_ms=7.80 simplify=117.72ms avg_simplify_ms=2.35
- Engine hotspots: sum simplify=269.72ms avg_simplify_ms=1.35 wall=836.98ms, shifted_quotient simplify=266.93ms avg_simplify_ms=2.67 wall=960.94ms, product simplify=167.98ms avg_simplify_ms=1.68 wall=591.15ms, difference simplify=117.72ms avg_simplify_ms=2.35 wall=389.79ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=960.94ms avg_case_ms=9.61 avg_simplify_ms=2.67, sum@0+100 failed=0 elapsed=611.34ms avg_case_ms=6.11 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=591.15ms avg_case_ms=5.91 avg_simplify_ms=1.68, difference@0+50 failed=0 elapsed=389.79ms avg_case_ms=7.80 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=225.64ms avg_case_ms=2.26 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.39ms median_wire=16.46ms median_wall=63.65ms, sum@0+100 #173 sum runs=3 median_simplify=14.95ms median_wire=14.99ms median_wall=57.27ms, product@0+100 #175 product runs=3 median_simplify=14.88ms median_wire=14.93ms median_wall=56.58ms, difference@0+50 #174 difference runs=3 median_simplify=15.78ms median_wire=15.82ms median_wall=59.63ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.00ms median_wire=13.07ms median_wall=49.99ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.78s | passed=450 failed=0 total=450 avg_case=6.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.99s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
