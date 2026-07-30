# Engine Improvement Scorecard

- Generated: 2026-07-30T10:59:00.276437+00:00
- Git branch: main
- Git commit: `1b87055ad4c235ac9f2837b173cd55a2fbf29bfb`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=987.27ms avg_case_ms=9.87 simplify=280.37ms avg_simplify_ms=2.80, sum total=200 failed=0 elapsed=873.94ms avg_case_ms=4.37 simplify=281.46ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=621.89ms avg_case_ms=6.22 simplify=180.82ms avg_simplify_ms=1.81, difference total=50 failed=0 elapsed=400.59ms avg_case_ms=8.01 simplify=119.11ms avg_simplify_ms=2.38
- Engine hotspots: sum simplify=281.46ms avg_simplify_ms=1.41 wall=873.94ms, shifted_quotient simplify=280.37ms avg_simplify_ms=2.80 wall=987.27ms, product simplify=180.82ms avg_simplify_ms=1.81 wall=621.89ms, difference simplify=119.11ms avg_simplify_ms=2.38 wall=400.59ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=987.27ms avg_case_ms=9.87 avg_simplify_ms=2.80, sum@0+100 failed=0 elapsed=638.86ms avg_case_ms=6.39 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=621.89ms avg_case_ms=6.22 avg_simplify_ms=1.81, difference@0+50 failed=0 elapsed=400.59ms avg_case_ms=8.01 avg_simplify_ms=2.38, sum@700+100 failed=0 elapsed=235.08ms avg_case_ms=2.35 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.11ms median_wire=16.18ms median_wall=62.87ms, sum@0+100 #173 sum runs=3 median_simplify=15.37ms median_wire=15.42ms median_wall=58.94ms, difference@0+50 #174 difference runs=3 median_simplify=15.25ms median_wire=15.31ms median_wall=57.91ms, product@0+100 #175 product runs=3 median_simplify=15.41ms median_wire=15.46ms median_wall=58.96ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.39ms median_wire=13.46ms median_wall=50.34ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.76s | passed=1 failed=0 |
