# Engine Improvement Scorecard

- Generated: 2026-07-03T22:52:54.654322+00:00
- Git branch: main
- Git commit: `fb67706f5042d637ccbcad4fe6802e2b9dad2fc1`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=355

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=926.07ms avg_case_ms=9.26 simplify=257.58ms avg_simplify_ms=2.58, sum total=200 failed=0 elapsed=825.16ms avg_case_ms=4.13 simplify=266.56ms avg_simplify_ms=1.33, product total=100 failed=0 elapsed=579.59ms avg_case_ms=5.80 simplify=165.90ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=380.83ms avg_case_ms=7.62 simplify=115.70ms avg_simplify_ms=2.31
- Engine hotspots: sum simplify=266.56ms avg_simplify_ms=1.33 wall=825.16ms, shifted_quotient simplify=257.58ms avg_simplify_ms=2.58 wall=926.07ms, product simplify=165.90ms avg_simplify_ms=1.66 wall=579.59ms, difference simplify=115.70ms avg_simplify_ms=2.31 wall=380.83ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=926.07ms avg_case_ms=9.26 avg_simplify_ms=2.58, sum@0+100 failed=0 elapsed=601.97ms avg_case_ms=6.02 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=579.59ms avg_case_ms=5.80 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=380.83ms avg_case_ms=7.62 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=223.20ms avg_case_ms=2.23 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.22ms median_wire=16.29ms median_wall=61.89ms, sum@0+100 #173 sum runs=3 median_simplify=15.04ms median_wire=15.09ms median_wall=57.51ms, difference@0+50 #174 difference runs=3 median_simplify=14.65ms median_wire=14.69ms median_wall=56.46ms, product@0+100 #175 product runs=3 median_simplify=14.55ms median_wire=14.60ms median_wall=60.67ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.38ms median_wire=12.44ms median_wall=47.36ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.71s | passed=450 failed=0 total=450 avg_case=6.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
