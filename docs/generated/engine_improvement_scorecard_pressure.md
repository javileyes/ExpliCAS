# Engine Improvement Scorecard

- Generated: 2026-07-17T13:04:26.146963+00:00
- Git branch: main
- Git commit: `cfee09381ca84688ed4032ca586ba12193b83001`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=982.98ms avg_case_ms=9.83 simplify=278.12ms avg_simplify_ms=2.78, sum total=200 failed=0 elapsed=880.35ms avg_case_ms=4.40 simplify=281.58ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=601.10ms avg_case_ms=6.01 simplify=172.20ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=396.18ms avg_case_ms=7.92 simplify=120.07ms avg_simplify_ms=2.40
- Engine hotspots: sum simplify=281.58ms avg_simplify_ms=1.41 wall=880.35ms, shifted_quotient simplify=278.12ms avg_simplify_ms=2.78 wall=982.98ms, product simplify=172.20ms avg_simplify_ms=1.72 wall=601.10ms, difference simplify=120.07ms avg_simplify_ms=2.40 wall=396.18ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=982.98ms avg_case_ms=9.83 avg_simplify_ms=2.78, sum@0+100 failed=0 elapsed=653.12ms avg_case_ms=6.53 avg_simplify_ms=2.01, product@0+100 failed=0 elapsed=601.10ms avg_case_ms=6.01 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=396.18ms avg_case_ms=7.92 avg_simplify_ms=2.40, sum@700+100 failed=0 elapsed=227.23ms avg_case_ms=2.27 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.82ms median_wire=16.89ms median_wall=64.70ms, sum@0+100 #173 sum runs=3 median_simplify=15.25ms median_wire=15.29ms median_wall=58.17ms, difference@0+50 #174 difference runs=3 median_simplify=15.20ms median_wire=15.24ms median_wall=58.29ms, product@0+100 #175 product runs=3 median_simplify=15.43ms median_wire=15.49ms median_wall=58.47ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.31ms median_wire=12.39ms median_wall=47.71ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.51s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
