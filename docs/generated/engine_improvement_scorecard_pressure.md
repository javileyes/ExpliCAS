# Engine Improvement Scorecard

- Generated: 2026-06-12T02:43:39.641545+00:00
- Git branch: main
- Git commit: `2401ec8300dac2938548fd6707e365067af010d3`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=777.96ms avg_case_ms=7.78 simplify=219.59ms avg_simplify_ms=2.20, sum total=200 failed=0 elapsed=686.94ms avg_case_ms=3.43 simplify=229.75ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=473.80ms avg_case_ms=4.74 simplify=135.42ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=327.10ms avg_case_ms=6.54 simplify=103.58ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=229.75ms avg_simplify_ms=1.15 wall=686.94ms, shifted_quotient simplify=219.59ms avg_simplify_ms=2.20 wall=777.96ms, product simplify=135.42ms avg_simplify_ms=1.35 wall=473.80ms, difference simplify=103.58ms avg_simplify_ms=2.07 wall=327.10ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=777.96ms avg_case_ms=7.78 avg_simplify_ms=2.20, sum@0+100 failed=0 elapsed=497.95ms avg_case_ms=4.98 avg_simplify_ms=1.60, product@0+100 failed=0 elapsed=473.80ms avg_case_ms=4.74 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=327.10ms avg_case_ms=6.54 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=188.99ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.99ms median_wire=13.06ms median_wall=49.62ms, sum@0+100 #173 sum runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.45ms, difference@0+50 #174 difference runs=3 median_simplify=12.34ms median_wire=12.40ms median_wall=47.72ms, product@0+100 #175 product runs=3 median_simplify=11.90ms median_wire=11.96ms median_wall=44.86ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.69ms median_wire=10.76ms median_wall=40.29ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.85s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.31s | passed=1 failed=0 |
