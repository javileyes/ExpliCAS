# Engine Improvement Scorecard

- Generated: 2026-07-10T17:56:53.962385+00:00
- Git branch: main
- Git commit: `be5ed3ace258a82e81e7c1762fe7ebde87221966`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=945.95ms avg_case_ms=9.46 simplify=264.30ms avg_simplify_ms=2.64, sum total=200 failed=0 elapsed=840.81ms avg_case_ms=4.20 simplify=270.36ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=581.67ms avg_case_ms=5.82 simplify=165.65ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=390.69ms avg_case_ms=7.81 simplify=118.09ms avg_simplify_ms=2.36
- Engine hotspots: sum simplify=270.36ms avg_simplify_ms=1.35 wall=840.81ms, shifted_quotient simplify=264.30ms avg_simplify_ms=2.64 wall=945.95ms, product simplify=165.65ms avg_simplify_ms=1.66 wall=581.67ms, difference simplify=118.09ms avg_simplify_ms=2.36 wall=390.69ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=945.95ms avg_case_ms=9.46 avg_simplify_ms=2.64, sum@0+100 failed=0 elapsed=612.66ms avg_case_ms=6.13 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=581.67ms avg_case_ms=5.82 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=390.69ms avg_case_ms=7.81 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=228.15ms avg_case_ms=2.28 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.48ms median_wire=16.55ms median_wall=63.58ms, sum@0+100 #173 sum runs=3 median_simplify=14.80ms median_wire=14.84ms median_wall=57.34ms, product@0+100 #175 product runs=3 median_simplify=14.68ms median_wire=14.73ms median_wall=56.32ms, difference@0+50 #174 difference runs=3 median_simplify=14.70ms median_wire=14.75ms median_wall=56.61ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.62ms median_wire=12.69ms median_wall=48.12ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.76s | passed=450 failed=0 total=450 avg_case=6.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.03s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
