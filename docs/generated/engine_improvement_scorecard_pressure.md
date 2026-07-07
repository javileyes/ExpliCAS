# Engine Improvement Scorecard

- Generated: 2026-07-07T20:02:57.360007+00:00
- Git branch: main
- Git commit: `e54cc532b3ea219badde38df3bbf2535dfdf857b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=946.31ms avg_case_ms=9.46 simplify=263.44ms avg_simplify_ms=2.63, sum total=200 failed=0 elapsed=848.80ms avg_case_ms=4.24 simplify=275.88ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=591.41ms avg_case_ms=5.91 simplify=169.07ms avg_simplify_ms=1.69, difference total=50 failed=0 elapsed=397.68ms avg_case_ms=7.95 simplify=120.93ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=275.88ms avg_simplify_ms=1.38 wall=848.80ms, shifted_quotient simplify=263.44ms avg_simplify_ms=2.63 wall=946.31ms, product simplify=169.07ms avg_simplify_ms=1.69 wall=591.41ms, difference simplify=120.93ms avg_simplify_ms=2.42 wall=397.68ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=946.31ms avg_case_ms=9.46 avg_simplify_ms=2.63, sum@0+100 failed=0 elapsed=621.71ms avg_case_ms=6.22 avg_simplify_ms=1.94, product@0+100 failed=0 elapsed=591.41ms avg_case_ms=5.91 avg_simplify_ms=1.69, difference@0+50 failed=0 elapsed=397.68ms avg_case_ms=7.95 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=227.08ms avg_case_ms=2.27 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.66ms median_wire=16.73ms median_wall=64.13ms, difference@0+50 #174 difference runs=3 median_simplify=15.08ms median_wire=15.13ms median_wall=57.68ms, sum@0+100 #173 sum runs=3 median_simplify=14.93ms median_wire=14.99ms median_wall=57.05ms, product@0+100 #175 product runs=3 median_simplify=15.11ms median_wire=15.17ms median_wall=58.12ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.56ms median_wire=12.63ms median_wall=47.83ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.78s | passed=450 failed=0 total=450 avg_case=6.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.89s | passed=1 failed=0 |
