# Engine Improvement Scorecard

- Generated: 2026-07-17T21:16:51.453528+00:00
- Git branch: main
- Git commit: `9a481f9f3e84e434040932be7022fe623571e022`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=993.10ms avg_case_ms=9.93 simplify=276.63ms avg_simplify_ms=2.77, sum total=200 failed=0 elapsed=876.91ms avg_case_ms=4.38 simplify=292.93ms avg_simplify_ms=1.46, product total=100 failed=0 elapsed=616.28ms avg_case_ms=6.16 simplify=175.99ms avg_simplify_ms=1.76, difference total=50 failed=0 elapsed=397.78ms avg_case_ms=7.96 simplify=121.48ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=292.93ms avg_simplify_ms=1.46 wall=876.91ms, shifted_quotient simplify=276.63ms avg_simplify_ms=2.77 wall=993.10ms, product simplify=175.99ms avg_simplify_ms=1.76 wall=616.28ms, difference simplify=121.48ms avg_simplify_ms=2.43 wall=397.78ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=993.10ms avg_case_ms=9.93 avg_simplify_ms=2.77, sum@0+100 failed=0 elapsed=645.77ms avg_case_ms=6.46 avg_simplify_ms=2.09, product@0+100 failed=0 elapsed=616.28ms avg_case_ms=6.16 avg_simplify_ms=1.76, difference@0+50 failed=0 elapsed=397.78ms avg_case_ms=7.96 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=231.14ms avg_case_ms=2.31 avg_simplify_ms=0.84
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.28ms median_wire=15.33ms median_wall=58.32ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.00ms median_wire=17.07ms median_wall=65.23ms, product@0+100 #175 product runs=3 median_simplify=15.60ms median_wire=15.65ms median_wall=59.63ms, difference@0+50 #174 difference runs=3 median_simplify=15.19ms median_wire=15.24ms median_wall=58.42ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.01ms median_wire=13.08ms median_wall=49.90ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.60s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
