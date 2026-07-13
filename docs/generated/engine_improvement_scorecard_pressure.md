# Engine Improvement Scorecard

- Generated: 2026-07-13T10:18:16.793670+00:00
- Git branch: main
- Git commit: `2380fbb2beee7d913a2bdb7d6321ff542dd3667e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=941.32ms avg_case_ms=9.41 simplify=260.86ms avg_simplify_ms=2.61, sum total=200 failed=0 elapsed=853.22ms avg_case_ms=4.27 simplify=273.68ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=584.27ms avg_case_ms=5.84 simplify=165.66ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=388.57ms avg_case_ms=7.77 simplify=117.58ms avg_simplify_ms=2.35
- Engine hotspots: sum simplify=273.68ms avg_simplify_ms=1.37 wall=853.22ms, shifted_quotient simplify=260.86ms avg_simplify_ms=2.61 wall=941.32ms, product simplify=165.66ms avg_simplify_ms=1.66 wall=584.27ms, difference simplify=117.58ms avg_simplify_ms=2.35 wall=388.57ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=941.32ms avg_case_ms=9.41 avg_simplify_ms=2.61, sum@0+100 failed=0 elapsed=628.79ms avg_case_ms=6.29 avg_simplify_ms=1.94, product@0+100 failed=0 elapsed=584.27ms avg_case_ms=5.84 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=388.57ms avg_case_ms=7.77 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=224.43ms avg_case_ms=2.24 avg_simplify_ms=0.80
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=14.88ms median_wire=14.93ms median_wall=57.16ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=15.72ms median_wire=15.79ms median_wall=63.72ms, difference@0+50 #174 difference runs=3 median_simplify=15.02ms median_wire=15.07ms median_wall=58.89ms, product@0+100 #175 product runs=3 median_simplify=15.28ms median_wire=15.34ms median_wall=58.15ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.43ms median_wire=11.51ms median_wall=45.76ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.77s | passed=450 failed=0 total=450 avg_case=6.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.89s | passed=1 failed=0 |
