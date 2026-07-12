# Engine Improvement Scorecard

- Generated: 2026-07-12T22:36:00.635957+00:00
- Git branch: main
- Git commit: `f24b8f4ebf2140ddfe9dab5b82815f4bc16c82c3`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=949.04ms avg_case_ms=9.49 simplify=264.46ms avg_simplify_ms=2.64, sum total=200 failed=0 elapsed=837.04ms avg_case_ms=4.19 simplify=269.09ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=584.62ms avg_case_ms=5.85 simplify=166.75ms avg_simplify_ms=1.67, difference total=50 failed=0 elapsed=393.00ms avg_case_ms=7.86 simplify=119.00ms avg_simplify_ms=2.38
- Engine hotspots: sum simplify=269.09ms avg_simplify_ms=1.35 wall=837.04ms, shifted_quotient simplify=264.46ms avg_simplify_ms=2.64 wall=949.04ms, product simplify=166.75ms avg_simplify_ms=1.67 wall=584.62ms, difference simplify=119.00ms avg_simplify_ms=2.38 wall=393.00ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=949.04ms avg_case_ms=9.49 avg_simplify_ms=2.64, sum@0+100 failed=0 elapsed=607.89ms avg_case_ms=6.08 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=584.62ms avg_case_ms=5.85 avg_simplify_ms=1.67, difference@0+50 failed=0 elapsed=393.00ms avg_case_ms=7.86 avg_simplify_ms=2.38, sum@700+100 failed=0 elapsed=229.15ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.69ms median_wire=16.76ms median_wall=63.72ms, difference@0+50 #174 difference runs=3 median_simplify=14.84ms median_wire=14.89ms median_wall=56.85ms, product@0+100 #175 product runs=3 median_simplify=15.20ms median_wire=15.25ms median_wall=57.42ms, sum@0+100 #173 sum runs=3 median_simplify=15.01ms median_wire=15.06ms median_wall=57.58ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.96ms median_wire=13.03ms median_wall=49.64ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.76s | passed=450 failed=0 total=450 avg_case=6.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.88s | passed=1 failed=0 |
