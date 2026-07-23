# Engine Improvement Scorecard

- Generated: 2026-07-23T15:52:53.709350+00:00
- Git branch: main
- Git commit: `fb412cda60bd3dd220df86cbbf4f83cd517cb89e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=995.08ms avg_case_ms=9.95 simplify=278.09ms avg_simplify_ms=2.78, sum total=200 failed=0 elapsed=861.93ms avg_case_ms=4.31 simplify=280.54ms avg_simplify_ms=1.40, product total=100 failed=0 elapsed=603.93ms avg_case_ms=6.04 simplify=173.55ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=402.58ms avg_case_ms=8.05 simplify=122.24ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=280.54ms avg_simplify_ms=1.40 wall=861.93ms, shifted_quotient simplify=278.09ms avg_simplify_ms=2.78 wall=995.08ms, product simplify=173.55ms avg_simplify_ms=1.74 wall=603.93ms, difference simplify=122.24ms avg_simplify_ms=2.44 wall=402.58ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=995.08ms avg_case_ms=9.95 avg_simplify_ms=2.78, sum@0+100 failed=0 elapsed=629.85ms avg_case_ms=6.30 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=603.93ms avg_case_ms=6.04 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=402.58ms avg_case_ms=8.05 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=232.08ms avg_case_ms=2.32 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.88ms median_wire=16.95ms median_wall=64.20ms, product@0+100 #175 product runs=3 median_simplify=15.10ms median_wire=15.15ms median_wall=58.80ms, sum@0+100 #173 sum runs=3 median_simplify=15.38ms median_wire=15.43ms median_wall=59.17ms, difference@0+50 #174 difference runs=3 median_simplify=16.80ms median_wire=16.86ms median_wall=59.75ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.01ms median_wire=13.10ms median_wall=49.71ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.68s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
