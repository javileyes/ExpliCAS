# Engine Improvement Scorecard

- Generated: 2026-06-15T11:50:07.891513+00:00
- Git branch: main
- Git commit: `cfca28d12265b9a6975f1d81f3dc49e8fa513766`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=793.32ms avg_case_ms=7.93 simplify=226.28ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=703.34ms avg_case_ms=3.52 simplify=235.73ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=480.03ms avg_case_ms=4.80 simplify=138.17ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=330.38ms avg_case_ms=6.61 simplify=105.29ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=235.73ms avg_simplify_ms=1.18 wall=703.34ms, shifted_quotient simplify=226.28ms avg_simplify_ms=2.26 wall=793.32ms, product simplify=138.17ms avg_simplify_ms=1.38 wall=480.03ms, difference simplify=105.29ms avg_simplify_ms=2.11 wall=330.38ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=793.32ms avg_case_ms=7.93 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=507.74ms avg_case_ms=5.08 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=480.03ms avg_case_ms=4.80 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=330.38ms avg_case_ms=6.61 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=195.60ms avg_case_ms=1.96 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.03ms median_wire=13.11ms median_wall=49.95ms, difference@0+50 #174 difference runs=3 median_simplify=11.87ms median_wire=11.92ms median_wall=45.17ms, sum@0+100 #173 sum runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=45.12ms, product@0+100 #175 product runs=3 median_simplify=11.86ms median_wire=11.91ms median_wall=44.84ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.90ms median_wire=10.98ms median_wall=41.31ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.85s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
