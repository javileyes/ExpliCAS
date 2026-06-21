# Engine Improvement Scorecard

- Generated: 2026-06-21T09:52:06.176440+00:00
- Git branch: main
- Git commit: `23f2dbe57da12eb555d518a04ab9f2e6f22be89d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=799.33ms avg_case_ms=7.99 simplify=230.53ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=704.48ms avg_case_ms=3.52 simplify=238.73ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=483.05ms avg_case_ms=4.83 simplify=139.53ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=333.50ms avg_case_ms=6.67 simplify=105.94ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=238.73ms avg_simplify_ms=1.19 wall=704.48ms, shifted_quotient simplify=230.53ms avg_simplify_ms=2.31 wall=799.33ms, product simplify=139.53ms avg_simplify_ms=1.40 wall=483.05ms, difference simplify=105.94ms avg_simplify_ms=2.12 wall=333.50ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=799.33ms avg_case_ms=7.99 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=509.30ms avg_case_ms=5.09 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=483.05ms avg_case_ms=4.83 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=333.50ms avg_case_ms=6.67 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=195.18ms avg_case_ms=1.95 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.90ms median_wire=12.97ms median_wall=49.29ms, sum@0+100 #173 sum runs=3 median_simplify=11.90ms median_wire=11.95ms median_wall=44.78ms, difference@0+50 #174 difference runs=3 median_simplify=11.65ms median_wire=11.70ms median_wall=44.18ms, product@0+100 #175 product runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.04ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.56ms median_wire=10.63ms median_wall=40.53ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
