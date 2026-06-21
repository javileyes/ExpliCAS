# Engine Improvement Scorecard

- Generated: 2026-06-21T10:11:29.340164+00:00
- Git branch: main
- Git commit: `e6b311a7cd66fba6c6b53f417fbf66518664a6a1`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=787.50ms avg_case_ms=7.87 simplify=225.08ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=696.57ms avg_case_ms=3.48 simplify=235.77ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=475.45ms avg_case_ms=4.75 simplify=136.95ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=328.97ms avg_case_ms=6.58 simplify=105.10ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=235.77ms avg_simplify_ms=1.18 wall=696.57ms, shifted_quotient simplify=225.08ms avg_simplify_ms=2.25 wall=787.50ms, product simplify=136.95ms avg_simplify_ms=1.37 wall=475.45ms, difference simplify=105.10ms avg_simplify_ms=2.10 wall=328.97ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=787.50ms avg_case_ms=7.87 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=506.22ms avg_case_ms=5.06 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=475.45ms avg_case_ms=4.75 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=328.97ms avg_case_ms=6.58 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=190.34ms avg_case_ms=1.90 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.93ms median_wire=13.00ms median_wall=49.71ms, difference@0+50 #174 difference runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.20ms, sum@0+100 #173 sum runs=3 median_simplify=11.68ms median_wire=11.73ms median_wall=44.67ms, product@0+100 #175 product runs=3 median_simplify=11.70ms median_wire=11.74ms median_wall=44.67ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.66ms median_wire=10.74ms median_wall=40.65ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
