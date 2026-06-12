# Engine Improvement Scorecard

- Generated: 2026-06-12T07:45:59.958234+00:00
- Git branch: main
- Git commit: `0a9baa4527fc4cc784c6860f4dd40d5af5698129`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=779.58ms avg_case_ms=7.80 simplify=221.20ms avg_simplify_ms=2.21, sum total=200 failed=0 elapsed=692.07ms avg_case_ms=3.46 simplify=232.13ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=486.20ms avg_case_ms=4.86 simplify=139.00ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=329.07ms avg_case_ms=6.58 simplify=104.86ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=232.13ms avg_simplify_ms=1.16 wall=692.07ms, shifted_quotient simplify=221.20ms avg_simplify_ms=2.21 wall=779.58ms, product simplify=139.00ms avg_simplify_ms=1.39 wall=486.20ms, difference simplify=104.86ms avg_simplify_ms=2.10 wall=329.07ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=779.58ms avg_case_ms=7.80 avg_simplify_ms=2.21, sum@0+100 failed=0 elapsed=500.62ms avg_case_ms=5.01 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=486.20ms avg_case_ms=4.86 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=329.07ms avg_case_ms=6.58 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=191.45ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.84ms median_wire=12.91ms median_wall=48.93ms, product@0+100 #175 product runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.37ms, sum@0+100 #173 sum runs=3 median_simplify=11.86ms median_wire=11.92ms median_wall=45.28ms, difference@0+50 #174 difference runs=3 median_simplify=11.82ms median_wire=11.87ms median_wall=45.03ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.60ms median_wire=10.67ms median_wall=40.16ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.11s | passed=1 failed=0 |
