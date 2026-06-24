# Engine Improvement Scorecard

- Generated: 2026-06-24T16:06:33.704094+00:00
- Git branch: main
- Git commit: `900600ba06b701a16140cb9e04972e2b84567386`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=797.04ms avg_case_ms=7.97 simplify=228.98ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=702.61ms avg_case_ms=3.51 simplify=237.80ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=479.81ms avg_case_ms=4.80 simplify=139.06ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=338.52ms avg_case_ms=6.77 simplify=110.87ms avg_simplify_ms=2.22
- Engine hotspots: sum simplify=237.80ms avg_simplify_ms=1.19 wall=702.61ms, shifted_quotient simplify=228.98ms avg_simplify_ms=2.29 wall=797.04ms, product simplify=139.06ms avg_simplify_ms=1.39 wall=479.81ms, difference simplify=110.87ms avg_simplify_ms=2.22 wall=338.52ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=797.04ms avg_case_ms=7.97 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=508.90ms avg_case_ms=5.09 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=479.81ms avg_case_ms=4.80 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=338.52ms avg_case_ms=6.77 avg_simplify_ms=2.22, sum@700+100 failed=0 elapsed=193.71ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.93ms median_wire=13.00ms median_wall=49.53ms, difference@0+50 #174 difference runs=3 median_simplify=11.76ms median_wire=11.82ms median_wall=44.36ms, sum@0+100 #173 sum runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=44.24ms, product@0+100 #175 product runs=3 median_simplify=11.57ms median_wire=11.63ms median_wall=44.30ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.76ms median_wire=10.83ms median_wall=40.59ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
