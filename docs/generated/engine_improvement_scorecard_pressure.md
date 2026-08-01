# Engine Improvement Scorecard

- Generated: 2026-08-01T23:06:21.320027+00:00
- Git branch: main
- Git commit: `0655aa133b61436ae10ca304580434704c648787`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=391

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=985.56ms avg_case_ms=9.86 simplify=275.55ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=854.95ms avg_case_ms=4.27 simplify=273.86ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=607.78ms avg_case_ms=6.08 simplify=174.55ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=397.72ms avg_case_ms=7.95 simplify=116.81ms avg_simplify_ms=2.34
- Engine hotspots: shifted_quotient simplify=275.55ms avg_simplify_ms=2.76 wall=985.56ms, sum simplify=273.86ms avg_simplify_ms=1.37 wall=854.95ms, product simplify=174.55ms avg_simplify_ms=1.75 wall=607.78ms, difference simplify=116.81ms avg_simplify_ms=2.34 wall=397.72ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=985.56ms avg_case_ms=9.86 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=621.84ms avg_case_ms=6.22 avg_simplify_ms=1.90, product@0+100 failed=0 elapsed=607.78ms avg_case_ms=6.08 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=397.72ms avg_case_ms=7.95 avg_simplify_ms=2.34, sum@700+100 failed=0 elapsed=233.10ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.65ms median_wire=16.72ms median_wall=64.49ms, product@0+100 #175 product runs=3 median_simplify=14.83ms median_wire=14.88ms median_wall=57.08ms, sum@0+100 #173 sum runs=3 median_simplify=14.98ms median_wire=15.03ms median_wall=56.81ms, difference@0+50 #174 difference runs=3 median_simplify=15.20ms median_wire=15.25ms median_wall=57.68ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.86ms median_wire=12.93ms median_wall=49.15ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.85s | passed=450 failed=0 total=450 avg_case=6.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.07s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.75s | passed=1 failed=0 |
