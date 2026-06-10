# Engine Improvement Scorecard

- Generated: 2026-06-10T18:42:51.835656+00:00
- Git branch: main
- Git commit: `12c705b388dfa05a80da4bab291f66c922f5c79d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=793.23ms avg_case_ms=7.93 simplify=225.33ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=730.25ms avg_case_ms=3.65 simplify=247.85ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=475.93ms avg_case_ms=4.76 simplify=136.98ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=332.59ms avg_case_ms=6.65 simplify=105.75ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=247.85ms avg_simplify_ms=1.24 wall=730.25ms, shifted_quotient simplify=225.33ms avg_simplify_ms=2.25 wall=793.23ms, product simplify=136.98ms avg_simplify_ms=1.37 wall=475.93ms, difference simplify=105.75ms avg_simplify_ms=2.11 wall=332.59ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=793.23ms avg_case_ms=7.93 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=519.75ms avg_case_ms=5.20 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=475.93ms avg_case_ms=4.76 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=332.59ms avg_case_ms=6.65 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=210.50ms avg_case_ms=2.11 avg_simplify_ms=0.78
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=45.15ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.75ms median_wire=12.82ms median_wall=48.77ms, difference@0+50 #174 difference runs=3 median_simplify=11.66ms median_wire=11.72ms median_wall=44.62ms, product@0+100 #175 product runs=3 median_simplify=11.44ms median_wire=11.49ms median_wall=43.86ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.58ms median_wire=11.66ms median_wall=43.01ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.48s | passed=1 failed=0 |
