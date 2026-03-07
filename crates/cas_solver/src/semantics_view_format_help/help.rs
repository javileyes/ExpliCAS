pub fn semantics_help_message() -> &'static str {
    r#"Semantics: Control evaluation semantics

Usage:
  semantics                    Show current settings
  semantics set <axis> <val>   Set one axis
  semantics set k=v k=v ...    Set multiple axes

Axes:
  domain      strict | generic | assume
              strict:  No domain assumptions (x/x stays x/x)
              generic: Classic CAS 'almost everywhere' algebra
              assume:  Use assumptions with warnings

  value       real | complex
              real:    ℝ only (sqrt(-1) undefined)
              complex: ℂ enabled (sqrt(-1) = i)

  branch      principal
              (only active when value=complex)

  inv_trig    strict | principal
              strict:    arctan(tan(x)) unchanged
              principal: arctan(tan(x)) → x with warning

  const_fold  off | safe
              off:  No constant folding
              safe: Fold literals (2^3 → 8)

  assume_scope real | wildcard
              real:     Assume for ℝ, error if ℂ needed
              wildcard: Assume for ℝ, residual+warning if ℂ needed
              (only active when domain_mode=assume)

  requires    essential | all
              essential: Show only requires whose witness was consumed
              all:       Show all requires including implicit ones

Examples:
  semantics set domain strict
  semantics set value complex inv_trig principal
  semantics set domain=strict value=complex
  semantics set assume_scope wildcard

Presets:
  semantics preset              List available presets
  semantics preset <name>       Apply a preset
  semantics preset help <name>  Show preset details"#
}
