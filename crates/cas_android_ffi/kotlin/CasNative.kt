package es.javiergimenez.explicas

/**
 * JNI bridge to ExpliCAS native library.
 *
 * Schema version: 1
 * ABI version: 2
 *
 * Usage:
 * ```kotlin
 * val result = CasNative.evalJson("x^2+1", """{"budget":{"preset":"cli"}}""")
 * val version = CasNative.abiVersion()
 * ```
 */
object CasNative {
    init {
        System.loadLibrary("cas_android_ffi")
    }

    /**
     * Returns the ABI version of the native library.
     * Use for diagnostics and version mismatch detection.
     *
     * @return ABI version (currently 2)
     */
    external fun abiVersion(): Int

    /**
     * Evaluate a mathematical expression.
     *
     * @param expr Expression string (e.g., "x^2 + 2*x + 1")
     * @param optsJson Options JSON with budget and settings
     * @return JSON response with schema_version: 1
     *
     * Example optsJson:
     * ```json
     * {
     *   "budget": {
     *     "preset": "cli",        // "small", "cli", "unlimited"
     *     "mode": "best-effort"   // "strict" or "best-effort"
     *   },
     *   "steps": false,
     *   "pretty": false
     * }
     * ```
     *
     * Response on success:
     * ```json
     * {
     *   "schema_version": 1,
     *   "ok": true,
     *   "result": "x² + 1",
     *   "budget": { "preset": "cli", "mode": "best-effort" },
     *   "steps": [],
     *   "warnings": []
     * }
     * ```
     *
     * Response on error:
     * ```json
     * {
     *   "schema_version": 1,
     *   "ok": false,
     *   "error": {
     *     "kind": "ParseError",
     *     "code": "E_PARSE",
     *     "message": "unexpected token"
     *   },
     *   "budget": { "preset": "cli", "mode": "best-effort" }
     * }
     * ```
     *
     * See docs/JSON_API_SPEC.md for full schema.
     */
    external fun evalJson(expr: String, optsJson: String): String
}
