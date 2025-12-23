package com.sigma.cas

/**
 * JNI bridge to ExpliCAS native library.
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
     * @return ABI version (currently 1)
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
     *     "preset": "cli",     // "small", "cli", or "unlimited"
     *     "mode": "best-effort" // "strict" or "best-effort"
     *   },
     *   "pretty": true
     * }
     * ```
     *
     * Response on success:
     * ```json
     * {
     *   "schema_version": 1,
     *   "ok": true,
     *   "input": "x^2+1",
     *   "result": "1 + x²",
     *   "budget": { "preset": "cli", "mode": "best-effort" },
     *   "timings_us": { "parse_us": 100, "simplify_us": 500, "total_us": 600 }
     * }
     * ```
     *
     * Response on error:
     * ```json
     * {
     *   "schema_version": 1,
     *   "ok": false,
     *   "error": { "kind": "ParseError", "message": "..." },
     *   "budget": { "preset": "cli", "mode": "best-effort" }
     * }
     * ```
     */
    external fun evalJson(expr: String, optsJson: String): String
}
