# cas_android_ffi

Android JNI bridge for ExpliCAS engine. Provides a single function to evaluate expressions and return JSON results.

## Building for Android

### Prerequisites

```bash
# Install Rust Android targets
rustup target add aarch64-linux-android  # arm64-v8a
rustup target add x86_64-linux-android   # x86_64 (emulator)

# Install cargo-ndk (recommended)
cargo install cargo-ndk
```

### Compile .so files

From the workspace root:

```bash
# For arm64-v8a (tablets, phones)
cargo ndk -t arm64-v8a -o android/app/src/main/jniLibs build --release -p cas_android_ffi

# For x86_64 (emulator)
cargo ndk -t x86_64 -o android/app/src/main/jniLibs build --release -p cas_android_ffi
```

This produces:
- `android/app/src/main/jniLibs/arm64-v8a/libcas_android_ffi.so`
- `android/app/src/main/jniLibs/x86_64/libcas_android_ffi.so`

## JNI Functions

### Kotlin usage

```kotlin
package com.sigma.cas

object CasNative {
    init {
        System.loadLibrary("cas_android_ffi")
    }
    
    external fun abiVersion(): Int
    external fun evalJson(expr: String, optsJson: String): String
}
```

### Functions

| Function | Parameters | Returns |
|----------|------------|---------|
| `abiVersion()` | none | `Int` (currently 1) |
| `evalJson(expr, optsJson)` | expression, options JSON | JSON response |

### Options JSON

```json
{
  "budget": {
    "preset": "cli",
    "mode": "best-effort"
  },
  "pretty": true
}
```

| Field | Values | Default |
|-------|--------|---------|
| `budget.preset` | `small`, `cli`, `unlimited` | `cli` |
| `budget.mode` | `strict`, `best-effort` | `best-effort` |
| `pretty` | boolean | `true` |

## Response JSON (schema_version: 1)

### Success

```json
{
  "schema_version": 1,
  "ok": true,
  "input": "2+x^2/(sqrt(2)+3)",
  "result": "(x² + 2·√2 + 6) / (3 + √2)",
  "result_truncated": false,
  "steps_count": 3,
  "budget": {
    "preset": "cli",
    "mode": "best-effort"
  },
  "timings_us": {
    "parse_us": 150,
    "simplify_us": 2500,
    "total_us": 2700
  }
}
```

### Error

```json
{
  "schema_version": 1,
  "ok": false,
  "error": {
    "kind": "ParseError",
    "message": "unexpected token at position 5"
  },
  "budget": {
    "preset": "cli",
    "mode": "best-effort"
  }
}
```

### Error kinds

| Kind | Description |
|------|-------------|
| `ParseError` | Expression syntax error |
| `EvalError` | Evaluation failed (e.g., budget exceeded) |
| `InternalError` | Internal error (panic caught) |

## Kotlin Prototype

Complete Kotlin prototype files are in the `kotlin/` directory:

- `CasNative.kt` - JNI bridge class with documentation
- `MainActivity.kt` - Demo Activity with UI and coroutines
- `build.gradle.kts` - App module configuration
- `activity_main.xml` - Layout file

## Example Kotlin integration

```kotlin
import org.json.JSONObject

fun evalAndExtractResult(expr: String): String {
    val opts = """{"budget":{"preset":"cli","mode":"best-effort"}}"""
    val json = CasNative.evalJson(expr, opts)
    
    val obj = JSONObject(json)
    
    if (!obj.optBoolean("ok", true)) {
        val err = obj.optJSONObject("error")
        return "ERROR: " + (err?.optString("message") ?: "unknown")
    }
    
    return obj.optString("result", "NO_RESULT")
}

// Usage
val result = evalAndExtractResult("x^2 + 2*x + 1")
println(result)  // "(1 + x)²" or similar
```

## Safety

- All panics are caught with `catch_unwind` - never crashes the JVM
- Always returns valid JSON, even on internal errors
- Thread-safe: each call creates its own engine instance
