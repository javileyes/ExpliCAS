# ExpliCAS Android FFI - Guía de Integración

Esta guía explica cómo compilar la librería nativa `cas_android_ffi` y cómo integrarla en un proyecto Android/Kotlin.

## Índice

1. [Requisitos previos](#requisitos-previos)
2. [Compilación de la librería](#compilación-de-la-librería)
3. [Integración en Android](#integración-en-android)
4. [Uso desde Kotlin](#uso-desde-kotlin)
5. [API Reference](#api-reference)
6. [Troubleshooting](#troubleshooting)

---

## Requisitos previos

### En el sistema de desarrollo (macOS/Linux)

```bash
# 1. Instalar Rust (si no lo tienes)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Añadir target Android
rustup target add aarch64-linux-android    # arm64-v8a (dispositivos reales)
rustup target add x86_64-linux-android     # x86_64 (emulador)

# 3. Instalar cargo-ndk (simplifica la compilación)
cargo install cargo-ndk

# 4. Instalar Android NDK
# Opción A: Desde Android Studio → SDK Manager → SDK Tools → NDK
# Opción B: Desde terminal
sdkmanager "ndk;26.1.10909125"
```

### Verificar instalación

```bash
# Verificar targets
rustup target list --installed | grep android

# Verificar NDK
ls ~/Library/Android/sdk/ndk/
```

---

## Compilación de la librería

### Comando básico

Desde la raíz del workspace (`/Users/javiergimenezmoya/developer/math`):

```bash
# Para dispositivo real (Pixel 8, tablets, etc.)
cargo ndk -t arm64-v8a -o ./jniLibs build --release -p cas_android_ffi

# Para emulador x86_64
cargo ndk -t x86_64 -o ./jniLibs build --release -p cas_android_ffi

# Ambos targets a la vez
cargo ndk -t arm64-v8a -t x86_64 -o ./jniLibs build --release -p cas_android_ffi
```

### Resultado

```
jniLibs/
├── arm64-v8a/
│   └── libcas_android_ffi.so    (~2-4 MB)
└── x86_64/
    └── libcas_android_ffi.so    (~2-4 MB)
```

### Makefile target (recomendado)

Añadir a tu `Makefile`:

```makefile
.PHONY: android

android: ## Build Android FFI library for arm64-v8a
	cargo ndk -t arm64-v8a -o ./jniLibs build --release -p cas_android_ffi
	@echo "Library built: jniLibs/arm64-v8a/libcas_android_ffi.so"
```

---

## Integración en Android

### 1. Estructura del proyecto Android

```
app/
├── src/main/
│   ├── java/es/javiergimenez/explicas/
│   │   ├── CasNative.kt           # JNI bridge
│   │   └── MainActivity.kt        # UI
│   ├── jniLibs/
│   │   └── arm64-v8a/
│   │       └── libcas_android_ffi.so  # ← Copiar aquí
│   └── res/layout/
│       ├── activity_main.xml
│       └── item_step.xml
└── build.gradle.kts
```

### 2. Configurar build.gradle.kts

```kotlin
android {
    defaultConfig {
        ndk {
            // Solo incluir el ABI que necesitas
            abiFilters += listOf("arm64-v8a")
            // Para desarrollo con emulador, añadir:
            // abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }
}

dependencies {
    // Coroutines (para llamadas en background)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")
    
    // RecyclerView (para mostrar steps)
    implementation("androidx.recyclerview:recyclerview:1.3.2")
}
```

### 3. Copiar librería nativa

```bash
# Desde la raíz del proyecto Rust
cp jniLibs/arm64-v8a/libcas_android_ffi.so \
   /path/to/android/app/src/main/jniLibs/arm64-v8a/
```

---

## Uso desde Kotlin

### 1. Crear el bridge JNI

```kotlin
// CasNative.kt
package es.javiergimenez.explicas

object CasNative {
    init {
        System.loadLibrary("cas_android_ffi")
    }

    /** Devuelve versión ABI para diagnóstico */
    external fun abiVersion(): Int

    /** Evalúa expresión y devuelve JSON */
    external fun evalJson(expr: String, optsJson: String): String
}
```

### 2. Llamar desde la UI (con coroutines)

```kotlin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

suspend fun evaluate(expr: String): String {
    return withContext(Dispatchers.Default) {
        val opts = """{"budget":{"preset":"cli","mode":"best-effort"}}"""
        val json = CasNative.evalJson(expr, opts)
        
        val obj = JSONObject(json)
        if (obj.optBoolean("ok", true)) {
            obj.optString("result", "NO_RESULT")
        } else {
            val error = obj.optJSONObject("error")
            "ERROR: ${error?.optString("message")}"
        }
    }
}
```

### 3. Ejemplo completo con steps

```kotlin
data class EvalResponse(
    val ok: Boolean,
    val result: String?,
    val steps: List<Step>,
    val error: String?
)

data class Step(
    val index: Int,
    val rule: String,
    val description: String,
    val before: String?,
    val after: String?,
    val importance: String
)

fun parseResponse(json: String): EvalResponse {
    val obj = JSONObject(json)
    val ok = obj.optBoolean("ok", true)
    
    if (!ok) {
        val error = obj.optJSONObject("error")
        return EvalResponse(false, null, emptyList(), error?.optString("message"))
    }
    
    val result = obj.optString("result")
    val stepsArray = obj.optJSONArray("steps") ?: JSONArray()
    val steps = (0 until stepsArray.length()).map { i ->
        val s = stepsArray.getJSONObject(i)
        Step(
            index = s.optInt("index"),
            rule = s.optString("rule"),
            description = s.optString("description"),
            before = s.optString("before", null),
            after = s.optString("after", null),
            importance = s.optString("importance", "medium")
        )
    }
    
    return EvalResponse(true, result, steps, null)
}
```

---

## API Reference

### Funciones JNI

| Función | Signature | Descripción |
|---------|-----------|-------------|
| `abiVersion` | `() -> Int` | Devuelve versión ABI (actualmente 1) |
| `evalJson` | `(expr, opts) -> String` | Evalúa expresión, devuelve JSON |

### Options JSON

```json
{
  "budget": {
    "preset": "cli",       // "small" | "cli" | "unlimited"
    "mode": "best-effort"  // "strict" | "best-effort"
  }
}
```

### Response JSON (schema_version: 1)

#### Success

```json
{
  "schema_version": 1,
  "ok": true,
  "input": "x^2 + 2*x + 1",
  "result": "(1 + x)²",
  "steps_count": 2,
  "steps": [
    {
      "index": 1,
      "rule": "Factor Perfect Square",
      "description": "x² + 2x + 1 = (x + 1)²",
      "before": "x² + 2·x + 1",
      "after": "(1 + x)²",
      "importance": "high"
    }
  ],
  "budget": { "preset": "cli", "mode": "best-effort" },
  "timings_us": { "parse_us": 150, "simplify_us": 2500, "total_us": 2700 }
}
```

#### Error

```json
{
  "schema_version": 1,
  "ok": false,
  "error": { "kind": "ParseError", "message": "unexpected token" },
  "budget": { "preset": "cli", "mode": "best-effort" }
}
```

### Importance levels

| Level | Significado | Mostrar por defecto |
|-------|-------------|---------------------|
| `trivial` | x + 0 → x | No |
| `low` | Combinar constantes | No |
| `medium` | Transformaciones algebraicas | **Sí** |
| `high` | Factor, expand, integrate | **Sí** |

---

## Troubleshooting

### Error: "libcas_android_ffi.so not found"

1. Verificar que el `.so` está en la carpeta correcta:
   ```
   app/src/main/jniLibs/arm64-v8a/libcas_android_ffi.so
   ```

2. Verificar nombre exacto (sin `lib` en el código):
   ```kotlin
   System.loadLibrary("cas_android_ffi")  // ✓ Correcto
   System.loadLibrary("libcas_android_ffi.so")  // ✗ Incorrecto
   ```

3. Verificar ABI en build.gradle:
   ```kotlin
   ndk { abiFilters += "arm64-v8a" }
   ```

### Error: "UnsatisfiedLinkError"

Los nombres de las funciones JNI deben coincidir exactamente con el package:

```
Package: es.javiergimenez.explicas
Función Rust: Java_es_javiergimenez_explicas_CasNative_evalJson
```

Si cambias el package, regenera el `.so` con los nombres correctos.

### App se congela al evaluar

Asegúrate de llamar a `evalJson` fuera del hilo UI:

```kotlin
lifecycleScope.launch(Dispatchers.Default) {  // ← Background thread
    val result = CasNative.evalJson(expr, opts)
    withContext(Dispatchers.Main) {  // ← Volver a UI
        textView.text = result
    }
}
```

### APK muy grande

El `.so` release es ~2-4 MB. Para reducir:

1. Usar solo el ABI necesario (arm64-v8a para producción)
2. Considerar App Bundle (.aab) que separa por ABI

---

## Archivos de referencia

Los siguientes archivos están disponibles en `android_app/`:

| Archivo | Descripción |
|---------|-------------|
| `CasNative.kt` | Bridge JNI con documentación |
| `MainActivity.kt` | Activity completa con RecyclerView para steps |
| `activity_main.xml` | Layout principal |
| `item_step.xml` | Layout de cada step |
| `circle_background.xml` | Drawable para índice de step |
| `build.gradle.kts` | Configuración Gradle de referencia |
