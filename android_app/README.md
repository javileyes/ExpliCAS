# Android Prototype Reference

Reference files for integrating ExpliCAS engine into Android.

## Files

- `CasNative.kt` - JNI bridge class
- `MainActivity.kt` - Demo Activity with UI
- `build.gradle.kts` - App module configuration

## Usage

1. Build the native library:
```bash
cargo ndk -t arm64-v8a -o app/src/main/jniLibs build --release -p cas_android_ffi
```

2. Copy Kotlin files to your Android project
3. Update package name if different from `es.javiergimenez.explicas`
